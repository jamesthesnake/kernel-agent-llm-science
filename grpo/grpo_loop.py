from __future__ import annotations
import json, math, random, os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import torch
from grpo.markers import THINK_START, THINK_END, ANS_START, ANS_END
from grpo.reward import compute_reward, parse_plan_json_from_answer
from grpo.rewrite import rewrite_spec
from grpo.prompt_builder import ContrastivePromptBuilder
from grpo.exemplar_db import ExemplarDB
from grpo.losses import grpo_loss_with_ref_kl, group_normalize_advantages, reward_smoothing_rsmooth
from executor.measure import smooth_group_rewards
from executor.guards import GuardedRunner, enforce_restrictions
from providers.base import Policy, FrozenRef
from agents.schemas import TritonPlan, CudaPlan
from executor.trtion_exec import run as triton_exec_run
from executor.cuda_exec import run as cuda_exec_run

@dataclass
class Task:
    backend: str
    op: str
    dtype: str
    shape: Dict[str, int]

def prompt_template(system_spec: str, exemplar_db=None, task_context=None) -> str:
    """Updated prompt template using ContrastivePromptBuilder with CUDA-L1 improvements"""
    if exemplar_db and task_context:
        builder = ContrastivePromptBuilder(exemplar_db)
        return builder.build_prompt(system_spec, task_context)
    else:
        # Fallback to original template
        return f"""{system_spec}

{THINK_START}
# 1) Performance Analysis
# 2) Algorithm Design
# 3) Kernel Sketch & Meta-space
# 4) Risks (spills, banks, divergence)
{THINK_END}

{ANS_START}
# ONE RAW JSON PLAN ONLY. No prose.
{ANS_END}
"""

def run_single_plan(plan: dict, device: int, verify_device: int | None, timeout_s: float | None, vram_gb: float | None) -> Dict[str, Any] | None:
    try:
        backend = plan.get("backend")
        if backend == "triton":
            res = triton_exec_run(TritonPlan.model_validate(plan),
                                  device=device, timeout_s=timeout_s, vram_gb=vram_gb, verify_device=verify_device)
        elif backend == "cuda":
            res = cuda_exec_run(CudaPlan.model_validate(plan),
                                device=device, timeout_s=timeout_s, vram_gb=vram_gb, verify_device=verify_device)
        else:
            return None
        return res.model_dump()
    except Exception as e:
        return None

def clipped_ratio_loss(
    logp: torch.Tensor, logp_ref: torch.Tensor, adv: torch.Tensor, clip: float
) -> torch.Tensor:
    """Legacy function - use grpo_loss_with_ref_kl for new code"""
    ratio = torch.exp(logp - logp_ref)  # element-wise
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
    return -torch.mean(torch.minimum(unclipped, clipped))

def group_advantages(rewards: List[float]) -> List[float]:
    m = sum(rewards) / max(1, len(rewards))
    v = sum((r - m) ** 2 for r in rewards) / max(1, len(rewards))
    s = math.sqrt(v + 1e-8)
    return [(r - m) / s for r in rewards]

class CurriculumBuffer:
    def __init__(self, maxlen=512):
        self.buf: List[Task] = []
        self.maxlen = maxlen

    def push_if_nontrivial(self, task: Task, rewards: List[float]):
        # non-trivial if variance above small epsilon and some successes but not all
        if len(rewards) >= 2:
            var = sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards)
            if var > 1e-4:
                self.buf.append(task)
                if len(self.buf) > self.maxlen:
                    self.buf.pop(0)

    def sample(self, k: int) -> List[Task]:
        if not self.buf:
            return []
        return random.sample(self.buf, min(k, len(self.buf)))

def sample_groups(policy: Policy, tasks: List[Task], G: int, temperature: float, top_p: float,
                 exemplar_db=None) -> List[Tuple[Task, List[str], List[str]]]:
    """Return list of (task, prompts, outputs) with CUDA-L1 contrastive prompts."""
    batches = []
    for t in tasks:
        sys_spec, meta = rewrite_spec(t.op, t.shape, t.dtype, t.backend)

        # Create task context for exemplar retrieval
        task_context = {
            'op': t.op,
            'backend': t.backend,
            'dtype': t.dtype,
            'shape': t.shape
        }

        prompt = prompt_template(sys_spec, exemplar_db, task_context)
        prompts = [prompt] * G
        outs = policy.generate(prompts, max_new_tokens=2048, temperature=temperature, top_p=top_p)
        batches.append((t, prompts, outs))
    return batches

def train_grpo(
    policy: Policy,
    ref: FrozenRef,
    tasks: List[Task],
    *,
    device_index: int = 0,
    verify_device: int | None = None,
    timeout_s: float | None = 2.0,
    vram_gb: float | None = 16.0,
    groups_per_step: int = 4,
    G: int = 4,
    clip: float = 0.2,
    cur_fraction: float = 0.25,
    rng_seed: int = 42,
    outdir: str = "grpo_logs",
    use_cuda_l1: bool = True,
    beta_kl: float = 0.01
):
    os.makedirs(outdir, exist_ok=True)
    random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    cur = CurriculumBuffer()
    step = 0

    # CUDA-L1 improvements initialization
    exemplar_db = ExemplarDB(db_path=os.path.join(outdir, "exemplars.db")) if use_cuda_l1 else None
    guarded_runner = GuardedRunner() if use_cuda_l1 else None
    previous_max_speedup = 0.0

    while True:
        # Mix fresh tasks + curriculum
        fresh = random.sample(tasks, k=min(groups_per_step, len(tasks)))
        from_cur = cur.sample(k=max(0, int(cur_fraction * groups_per_step)))
        batch_tasks = (fresh + from_cur)[:groups_per_step]

        # Generate G candidates per task with CUDA-L1 contrastive prompts
        batches = sample_groups(policy, batch_tasks, G=G, temperature=0.7, top_p=0.95,
                               exemplar_db=exemplar_db)

        # Evaluate, compute rewards with CUDA-L1 improvements
        all_losses = []
        for t, prompts, outs in batches:
            rewards = []
            answers = []
            plans = []
            results = []
            violations = []

            for out in outs:
                # Enforce markers + parse answer block → JSON
                from grpo.markers import extract_marked
                parsed = extract_marked(out)
                ans = parsed.answer or ""
                answers.append(out)
                plan = None
                if parsed.ok:
                    plan = parse_plan_json_from_answer(ans)

                # CUDA-L1: Check for restrictions violations
                if plan and use_cuda_l1 and guarded_runner:
                    kernel_code = plan.get("cuda_kernel", "") or plan.get("triton_kernel", "")
                    task_violations = enforce_restrictions(kernel_code)
                    violations.append(task_violations)

                    # Set up guards for the plan
                    task_id = f"step{step}_task{len(plans)}"
                    guard_setup = guarded_runner.setup_task_guards(task_id, plan, kernel_code)

                plans.append(plan)

                res = run_single_plan(plan, device_index, verify_device, timeout_s, vram_gb) if plan else None
                results.append(res)

                r_terms, _ = compute_reward(out, res, speedup_floor=0.0)
                rewards.append(r_terms.total)

            # CUDA-L1: Apply reward smoothing
            if use_cuda_l1:
                smoothed_rewards = smooth_group_rewards(rewards, normalize=True, clip_factor=2.0)
                rewards = smoothed_rewards

            # Per-group advantages with CUDA-L1 group normalization
            if use_cuda_l1:
                advs_tensor = torch.tensor(rewards, dtype=torch.float32)
                advs_normalized = group_normalize_advantages([advs_tensor])
                advs = advs_normalized[0].tolist()
            else:
                advs = group_advantages(rewards)

            # Compute logprob ratio (policy vs ref) on responses (full text)
            logp  = torch.tensor(policy.logprob(prompts, answers), dtype=torch.float32)
            logpr = torch.tensor(ref.logprob(prompts, answers), dtype=torch.float32)
            adv   = torch.tensor(advs, dtype=torch.float32)

            # CUDA-L1: Use improved GRPO loss following Equation (5)
            if use_cuda_l1:
                loss_result = grpo_loss_with_ref_kl(logp, logpr, adv, clip_ratio=clip, beta_kl=beta_kl)
                loss = loss_result['total_loss']
            else:
                loss = clipped_ratio_loss(logp, logpr, adv, clip=clip)

            policy.step(loss)
            all_losses.append(loss.item())

            # Curriculum: push if non-trivial
            cur.push_if_nontrivial(t, rewards)

            # CUDA-L1: Update exemplar database with successful kernels
            if use_cuda_l1 and exemplar_db:
                task_context = {
                    'op': t.op,
                    'backend': t.backend,
                    'dtype': t.dtype,
                    'shape': t.shape
                }

                for i, (plan, result, reward) in enumerate(zip(plans, results, rewards)):
                    if plan and result and reward > 0:
                        speedup = None
                        if result.get("best") and result["best"].get("speedup_vs_baseline"):
                            speedup = result["best"]["speedup_vs_baseline"]
                            if speedup > previous_max_speedup:
                                previous_max_speedup = speedup

                        if speedup and speedup > 1.0:  # Only store successful speedups
                            kernel_code = plan.get("cuda_kernel", "") or plan.get("triton_kernel", "")
                            metadata = {
                                'step': step,
                                'speedup': speedup,
                                'latency_ms': result["best"].get("latency_ms"),
                                'passed': result["best"].get("passed", False)
                            }
                            exemplar_db.add_exemplar(task_context, kernel_code, speedup, metadata)

            # Logging with CUDA-L1 enhancements
            from grpo.markers import extract_marked
            logrec = {
                "step": step,
                "task": t.__dict__,
                "rewards": rewards,
                "advantages": advs,
                "loss": float(loss.item()),
                "cuda_l1_enabled": use_cuda_l1,
                "samples": [
                    {
                        "prompt_head": prompts[0][:200],
                        "markers_present": "ok" if parse_plan_json_from_answer((extract_marked(answers[i]).answer or "") or "{}") else "bad",
                        "plan_ok": plans[i] is not None,
                        "result_ok": results[i] is not None and (results[i].get("best") or {}).get("passed", False),
                        "violations": violations[i] if use_cuda_l1 and i < len(violations) else [],
                        "speedup": results[i].get("best", {}).get("speedup_vs_baseline") if results[i] else None,
                    } for i in range(len(outs))
                ]
            }

            # Add CUDA-L1 specific metrics
            if use_cuda_l1 and isinstance(loss, dict):
                logrec.update({
                    "policy_loss": float(loss_result.get('policy_loss', 0)),
                    "kl_div": float(loss_result.get('kl_div', 0)),
                    "mean_ratio": float(loss_result.get('mean_ratio', 0)),
                    "clipped_fraction": float(loss_result.get('clipped_fraction', 0))
                })
            with open(os.path.join(outdir, f"step{step:06d}_{t.op}_{t.backend}.json"), "w") as f:
                json.dump(logrec, f, indent=2)

        step += 1
        # simple print
        print(f"[GRPO] step {step} mean loss {sum(all_losses)/max(1,len(all_losses)):.4f} | groups={len(batches)}")
