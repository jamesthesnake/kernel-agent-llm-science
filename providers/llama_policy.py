from __future__ import annotations
import json
import re
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from .base import Policy, FrozenRef

@dataclass
class LLaMAConfig:
    """Configuration for LLaMA-based kernel generation policy"""
    model_name: str = "meta-llama/Llama-3.2-1B"
    max_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    device: str = "cuda"
    load_in_4bit: bool = True
    use_flash_attention: bool = True

class LLaMAKernelPolicy(Policy):
    """
    LLaMA-based policy for autonomous CUDA/Triton kernel generation.
    This agent learns to write high-performance GPU kernels through RL training.
    """

    def __init__(self, config: LLaMAConfig = None):
        self.config = config or LLaMAConfig()
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.training_step = 0
        self._load_model()

    def _load_model(self):
        """Load LLaMA model for kernel generation"""
        try:
            print(f"🦙 Loading {self.config.model_name} for autonomous kernel generation...")

            # Try to import transformers
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                TrainingArguments,
                AdamW
            )

            # Configure 4-bit quantization for efficiency
            if self.config.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
            else:
                quantization_config = None

            # Load tokenizer
            print("📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                padding_side="left"
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            print("🧠 Loading language model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )

            # Setup optimizer for RL training
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=1e-5,
                weight_decay=0.01
            )

            print(f"✅ LLaMA model loaded successfully!")
            print(f"   Model: {self.config.model_name}")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"   Device: {next(self.model.parameters()).device}")

        except ImportError:
            print("⚠️  Transformers not available, using mock implementation")
            self._use_mock_implementation()
        except Exception as e:
            print(f"⚠️  Failed to load LLaMA model: {e}")
            print("   Using mock implementation for demonstration")
            self._use_mock_implementation()

    def _use_mock_implementation(self):
        """Fallback to mock implementation when LLaMA not available"""
        self.model = "mock"
        self.tokenizer = "mock"
        print("🎭 Using mock LLaMA implementation for demonstration")

    def generate(self, prompts: List[str], max_new_tokens: int = 2048,
                temperature: float = 0.7, top_p: float = 0.95) -> List[str]:
        """
        Generate CUDA/Triton kernel code using LLaMA.
        This is where the AI agent creates its optimization hypotheses and implementations.
        """
        if self.model == "mock":
            return self._generate_mock_responses(prompts)

        try:
            return self._generate_with_llama(prompts, max_new_tokens, temperature, top_p)
        except Exception as e:
            print(f"⚠️  Generation failed: {e}, using fallback")
            return self._generate_mock_responses(prompts)

    def _generate_with_llama(self, prompts: List[str], max_new_tokens: int,
                           temperature: float, top_p: float) -> List[str]:
        """Generate responses using actual LLaMA model"""
        responses = []

        for prompt in prompts:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length - max_new_tokens
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )

            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            responses.append(generated_text)

        return responses

    def _generate_mock_responses(self, prompts: List[str]) -> List[str]:
        """Generate mock responses for demonstration"""
        responses = []

        for i, prompt in enumerate(prompts):
            # Extract operation type from prompt
            op_type = "stencil3x3" if "stencil" in prompt.lower() else "row_softmax"
            backend = "cuda" if "cuda" in prompt.lower() else "triton"

            # Generate kernel code based on operation
            if op_type == "stencil3x3":
                kernel_code = self._generate_cuda_stencil_kernel(i)
                plan = self._create_cuda_plan(i, kernel_code)
            else:
                kernel_code = self._generate_triton_softmax_kernel(i)
                plan = self._create_triton_plan(i, kernel_code)

            # Create full response with thinking and answer sections
            response = f"""
## (i) Performance Analysis
The {op_type} operation presents memory bandwidth challenges requiring careful optimization.
Key bottlenecks: {"Memory coalescing and shared memory usage" if op_type == "stencil3x3" else "Softmax computation and numerical stability"}.

## (ii) Algorithm Design
Proposed approach: {"Shared memory tiling with optimized block dimensions" if op_type == "stencil3x3" else "Online softmax with block-wise computation"}.
Expected improvement: {"Reduced global memory traffic" if op_type == "stencil3x3" else "Better numerical stability and parallelism"}.

## (iii) Code
Implementation using {backend} for optimal performance on target hardware.

{plan}
"""
            responses.append(response)
            self.training_step += 1

        return responses

    def _generate_cuda_stencil_kernel(self, variant: int) -> str:
        """Generate CUDA stencil kernel with variations"""
        kernels = [
            # Variant 0: Basic implementation
            '''__global__ void stencil3x3_kernel(const float* __restrict__ inp,
                                         float* __restrict__ out,
                                         int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < W && idy < H) {
        float result = 0.0f;
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                int ni = min(max(idy + di, 0), H-1);
                int nj = min(max(idx + dj, 0), W-1);
                result += inp[ni * W + nj] * 0.111f;
            }
        }
        out[idy * W + idx] = result;
    }
}''',
            # Variant 1: Shared memory optimization
            '''__global__ void stencil3x3_kernel(const float* __restrict__ inp,
                                         float* __restrict__ out,
                                         int H, int W) {
    extern __shared__ float smem[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int smem_w = blockDim.x + 2;
    int smem_h = blockDim.y + 2;

    // Load to shared memory with halo
    if (idx < W && idy < H) {
        smem[(threadIdx.y + 1) * smem_w + (threadIdx.x + 1)] = inp[idy * W + idx];
    }

    __syncthreads();

    if (idx < W && idy < H) {
        float result = 0.0f;
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                int si = threadIdx.y + 1 + di;
                int sj = threadIdx.x + 1 + dj;
                result += smem[si * smem_w + sj] * 0.111f;
            }
        }
        out[idy * W + idx] = result;
    }
}''',
            # Variant 2: Vectorized loads
            '''__global__ void stencil3x3_kernel(const float* __restrict__ inp,
                                         float* __restrict__ out,
                                         int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < W && idy < H) {
        // Unrolled stencil computation
        float result = 0.0f;

        // Top row
        if (idy > 0) {
            if (idx > 0) result += inp[(idy-1) * W + (idx-1)] * 0.111f;
            result += inp[(idy-1) * W + idx] * 0.111f;
            if (idx < W-1) result += inp[(idy-1) * W + (idx+1)] * 0.111f;
        }

        // Middle row
        if (idx > 0) result += inp[idy * W + (idx-1)] * 0.111f;
        result += inp[idy * W + idx] * 0.111f;
        if (idx < W-1) result += inp[idy * W + (idx+1)] * 0.111f;

        // Bottom row
        if (idy < H-1) {
            if (idx > 0) result += inp[(idy+1) * W + (idx-1)] * 0.111f;
            result += inp[(idy+1) * W + idx] * 0.111f;
            if (idx < W-1) result += inp[(idy+1) * W + (idx+1)] * 0.111f;
        }

        out[idy * W + idx] = result;
    }
}'''
        ]

        return kernels[variant % len(kernels)]

    def _generate_triton_softmax_kernel(self, variant: int) -> str:
        """Generate Triton softmax kernel with variations"""
        kernels = [
            '''import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)''',

            '''import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    # Online softmax algorithm for numerical stability
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    m = tl.max(row, axis=0)
    row_shifted = row - m
    exp_row = tl.exp(row_shifted)
    sum_exp = tl.sum(exp_row, axis=0)
    softmax_output = exp_row / sum_exp

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)'''
        ]

        return kernels[variant % len(kernels)]

    def _create_cuda_plan(self, variant: int, kernel_code: str) -> str:
        """Create CUDA execution plan"""
        return json.dumps({
            "experiment_id": f"autonomous_cuda_{self.training_step}_{variant}",
            "backend": "cuda",
            "op": "stencil3x3",
            "dtype": "fp32",
            "shapes": [{"H": 1024, "W": 1024}],
            "hypothesis": f"CUDA optimization variant {variant} improves performance through {'shared memory' if variant == 1 else 'vectorization' if variant == 2 else 'baseline implementation'}",
            "metrics": ["latency", "throughput"],
            "tolerance": {"stencil3x3": {"fp32": 1e-3}},
            "param_grid": {"BLOCK_X": [16, 32], "BLOCK_Y": [16, 32]},
            "iters": 50,
            "cuda_kernel": kernel_code
        }, indent=2)

    def _create_triton_plan(self, variant: int, kernel_code: str) -> str:
        """Create Triton execution plan"""
        return json.dumps({
            "experiment_id": f"autonomous_triton_{self.training_step}_{variant}",
            "backend": "triton",
            "op": "row_softmax",
            "dtype": "fp32",
            "shapes": [{"B": 64, "N": 2048}],
            "hypothesis": f"Triton softmax variant {variant} improves numerical stability and performance",
            "metrics": ["latency", "throughput"],
            "tolerance": {"row_softmax": {"fp32": 1e-5}},
            "param_grid": {"BLOCK": [64, 128, 256], "num_warps": [4, 8], "num_stages": [2, 4]},
            "triton_kernel": kernel_code
        }, indent=2)

    def logprob(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Calculate log probabilities for GRPO training"""
        if self.model == "mock":
            # Mock log probabilities for demonstration
            return [-2.3 - 0.1 * (i % 3) for i in range(len(responses))]

        try:
            return self._calculate_logprobs(prompts, responses)
        except Exception as e:
            print(f"⚠️  Logprob calculation failed: {e}")
            return [-2.5] * len(responses)

    def _calculate_logprobs(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Calculate actual log probabilities using LLaMA"""
        logprobs = []

        for prompt, response in zip(prompts, responses):
            # Tokenize prompt and response
            full_text = prompt + response
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            full_tokens = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=False)

            if torch.cuda.is_available():
                prompt_tokens = {k: v.cuda() for k, v in prompt_tokens.items()}
                full_tokens = {k: v.cuda() for k, v in full_tokens.items()}

            prompt_len = prompt_tokens['input_ids'].shape[1]
            response_tokens = full_tokens['input_ids'][:, prompt_len:]

            # Calculate log probabilities
            with torch.no_grad():
                outputs = self.model(**full_tokens, labels=full_tokens['input_ids'])
                logits = outputs.logits

                # Get log probabilities for response tokens
                response_logits = logits[:, prompt_len-1:-1, :]  # Shift by 1 for next token prediction
                response_logprobs = torch.log_softmax(response_logits, dim=-1)

                # Sum log probabilities for the response
                token_logprobs = []
                for i, token_id in enumerate(response_tokens[0]):
                    token_logprob = response_logprobs[0, i, token_id].item()
                    token_logprobs.append(token_logprob)

                avg_logprob = sum(token_logprobs) / len(token_logprobs) if token_logprobs else -10.0
                logprobs.append(avg_logprob)

        return logprobs

    def step(self, loss: torch.Tensor):
        """Training step for the policy"""
        if self.model == "mock":
            print(f"🎭 Mock training step: loss = {loss:.4f}")
            return

        try:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if self.training_step % 10 == 0:
                print(f"🧠 Training step {self.training_step}: loss = {loss:.4f}")

        except Exception as e:
            print(f"⚠️  Training step failed: {e}")

        self.training_step += 1


class LLaMAFrozenRef(FrozenRef):
    """Frozen reference policy for GRPO using LLaMA"""

    def __init__(self, base_policy: LLaMAKernelPolicy):
        self.base_policy = base_policy
        # In practice, this would be a frozen copy of the initial model

    def logprob(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Reference log probabilities (frozen)"""
        if self.base_policy.model == "mock":
            # Mock reference log probabilities (slightly lower than policy)
            return [-2.8 - 0.05 * (i % 3) for i in range(len(responses))]

        # For actual implementation, use frozen model weights
        # For demo, add small offset to current policy logprobs
        policy_logprobs = self.base_policy.logprob(prompts, responses)
        return [lp - 0.3 for lp in policy_logprobs]  # Reference is slightly worse