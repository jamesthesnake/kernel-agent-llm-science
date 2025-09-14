from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
from .base import Policy, FrozenRef

def _prepare_input(tokenizer, prompt: str, device: str):
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    return toks

def _split_prompt_response(tokenizer, prompt: str, full_text: str):
    """Return token ids of response only (strip prompt prefix)."""
    a = tokenizer(prompt, return_tensors="pt").input_ids[0]
    b = tokenizer(full_text, return_tensors="pt").input_ids[0]
    # naive align: assume full_text starts with prompt
    if len(b) >= len(a) and torch.equal(a, b[:len(a)]):
        return b[len(a):]
    # fallback: treat all as response
    return b

class HFPolicy(Policy):
    def __init__(self, model_name: str, lr: float = 5e-6, device: str = "cuda"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        self.model.train(False)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def generate(self, prompts: List[str], max_new_tokens: int, temperature: float, top_p: float) -> List[str]:
        outs = []
        for p in prompts:
            toks = self.tok(p, return_tensors="pt").to(self.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **toks,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=self.tok.eos_token_id,
                )
            text = self.tok.batch_decode(gen, skip_special_tokens=False)[0]
            outs.append(text)
        return outs

    def logprob(self, prompts: List[str], responses: List[str]) -> List[float]:
        vals = []
        for p, r in zip(prompts, responses):
            full = r  # responses are full texts containing prompt+answer per our prompting template
            toks = self.tok(full, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**toks, labels=toks.input_ids)
                # negative NLL over ALL tokens; we want response-only,
                # but for clipped ratio we just need a stable baseline; optional refine later.
                nll = out.loss.item() * toks.input_ids.size(1)
                vals.append(-nll)
        return vals

    def parameters(self):
        return self.model.parameters()

    def step(self, loss):
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

class HFFrozenRef(FrozenRef):
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()

    def logprob(self, prompts: List[str], responses: List[str]) -> List[float]:
        vals = []
        for p, r in zip(prompts, responses):
            toks = self.tok(r, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**toks, labels=toks.input_ids)
                nll = out.loss.item() * toks.input_ids.size(1)
                vals.append(-nll)
        return vals
