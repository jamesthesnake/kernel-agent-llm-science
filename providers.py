from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple

class Policy(ABC):
    @abstractmethod
    def generate(self, prompts: List[str], max_new_tokens: int, temperature: float, top_p: float) -> List[str]:
        ...

    @abstractmethod
    def logprob(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Return total token logprob for each prompt→response pair (response includes markers)."""
        ...

    @abstractmethod
    def parameters(self):
        """Return underlying trainable parameters if applicable (for local finetuning)."""
        ...

    @abstractmethod
    def step(self, loss):
        """Apply optimizer step if training."""
        ...

class FrozenRef(ABC):
    @abstractmethod
    def logprob(self, prompts: List[str], responses: List[str]) -> List[float]:
        ...
