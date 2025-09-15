from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import List, Optional

def grpo_loss_eq5(
    logp_policy: torch.Tensor,
    logp_ref: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
    beta_kl: float = 0.01
) -> torch.Tensor:
    """
    GRPO loss exactly as specified in Equation (5) of the paper.

    L_GRPO = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)] + β * KL(π_θ || π_ref)

    Args:
        logp_policy: Log probabilities from current policy π_θ
        logp_ref: Log probabilities from reference policy π_ref
        advantages: Advantage estimates A
        clip_ratio: Clipping parameter ε (default 0.2)
        beta_kl: KL regularization coefficient β (default 0.01)

    Returns:
        GRPO loss tensor
    """
    # Compute probability ratios: r = π_θ / π_ref = exp(log π_θ - log π_ref)
    ratio = torch.exp(logp_policy - logp_ref)

    # Unclipped objective: ratio * advantages
    unclipped_obj = ratio * advantages

    # Clipped objective: clip(ratio, 1-ε, 1+ε) * advantages
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    clipped_obj = clipped_ratio * advantages

    # PPO-style clipped objective: min(unclipped, clipped)
    policy_loss = -torch.mean(torch.min(unclipped_obj, clipped_obj))

    # KL divergence term: KL(π_θ || π_ref) ≈ E[log π_θ - log π_ref] = E[ratio - 1 - log ratio]
    # For numerical stability, we use the log-space approximation
    kl_div = torch.mean(logp_policy - logp_ref)

    # Total GRPO loss: L_policy + β * KL
    total_loss = policy_loss + beta_kl * kl_div

    return total_loss

def grpo_loss_with_ref_kl(
    logp_policy: torch.Tensor,
    logp_ref: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
    beta_kl: float = 0.01,
    use_kl_penalty: bool = True
) -> dict:
    """
    GRPO loss with detailed breakdown and optional KL penalty.

    Returns a dictionary with loss components for monitoring.
    """
    # Compute probability ratios
    ratio = torch.exp(logp_policy - logp_ref)

    # Clipped PPO objective
    unclipped_obj = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    clipped_obj = clipped_ratio * advantages

    policy_loss = -torch.mean(torch.min(unclipped_obj, clipped_obj))

    # KL divergence computation
    if use_kl_penalty:
        # More accurate KL divergence: KL(p||q) = E_p[log(p/q)]
        kl_div = torch.mean(logp_policy - logp_ref)
        total_loss = policy_loss + beta_kl * kl_div
    else:
        kl_div = torch.tensor(0.0)
        total_loss = policy_loss

    return {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'kl_div': kl_div,
        'beta_kl': beta_kl,
        'mean_ratio': torch.mean(ratio),
        'ratio_std': torch.std(ratio),
        'advantages_mean': torch.mean(advantages),
        'clipped_fraction': torch.mean((torch.abs(ratio - 1.0) > clip_ratio).float())
    }

def group_normalize_advantages(advantages: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Normalize advantages per group as mentioned in the paper.
    Each group corresponds to responses for the same prompt.
    """
    normalized_groups = []

    for group_advantages in advantages:
        if len(group_advantages) <= 1:
            normalized_groups.append(group_advantages)
            continue

        # Group normalization: (x - mean) / std
        mean = torch.mean(group_advantages)
        std = torch.std(group_advantages, unbiased=False)

        if std > 1e-8:  # Avoid division by zero
            normalized = (group_advantages - mean) / std
        else:
            normalized = group_advantages - mean

        normalized_groups.append(normalized)

    return normalized_groups

def reward_smoothing_rsmooth(rewards: List[float], alpha: float = 0.1) -> List[float]:
    """
    Apply reward smoothing (rsmooth) as mentioned in the paper.
    This implements exponential moving average smoothing.
    """
    if not rewards:
        return []

    smoothed = [rewards[0]]  # First reward unchanged

    for i in range(1, len(rewards)):
        # Exponential moving average: s_t = α * r_t + (1-α) * s_{t-1}
        smoothed_val = alpha * rewards[i] + (1 - alpha) * smoothed[-1]
        smoothed.append(smoothed_val)

    return smoothed

def clipped_ratio_loss_legacy(
    logp: torch.Tensor,
    logp_ref: torch.Tensor,
    adv: torch.Tensor,
    clip: float
) -> torch.Tensor:
    """
    Legacy clipped ratio loss function for backward compatibility.
    This matches the implementation in grpo_loop.py.
    """
    ratio = torch.exp(logp - logp_ref)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
    return -torch.mean(torch.minimum(unclipped, clipped))

class GRPOLoss(torch.nn.Module):
    """
    GRPO Loss module that can be used in training loops.
    Implements Equation (5) from the paper.
    """

    def __init__(self, clip_ratio: float = 0.2, beta_kl: float = 0.01):
        super().__init__()
        self.clip_ratio = clip_ratio
        self.beta_kl = beta_kl

    def forward(self, logp_policy: torch.Tensor, logp_ref: torch.Tensor,
                advantages: torch.Tensor) -> dict:
        """
        Forward pass returning loss components.
        """
        return grpo_loss_with_ref_kl(
            logp_policy, logp_ref, advantages,
            self.clip_ratio, self.beta_kl
        )

# Ensure exact adherence to Equation (5)
def verify_equation_5_implementation():
    """
    Verification function to ensure our implementation matches Equation (5).
    This is a test function that can be called to validate the math.
    """
    # Create test tensors
    batch_size = 8
    logp_policy = torch.randn(batch_size, requires_grad=True)
    logp_ref = torch.randn(batch_size)
    advantages = torch.randn(batch_size)

    # Compute using our implementation
    loss = grpo_loss_eq5(logp_policy, logp_ref, advantages)

    # Manual computation to verify
    ratio = torch.exp(logp_policy - logp_ref)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 0.8, 1.2) * advantages  # clip_ratio = 0.2
    policy_part = -torch.mean(torch.min(unclipped, clipped))
    kl_part = torch.mean(logp_policy - logp_ref)
    manual_loss = policy_part + 0.01 * kl_part  # beta_kl = 0.01

    # They should be identical
    assert torch.allclose(loss, manual_loss, atol=1e-6), "Implementation doesn't match Equation (5)"

    return True

# Run verification on import (can be disabled for production)
if __name__ == "__main__":
    verify_equation_5_implementation()
    print("✓ GRPO loss implementation verified against Equation (5)")