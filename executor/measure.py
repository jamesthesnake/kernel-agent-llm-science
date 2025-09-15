from __future__ import annotations
import time
import math
import statistics
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import torch

@dataclass
class TimingResult:
    """Result of a timing measurement"""
    median_ms: float
    mean_ms: float
    std_ms: float
    samples: List[float]
    dropped_runs: int
    variance_ok: bool

@dataclass
class RewardMeasurement:
    """Reward measurement with smoothing applied"""
    raw_speedup: float
    smoothed_speedup: float
    conservative_speedup: float
    bucket_variance: float
    needs_reverify: bool

class RobustTimer:
    """
    Implements the paper's robust timing methodology:
    - Multi-round window
    - Bucketized single-run ratios (7 buckets)
    - Drop runs with high inter-bucket variance (>0.005)
    - Use median of bucket means as reward
    - Conservative rounding toward 1.00
    """

    def __init__(self, num_buckets: int = 7, variance_threshold: float = 0.005):
        self.num_buckets = num_buckets
        self.variance_threshold = variance_threshold

    def time_with_buckets(self, func: Callable[[], None], iters: int = 50,
                         warmup: int = 5, max_seconds: Optional[float] = None) -> TimingResult:
        """
        Time function with bucketized variance control.
        Returns median of bucket means as the final measurement.
        """
        if warmup > 0:
            for _ in range(warmup):
                func()
            torch.cuda.synchronize()

        times = []
        start_time = time.time()

        for i in range(iters):
            if max_seconds and (time.time() - start_time) > max_seconds:
                break

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            func()
            end.record()
            torch.cuda.synchronize()

            elapsed = start.elapsed_time(end)
            times.append(elapsed)

        if not times:
            raise TimeoutError("No timing samples collected")

        # Bucketize the timing samples
        buckets = self._bucketize_times(times)

        # Check inter-bucket variance
        bucket_means = [statistics.mean(bucket) for bucket in buckets if bucket]
        if len(bucket_means) < 2:
            # Not enough buckets for variance check
            return TimingResult(
                median_ms=statistics.median(times),
                mean_ms=statistics.mean(times),
                std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
                samples=times,
                dropped_runs=0,
                variance_ok=True
            )

        bucket_variance = statistics.variance(bucket_means)
        variance_ok = bucket_variance <= self.variance_threshold

        if variance_ok:
            # Use median of bucket means
            result_ms = statistics.median(bucket_means)
            dropped = 0
        else:
            # Drop high-variance runs and retry with remaining
            filtered_times = self._filter_high_variance_runs(times, buckets, bucket_means)
            result_ms = statistics.median(filtered_times) if filtered_times else statistics.median(times)
            dropped = len(times) - len(filtered_times)

        return TimingResult(
            median_ms=result_ms,
            mean_ms=statistics.mean(times),
            std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
            samples=times,
            dropped_runs=dropped,
            variance_ok=variance_ok
        )

    def _bucketize_times(self, times: List[float]) -> List[List[float]]:
        """Discretize timing samples into buckets"""
        if not times:
            return []

        min_time, max_time = min(times), max(times)
        if min_time == max_time:
            return [times]  # All same time

        bucket_width = (max_time - min_time) / self.num_buckets
        buckets = [[] for _ in range(self.num_buckets)]

        for t in times:
            bucket_idx = min(int((t - min_time) / bucket_width), self.num_buckets - 1)
            buckets[bucket_idx].append(t)

        return buckets

    def _filter_high_variance_runs(self, times: List[float], buckets: List[List[float]],
                                  bucket_means: List[float]) -> List[float]:
        """Filter out runs contributing to high inter-bucket variance"""
        # Simple strategy: remove outlier buckets and keep times from remaining buckets
        if len(bucket_means) < 3:
            return times  # Not enough data to filter

        mean_of_means = statistics.mean(bucket_means)
        std_of_means = statistics.stdev(bucket_means) if len(bucket_means) > 1 else 0

        # Keep buckets within 1 standard deviation of mean
        threshold = mean_of_means + std_of_means
        filtered_times = []

        for bucket, bucket_mean in zip(buckets, bucket_means):
            if abs(bucket_mean - mean_of_means) <= threshold:
                filtered_times.extend(bucket)

        return filtered_times if filtered_times else times

class RewardSmoother:
    """
    Implements reward smoothing and conservative rounding as per paper
    """

    def smooth_reward(self, speedup: float, baseline_ms: float,
                     previous_max: Optional[float] = None) -> RewardMeasurement:
        """
        Apply reward smoothing with conservative rounding and auto-reverify triggers
        """
        if not math.isfinite(speedup) or speedup <= 0:
            return RewardMeasurement(
                raw_speedup=speedup,
                smoothed_speedup=0.0,
                conservative_speedup=0.0,
                bucket_variance=0.0,
                needs_reverify=False
            )

        # Conservative rounding toward 1.00
        conservative = self._conservative_round(speedup)

        # Check for auto re-verify conditions
        needs_reverify = (
            speedup > 3.0 or  # speedup > 3x
            (previous_max and speedup > 2.0 * previous_max)  # > 2x previous max
        )

        return RewardMeasurement(
            raw_speedup=speedup,
            smoothed_speedup=speedup,  # Could add additional smoothing here
            conservative_speedup=conservative,
            bucket_variance=0.0,  # TODO: implement if needed
            needs_reverify=needs_reverify
        )

    def _conservative_round(self, speedup: float) -> float:
        """
        Conservative rounding toward 1.00 as per paper examples:
        1.118 → 1.11, 0.992 → 1.00
        """
        if speedup < 1.0:
            # Round up to 1.00 if close
            return 1.00 if speedup >= 0.99 else round(speedup, 2)
        else:
            # Round down for speedups > 1
            return math.floor(speedup * 100) / 100

    def verify_within_tolerance(self, speedup1: float, speedup2: float,
                               tolerance: float = 0.1) -> bool:
        """Check if two speedup measurements are within tolerance (10% default)"""
        if not (math.isfinite(speedup1) and math.isfinite(speedup2)):
            return False

        if speedup1 == 0 or speedup2 == 0:
            return speedup1 == speedup2

        ratio = abs(speedup1 - speedup2) / max(speedup1, speedup2)
        return ratio <= tolerance

def time_ms(func: Callable[[], None], iters: int = 50, warmup: int = 5,
           max_seconds: Optional[float] = None) -> float:
    """
    Legacy compatibility function using robust timer
    """
    timer = RobustTimer()
    result = timer.time_with_buckets(func, iters, warmup, max_seconds)
    return result.median_ms

def smooth_group_rewards(rewards: List[float], normalize: bool = True,
                        clip_factor: float = 2.0) -> List[float]:
    """
    Smooth and normalize rewards per group (rsmooth from paper).
    Normalize per-group and clip large jumps.
    """
    if not rewards:
        return []

    # Group normalization
    if normalize and len(rewards) > 1:
        mean_reward = statistics.mean(rewards)
        std_reward = statistics.stdev(rewards)
        if std_reward > 0:
            normalized = [(r - mean_reward) / std_reward for r in rewards]
        else:
            normalized = [0.0] * len(rewards)
    else:
        normalized = rewards[:]

    # Clip large jumps
    if clip_factor > 0 and len(normalized) > 1:
        median_val = statistics.median(normalized)
        mad = statistics.median([abs(r - median_val) for r in normalized])
        if mad > 0:
            threshold = clip_factor * mad
            clipped = [max(median_val - threshold, min(median_val + threshold, r))
                      for r in normalized]
            return clipped

    return normalized