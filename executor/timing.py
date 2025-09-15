from __future__ import annotations
import time
from typing import Callable, List, Optional, Set, Any
import torch
import gc
from contextlib import contextmanager

class StreamDetector:
    """
    Detects and tracks all CUDA streams created during kernel execution
    to prevent stream-timing exploits
    """

    def __init__(self):
        self.initial_streams: Set[int] = set()
        self.discovered_streams: Set[int] = set()

    def capture_initial_state(self):
        """Capture the initial CUDA streams before kernel execution"""
        self.initial_streams = self._get_all_cuda_streams()
        self.discovered_streams = set()

    def discover_new_streams(self):
        """Discover any new streams created during execution"""
        current_streams = self._get_all_cuda_streams()
        self.discovered_streams = current_streams - self.initial_streams

    def wait_all_streams(self):
        """
        Wait on all CUDA streams that the candidate created,
        not just the main stream, to prevent timing exploits
        """
        # Wait on main stream first
        torch.cuda.synchronize()

        # Wait on all discovered streams
        for stream_ptr in self.discovered_streams:
            try:
                # Get stream object from pointer and synchronize
                stream = torch.cuda.Stream._from_cudaStream_t(stream_ptr)
                stream.synchronize()
            except Exception:
                # If we can't sync a specific stream, do global sync
                torch.cuda.synchronize()
                break

        # Final global synchronization to be absolutely sure
        torch.cuda.synchronize()

    def _get_all_cuda_streams(self) -> Set[int]:
        """
        Get all active CUDA streams by inspecting CUDA runtime state.
        This is a simplified implementation - in practice might need
        more sophisticated CUDA runtime introspection.
        """
        streams = set()

        # Always include the default stream
        default_stream = torch.cuda.default_stream()
        streams.add(default_stream.cuda_stream)

        # Include current stream
        current_stream = torch.cuda.current_stream()
        streams.add(current_stream.cuda_stream)

        # Try to detect additional streams through garbage collection
        # This is heuristic - real implementation might use CUDA profiling APIs
        for obj in gc.get_objects():
            if isinstance(obj, torch.cuda.Stream):
                streams.add(obj.cuda_stream)

        return streams

@contextmanager
def anti_exploit_timing(warmup_calls: int = 5):
    """
    Context manager for timing that prevents stream-timing exploits.

    Usage:
        with anti_exploit_timing(warmup=5) as timer:
            # kernel execution
            result = kernel_call()
        latency_ms = timer.get_result()
    """
    detector = StreamDetector()

    class TimingContext:
        def __init__(self):
            self.start_event = None
            self.end_event = None
            self.warmup_done = False

        def start_timing(self):
            if not self.warmup_done:
                return  # Still in warmup phase

            detector.capture_initial_state()
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()

        def end_timing(self):
            if not self.start_event:
                return

            # Discover any new streams created during execution
            detector.discover_new_streams()

            # Record end event on current stream
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.end_event.record()

            # Critical: wait on ALL streams before final synchronization
            detector.wait_all_streams()

        def get_result(self) -> float:
            if not (self.start_event and self.end_event):
                raise RuntimeError("Timing not completed properly")

            return self.start_event.elapsed_time(self.end_event)

        def mark_warmup_done(self):
            self.warmup_done = True

    context = TimingContext()

    try:
        yield context
    finally:
        # Always ensure proper cleanup
        detector.wait_all_streams()

def time_kernel_robust(func: Callable[[], Any], iters: int = 50, warmup: int = 5,
                      max_seconds: Optional[float] = None) -> float:
    """
    Robust kernel timing that prevents stream-timing exploits.

    Args:
        func: Function to time (should contain kernel call)
        iters: Number of timing iterations
        warmup: Number of warmup iterations
        max_seconds: Maximum time to spend timing

    Returns:
        Average latency in milliseconds
    """
    detector = StreamDetector()

    # Warmup phase
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    # Timing phase with exploit protection
    times = []
    start_time = time.time()

    for i in range(iters):
        if max_seconds and (time.time() - start_time) > max_seconds:
            break

        # Capture initial stream state
        detector.capture_initial_state()

        # Start timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        func()  # Execute the kernel
        end_event.record()

        # Discover new streams and wait on all of them
        detector.discover_new_streams()
        detector.wait_all_streams()

        # Get timing result
        elapsed = start_event.elapsed_time(end_event)
        times.append(elapsed)

    if not times:
        raise TimeoutError("No timing samples collected")

    return sum(times) / len(times)

def robust_stream_sync():
    """
    Robust synchronization that waits on all streams.
    Use this before ending any timing measurement.
    """
    detector = StreamDetector()
    detector.capture_initial_state()
    detector.discover_new_streams()
    detector.wait_all_streams()

class StreamAwareTimer:
    """
    Timer class that's aware of stream timing exploits and prevents them
    """

    def __init__(self):
        self.detector = StreamDetector()
        self.start_event = None
        self.end_event = None
        self.timing_active = False

    def start(self):
        """Start timing with stream detection"""
        self.detector.capture_initial_state()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        self.timing_active = True

    def stop(self):
        """Stop timing and wait on all streams"""
        if not self.timing_active:
            raise RuntimeError("Timer not started")

        self.end_event = torch.cuda.Event(enable_timing=True)
        self.end_event.record()

        # Critical: discover and wait on all streams
        self.detector.discover_new_streams()
        self.detector.wait_all_streams()

        self.timing_active = False

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        if not (self.start_event and self.end_event):
            raise RuntimeError("Timer not properly started/stopped")

        return self.start_event.elapsed_time(self.end_event)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timing_active:
            self.stop()