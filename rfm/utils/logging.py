import torch
import os
from rich import print as rprint
from contextlib import contextmanager
from typing import Dict
from codetiming import Timer
import time


def is_rank_0():
    """Check if current process is rank 0 (main process)."""
    # First check environment variables (most reliable for accelerate)
    if "LOCAL_RANK" in os.environ:
        return int(os.environ.get("LOCAL_RANK", 0)) == 0

    # Fallback to torch.distributed
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank() == 0
        else:
            return True  # If not distributed, consider it rank 0
    except:
        return True  # If distributed module not available, consider it rank 0


def rank_0_print(*args, **kwargs):
    """Print only if on rank 0."""
    if is_rank_0():
        rprint(*args, **kwargs)


@contextmanager
def timer(name: str, verbose: bool = False):
    """Context manager for timing operations."""
    start_time = time.time()
    if verbose:
        print(f"    Starting {name}...")
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        if verbose:
            print(f"    {name} completed in {duration:.2f}s")


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last