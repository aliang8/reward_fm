import os
from rich import print as rprint


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


def rank_0_print(*args, verbose=True, **kwargs):
    """Print only if on rank 0."""
    if is_rank_0() and verbose:
        rprint(*args, **kwargs)


def banner(*lines, inner_padding=3):
    rank_0_print("\n" + "#" * 60)

    # top inner padding
    for _ in range(inner_padding):
        rank_0_print("#")

    # content lines
    for line in lines:
        rank_0_print("# " + line)

    # bottom inner padding
    for _ in range(inner_padding):
        rank_0_print("#")

    rank_0_print("#" * 60 + "\n")