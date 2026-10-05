"""GPU memory profiling — captures allocated and reserved at each stage."""

from contextlib import contextmanager
import torch


@contextmanager
def profile_memory(label: str):
    """Print allocated + reserved VRAM before and after a block."""
    torch.cuda.synchronize() 
    before_alloc = torch.cuda.memory_allocated() / 1024**3
    before_rsvd = torch.cuda.memory_reserved() / 1024**3

    yield

    torch.cuda.synchronize()
    after_alloc = torch.cuda.memory_allocated() / 1024**3
    after_rsvd = torch.cuda.memory_reserved() / 1024**3
    print(
        f"[{label}] alloc {before_alloc:.2f} → {after_alloc:.2f} GB | "
        f"rsvd {before_rsvd:.2f} → {after_rsvd:.2f} GB"
    )


def peak_snapshot() -> dict:
    """Peak allocated + reserved since the last reset_peak()."""
    return {
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
    }


def reset_peak() -> None:
    """Reset peak memory counters. Call right before trainer.train()."""
    torch.cuda.reset_peak_memory_stats()