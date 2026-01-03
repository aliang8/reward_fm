import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelOutput:
    pref_logits: torch.Tensor | None = None
    success_logits: torch.Tensor | None = None
    progress_logits: torch.Tensor | None = None
    sim_logits: torch.Tensor | None = None

    hidden_states: torch.Tensor | None = None


def convert_bins_to_continuous(bin_logits: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert discrete bins to continuous progress values in [0, 1] via weighted sum of bin centers."""
    num_bins = bin_logits.shape[-1]
    bin_probs = torch.softmax(bin_logits, dim=-1) if isinstance(bin_logits, torch.Tensor) else np.softmax(bin_logits, axis=-1)
    bin_centers = torch.linspace(0.0, 1.0, num_bins, device=bin_logits.device, dtype=bin_logits.dtype) if isinstance(bin_logits, torch.Tensor) else np.linspace(0.0, 1.0, num_bins)
    return (bin_probs * bin_centers).sum(dim=-1) if isinstance(bin_logits, torch.Tensor) else (bin_probs * bin_centers).sum(axis=-1)