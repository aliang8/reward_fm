import torch
from dataclasses import dataclass


@dataclass
class ModelOutput:
    pref_logits: torch.Tensor | None = None
    success_logits: torch.Tensor | None = None
    progress_logits: torch.Tensor | None = None
    sim_logits: torch.Tensor | None = None

    hidden_states: torch.Tensor | None = None
