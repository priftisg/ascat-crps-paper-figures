"""Exceedance-stack probabilistic product from ensemble Sobel-WGM forecasts."""

from typing import Optional, Sequence

import numpy as np
import torch

from .model import ensemble_forward

DEFAULT_TAUS = np.array(
    [4.0e-5, 6.8e-5, 1.0e-4, 1.2e-4, 1.6e-4, 2.0e-4], dtype=np.float64,
)


@torch.no_grad()
def exceedance_stack(
    model: torch.nn.Module,
    x: torch.Tensor,
    taus: Optional[Sequence[float]] = None,
    m: int = 8,
    h: float = 0.05,
    gradient_scale: float = 5.0e-4,
    members: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Soft P(S > tau) per pixel from M ensemble members.

    Returns (N, n_tau, H, W) probabilities in [0, 1].
    ``members`` may be supplied to avoid recomputing the ensemble.
    """
    if members is None:
        members = ensemble_forward(model, x, m=m)
    taus_arr = np.asarray(DEFAULT_TAUS if taus is None else taus, dtype=np.float64)
    t = torch.as_tensor(taus_arr / gradient_scale, device=members.device, dtype=members.dtype)
    # members (N,M,H,W) vs taus → (N,M,n_tau,H,W)
    p = torch.sigmoid((members.unsqueeze(2) - t[None, None, :, None, None]) / h)
    return p.mean(dim=1)


def stack_to_uint8(p_stack: torch.Tensor) -> np.ndarray:
    """Convert probability stack to compact uint8 storage (p * 255)."""
    return (p_stack.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)


def save_exceedance_npz(
    path,
    p_stack: torch.Tensor,
    taus: Sequence[float],
    extras: Optional[dict] = None,
) -> None:
    payload = {
        "taus": np.asarray(taus, dtype=np.float32),
        "p": stack_to_uint8(p_stack),
    }
    if extras:
        payload.update(extras)
    np.savez_compressed(path, **payload)
