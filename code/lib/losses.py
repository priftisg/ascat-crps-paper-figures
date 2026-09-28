"""Mask-aware regression, segmentation, and CRPS losses."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1.0e-6):
    return (value * mask).sum() / mask.sum().clamp_min(eps)


def _peak_weights(y: torch.Tensor, valid: torch.Tensor, peak: float = 4.0) -> torch.Tensor:
    """Emphasize high-gradient pixels (dense field is ~0 almost everywhere)."""
    return valid.squeeze(1) * (1.0 + peak * y.squeeze(1).clamp(0.0, 4.0))


def almost_fair_crps(
    members: torch.Tensor,
    y: torch.Tensor,
    valid: torch.Tensor,
    alpha: float = 0.95,
    peak: float = 4.0,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """
    Almost-fair CRPS (Lang et al. / AIFS-CRPS).

    members: (N, M, H, W); y, valid: (N, 1, H, W)
    """
    m = members.shape[1]
    skill = (members - y).abs().mean(dim=1)
    d = (members.unsqueeze(1) - members.unsqueeze(2)).abs()
    spread_naive = d.mean(dim=(1, 2)) / 2.0
    spread_fair = d.sum(dim=(1, 2)) / (2.0 * m * (m - 1))
    spread = alpha * spread_fair + (1.0 - alpha) * spread_naive
    crps = skill - spread
    w = _peak_weights(y, valid, peak=peak)
    return (crps * w).sum() / (w.sum() + eps)


def gaussian_crps(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    y: torch.Tensor,
    valid: torch.Tensor,
    peak: float = 4.0,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Closed-form Gaussian CRPS for (mu, log_sigma) heads."""
    sigma = log_sigma.exp().clamp_min(1.0e-4)
    z = (y - mu) / sigma
    pdf = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    crps = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))
    w = _peak_weights(y, valid, peak=peak)
    return (crps.squeeze(1) * w).sum() / (w.sum() + eps)


def deterministic_l1_crps(
    pred: torch.Tensor,
    y: torch.Tensor,
    valid: torch.Tensor,
    peak: float = 4.0,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """L1 = CRPS of a deterministic forecast (M = 1 rung)."""
    err = (pred - y).abs().squeeze(1)
    w = _peak_weights(y, valid, peak=peak)
    return (err * w).sum() / (w.sum() + eps)


@torch.no_grad()
def spread_skill_ratio(members: torch.Tensor, y: torch.Tensor, valid: torch.Tensor, eps=1e-6):
    """Ensemble stddev vs member-mean RMSE; target ~1."""
    mean = members.mean(dim=1, keepdim=True)
    spread = members.std(dim=1, unbiased=False)
    rmse = ((mean - y) ** 2).squeeze(1).sqrt()
    w = valid.squeeze(1)
    spread_m = (spread * w).sum() / (w.sum() + eps)
    rmse_m = (rmse * w).sum() / (w.sum() + eps)
    return float(spread_m / (rmse_m + eps)), float(spread_m), float(rmse_m)


class MaskedFocalTverskyLoss(nn.Module):
    """Focal Tversky loss; beta > alpha penalizes missed GF/CP pixels more."""

    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1.0):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

    def forward(self, logits, target, valid, pixel_weight=None):
        prob = torch.sigmoid(logits)
        weight = valid if pixel_weight is None else valid * pixel_weight
        tp = (prob * target * weight).sum()
        fp = (prob * (1.0 - target) * weight).sum()
        fn = ((1.0 - prob) * target * weight).sum()
        ti = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return (1.0 - ti).pow(self.gamma)


class Stage1MultiTaskLoss(nn.Module):
    """Weighted Huber for gradient magnitude plus Focal-Tversky/BCE for GF."""

    def __init__(self, config):
        super().__init__()
        self.grad_weight = float(config.get("stage1_gradient_loss_weight", 0.5))
        self.gf_weight = float(config.get("stage1_gf_loss_weight", 1.0))
        self.bce_weight = float(config.get("stage1_bce_loss_weight", 0.25))
        self.peak_weight = float(config.get("gradient_peak_weight", 4.0))
        self.peak_power = float(config.get("gradient_peak_power", 2.0))
        self.delta = float(config.get("huber_delta", 0.1))
        self.ft = MaskedFocalTverskyLoss(
            alpha=float(config.get("focal_tversky_alpha", 0.3)),
            beta=float(config.get("focal_tversky_beta", 0.7)),
            gamma=float(config.get("focal_tversky_gamma", 0.75)),
        )

    def forward(self, outputs, y_true):
        gradient = y_true[:, 0:1]
        gf = y_true[:, 1:2]
        valid = y_true[:, 2:3]
        point = F.huber_loss(
            outputs["gradient"], gradient, reduction="none", delta=self.delta,
        )
        weights = 1.0 + self.peak_weight * gradient.clamp(0.0, 1.0).pow(self.peak_power)
        gradient_loss = _masked_mean(point * weights, valid)
        ft_loss = self.ft(outputs["gf_logits"], gf, valid)
        bce = F.binary_cross_entropy_with_logits(
            outputs["gf_logits"], gf, reduction="none",
        )
        bce = _masked_mean(bce, valid)
        total = (
            self.grad_weight * gradient_loss
            + self.gf_weight * ft_loss
            + self.bce_weight * bce
        )
        return total, {
            "gradient_loss": gradient_loss.detach(),
            "focal_tversky": ft_loss.detach(),
            "gf_bce": bce.detach(),
        }


class Stage2ProxyLoss(nn.Module):
    """Focal-Tversky + Dice + BCE for Garg-proxy candidate segmentation."""

    def __init__(self, config):
        super().__init__()
        self.ft_weight = float(config.get("stage2_focal_tversky_weight", 1.0))
        self.dice_weight = float(config.get("stage2_dice_weight", 0.5))
        self.bce_weight = float(config.get("stage2_bce_weight", 0.25))
        self.ft = MaskedFocalTverskyLoss(
            alpha=float(config.get("focal_tversky_alpha", 0.3)),
            beta=float(config.get("focal_tversky_beta", 0.7)),
            gamma=float(config.get("focal_tversky_gamma", 0.75)),
        )

    def forward(self, logits, y_true):
        label, valid = y_true[:, 0:1], y_true[:, 1:2]
        ft = self.ft(logits, label, valid)
        prob = torch.sigmoid(logits)
        inter = (prob * label * valid).sum()
        union = ((prob + label) * valid).sum()
        dice = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)
        bce = _masked_mean(
            F.binary_cross_entropy_with_logits(logits, label, reduction="none"),
            valid,
        )
        total = self.ft_weight * ft + self.dice_weight * dice + self.bce_weight * bce
        return total, {
            "focal_tversky": ft.detach(),
            "dice_loss": dice.detach(),
            "bce": bce.detach(),
        }


class BinaryFocalLoss(nn.Module):
    """Focal BCE for sparse buoy-confirmed event/non-event supervision."""

    def __init__(self, alpha=0.95, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = float(alpha), float(gamma)

    def forward(self, logits, target):
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = torch.sigmoid(logits)
        pt = torch.where(target > 0.5, prob, 1.0 - prob)
        alpha_t = torch.where(
            target > 0.5,
            torch.as_tensor(self.alpha, device=logits.device),
            torch.as_tensor(1.0 - self.alpha, device=logits.device),
        )
        return (alpha_t * (1.0 - pt).pow(self.gamma) * bce).mean()


@torch.no_grad()
def stage1_metrics(outputs, y_true, threshold=0.5):
    """Batch sufficient statistics; aggregate counts before deriving scores."""
    gradient, gf, valid = y_true[:, 0:1], y_true[:, 1:2], y_true[:, 2:3]
    pred = (torch.sigmoid(outputs["gf_logits"]) >= threshold).float()
    high = valid * gf
    return {
        "abs_error_sum": float((torch.abs(outputs["gradient"] - gradient) * valid).sum()),
        "high_abs_error_sum": float(
            (torch.abs(outputs["gradient"] - gradient) * high).sum()
        ),
        "high_count": float(high.sum()),
        "gradient_target_sum": float((gradient * valid).sum()),
        "gradient_pred_sum": float((outputs["gradient"] * valid).sum()),
        "valid_count": float(valid.sum()),
        "tp": float((pred * gf * valid).sum()),
        "fp": float((pred * (1.0 - gf) * valid).sum()),
        "fn": float(((1.0 - pred) * gf * valid).sum()),
    }


def scores_from_counts(counts, eps=1.0e-6):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "mae": counts.get("abs_error_sum", 0.0) / max(counts.get("valid_count", 0.0), eps),
        "high_gradient_mae": counts.get("high_abs_error_sum", 0.0) / max(
            counts.get("high_count", 0.0), eps
        ),
        "gradient_bias": (
            counts.get("gradient_pred_sum", 0.0) - counts.get("gradient_target_sum", 0.0)
        ) / max(counts.get("valid_count", 0.0), eps),
        "precision": tp / max(tp + fp, eps),
        "recall": tp / max(tp + fn, eps),
        "iou": tp / max(tp + fp + fn, eps),
        "far": fp / max(tp + fp, eps),
        "dice": 2.0 * tp / max(2.0 * tp + fp + fn, eps),
    }


class DeterministicCRPSLoss(nn.Module):
    """M=1 rung: weighted L1 on normalized Sobel-WGM."""

    def __init__(self, config):
        super().__init__()
        self.peak = float(config.get("gradient_peak_weight", 4.0))

    def forward(self, pred, y_true):
        y, valid = y_true[:, 0:1], y_true[:, 1:2]
        loss = deterministic_l1_crps(pred, y, valid, peak=self.peak)
        return loss, {"crps": loss.detach()}


class GaussianCRPSLoss(nn.Module):
    """(mu, sigma) rung: closed-form Gaussian CRPS."""

    def __init__(self, config):
        super().__init__()
        self.peak = float(config.get("gradient_peak_weight", 4.0))

    def forward(self, outputs, y_true):
        y, valid = y_true[:, 0:1], y_true[:, 1:2]
        loss = gaussian_crps(
            outputs["mu"], outputs["log_sigma"], y, valid, peak=self.peak,
        )
        return loss, {"crps": loss.detach()}


class EnsembleCRPSLoss(nn.Module):
    """M-member afCRPS with latent-noise ensemble."""

    def __init__(self, config):
        super().__init__()
        ens = config.get("ensemble", {})
        self.members = int(ens.get("members", config.get("ensemble_members", 8)))
        self.alpha = float(ens.get("alpha", 0.95))
        self.peak = float(config.get("gradient_peak_weight", 4.0))

    def forward(self, members, y_true):
        y, valid = y_true[:, 0:1], y_true[:, 1:2]
        loss = almost_fair_crps(
            members, y, valid, alpha=self.alpha, peak=self.peak,
        )
        return loss, {"crps": loss.detach()}
