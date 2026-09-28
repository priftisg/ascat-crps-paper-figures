"""Full-swath U-Net with anisotropic pooling and probabilistic CRPS heads."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = min(8, out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _center_crop(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, _, h, w = x.shape
    if h == target_h and w == target_w:
        return x
    dh = max((h - target_h) // 2, 0)
    dw = max((w - target_w) // 2, 0)
    return x[:, :, dh : dh + target_h, dw : dw + target_w]


class FullSwathUNet(nn.Module):
    """
    U-Net for tall, narrow ASCAT swath halves.

    Input / output: (N, C, H, W) with H = along-track, W = cross-track (41).
    """

    def __init__(
        self,
        in_channels: int,
        filters: int = 16,
        num_layers: int = 4,
        pool_along: int = 2,
        pool_cross: int = 1,
        out_channels: int = 1,
    ):
        super().__init__()
        self.pool_size = (pool_along, pool_cross)
        self.out_channels = out_channels

        self.encoder = nn.ModuleList()
        ch_in = in_channels
        for i in range(num_layers):
            ch_out = filters * (2 ** min(i, 2))
            self.encoder.append(
                nn.ModuleDict({
                    "conv": ConvBlock(ch_in, ch_out),
                    "pool": nn.MaxPool2d(
                        kernel_size=self.pool_size,
                        stride=self.pool_size,
                    ),
                })
            )
            ch_in = ch_out

        self.bottleneck = ConvBlock(ch_in, filters * 4)
        ch_in = filters * 4

        self.decoder = nn.ModuleList()
        for i in range(num_layers):
            up_ch = filters * (2 ** min(num_layers - 1 - i, 2))
            skip_ch = filters * (2 ** min(num_layers - 1 - i, 2))
            self.decoder.append(
                nn.ModuleDict({
                    "up": nn.ConvTranspose2d(
                        ch_in,
                        up_ch,
                        kernel_size=self.pool_size,
                        stride=self.pool_size,
                    ),
                    "conv": ConvBlock(up_ch + skip_ch, up_ch),
                })
            )
            ch_in = up_ch

        self.head = nn.Conv2d(ch_in, out_channels, kernel_size=1)
        self._decoder_out_channels = ch_in

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.encoder:
            x = block["conv"](x)
            skips.append(x)
            x = block["pool"](x)
        x = self.bottleneck(x)
        for block, skip in zip(self.decoder, reversed(skips)):
            x = block["up"](x)
            x = _center_crop(x, skip.shape[2], skip.shape[3])
            x = torch.cat([x, skip], dim=1)
            x = block["conv"](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_decode(x))


class DeterministicSobelUNet(FullSwathUNet):
    """M=1 baseline: non-negative Sobel-WGM regression (softplus)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(super().forward(x))


class GaussianCRPSUNet(FullSwathUNet):
    """Parametric (mu, log_sigma) head for closed-form Gaussian CRPS."""

    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["out_channels"] = 2
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = super().forward(x)
        mu = F.softplus(out[:, 0:1])
        log_sigma = out[:, 1:2]
        return {"mu": mu, "log_sigma": log_sigma}


class NoisyUNet(nn.Module):
    """
    Latent-noise wrapper for spatially coherent ensemble members.

    Low-resolution Gaussian noise is bilinearly upsampled and concatenated
    to the input; one independent draw per member / forward pass.
    """

    def __init__(self, base_unet: nn.Module, noise_ch: int = 4, noise_stride: int = 8):
        super().__init__()
        self.net = base_unet
        self.k = int(noise_ch)
        self.s = int(noise_stride)

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        n, _, h, w = x.shape
        if z is None:
            z = torch.randn(
                n, self.k, max(h // self.s, 1), max(w // self.s, 1),
                device=x.device, dtype=x.dtype,
            )
        z = F.interpolate(z, size=(h, w), mode="bilinear", align_corners=False)
        return self.net(torch.cat([x, z], dim=1))


def ensemble_forward(model: nn.Module, x: torch.Tensor, m: int = 8) -> torch.Tensor:
    """Stack M stochastic forward passes → (N, M, H, W)."""
    members = []
    for _ in range(m):
        out = model(x)
        if isinstance(out, dict):
            out = out["mu"]
        members.append(out.squeeze(1))
    return torch.stack(members, dim=1)


def build_full_swath_unet(
    num_channels: int,
    filters: int = 16,
    num_layers: int = 4,
    pool_along: int = 2,
    pool_cross: int = 1,
    out_channels: int = 1,
) -> FullSwathUNet:
    return FullSwathUNet(
        in_channels=num_channels,
        filters=filters,
        num_layers=num_layers,
        pool_along=pool_along,
        pool_cross=pool_cross,
        out_channels=out_channels,
    )


def build_deterministic_unet(
    num_channels: int,
    filters: int = 16,
    num_layers: int = 4,
    pool_along: int = 2,
    pool_cross: int = 1,
) -> DeterministicSobelUNet:
    return DeterministicSobelUNet(
        in_channels=num_channels,
        filters=filters,
        num_layers=num_layers,
        pool_along=pool_along,
        pool_cross=pool_cross,
        out_channels=1,
    )


def build_gaussian_unet(
    num_channels: int,
    filters: int = 16,
    num_layers: int = 4,
    pool_along: int = 2,
    pool_cross: int = 1,
) -> GaussianCRPSUNet:
    return GaussianCRPSUNet(
        in_channels=num_channels,
        filters=filters,
        num_layers=num_layers,
        pool_along=pool_along,
        pool_cross=pool_cross,
    )


def build_ensemble_unet(
    num_channels: int,
    filters: int = 16,
    num_layers: int = 4,
    pool_along: int = 2,
    pool_cross: int = 1,
    noise_ch: int = 4,
    noise_stride: int = 8,
) -> NoisyUNet:
    base = DeterministicSobelUNet(
        in_channels=num_channels + noise_ch,
        filters=filters,
        num_layers=num_layers,
        pool_along=pool_along,
        pool_cross=pool_cross,
        out_channels=1,
    )
    return NoisyUNet(base, noise_ch=noise_ch, noise_stride=noise_stride)


# --- legacy multi-task builders kept for older scripts / tests ---------------

class MultiTaskStage1UNet(FullSwathUNet):
    """Stage-1 U-Net with continuous-gradient and GF-segmentation heads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        final_channels = self._decoder_out_channels
        del self.head
        self.gradient_head = nn.Conv2d(final_channels, 1, kernel_size=1)
        self.gf_head = nn.Conv2d(final_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.encode_decode(x)
        return {
            "gradient": F.softplus(self.gradient_head(feats)),
            "gf_logits": self.gf_head(feats),
        }


def build_stage1_unet(
    num_channels: int,
    filters: int = 16,
    num_layers: int = 4,
    pool_along: int = 2,
    pool_cross: int = 1,
) -> MultiTaskStage1UNet:
    return MultiTaskStage1UNet(
        in_channels=num_channels,
        filters=filters,
        num_layers=num_layers,
        pool_along=pool_along,
        pool_cross=pool_cross,
    )
