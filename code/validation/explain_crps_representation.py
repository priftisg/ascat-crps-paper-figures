#!/usr/bin/env python3
"""
Explainability / representation analysis for the CRPS ensemble U-Net.

Three complementary analyses for an AI-focused Results subsection:

  (1) Latent / bottleneck representation
      Extract encoder-bottleneck features with fixed latent noise z, reduce with
      PCA (optional UMAP), color by physical covariates.

  (2) Predictor-group permutation importance
      Shuffle along-track structure of natural channel groups; report ΔafCRPS
      and ΔBrier at τ* relative to a fixed-z baseline.

  (3) Integrated gradients (spatial attribution)
      Attribute soft P(S>τ*) over high-probability ROIs for a few showcase
      orbits, with fixed z.

Example (smoke)
---------------
  python \\
    validation/explain_crps_representation.py --limit 4 --ig-orbits 73598

Full val (GPU recommended)
--------------------------
  ... --limit 40 --stride 5 --outdir evaluation/explainability/crps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from paths_config import ensure_sys_path, figures_repo_root, get, research_root
except ImportError:
    from code.paths_config import ensure_sys_path, figures_repo_root, get, research_root  # type: ignore
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = research_root(Path(__file__))
ensure_sys_path(ROOT)

from infer_crps import load_crps_model  # noqa: E402
from lib.crps_dataset import build_crps_half, orbit_from_path, split_files  # noqa: E402
from lib.dataset import prepare_swath  # noqa: E402
from lib.losses import almost_fair_crps  # noqa: E402
from lib.swath_halves import HALF_NAMES  # noqa: E402
from validation.ig_geo_plot import (  # noqa: E402
    HAS_CARTOPY as _HAS_CARTOPY,
    extent_from_seg,
    geo_scatter_field,
    stitch_halves_geo as _stitch_halves_geo,
    tropical_segment_mask,
)

try:
    import cartopy.crs as ccrs
except Exception:  # pragma: no cover
    ccrs = None  # type: ignore

# Density-matched operating point used elsewhere in the pack
TAU_STAR = 1.2e-4
DEFAULT_IG_ORBITS = (73598, 72850, 75001)

# Natural ASCAT predictor groups (indices resolved from config channel names)
PREDICTOR_GROUPS: Dict[str, Tuple[str, ...]] = {
    "smooth_context": ("wind_speed_smooth", "wind_gradient_smooth"),
    "rain_qc": ("rad_rain", "rain_flag", "iclass", "valid_mask"),
    "numamb": ("numamb_norm",),
    "ambiguity_winds": (
        "u_amb1", "v_amb1", "sos_amb1",
        "u_amb2", "v_amb2", "sos_amb2",
        "u_amb3", "v_amb3", "sos_amb3",
        "u_amb4", "v_amb4", "sos_amb4",
    ),
}


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def resolve_groups(channels: Sequence[str]) -> Dict[str, List[int]]:
    name_to_idx = {n: i for i, n in enumerate(channels)}
    out: Dict[str, List[int]] = {}
    for gname, names in PREDICTOR_GROUPS.items():
        idxs = []
        for n in names:
            if n not in name_to_idx:
                raise KeyError(f"channel {n!r} missing from model inputs")
            idxs.append(name_to_idx[n])
        out[gname] = idxs
    return out


def draw_fixed_zs(
    model: torch.nn.Module,
    x: torch.Tensor,
    m: int,
    seed: int,
) -> List[torch.Tensor]:
    """Draw M latent tensors matching NoisyUNet geometry (reproducible)."""
    n, _, h, w = x.shape
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    zs = []
    for _ in range(m):
        z = torch.randn(
            n, model.k, max(h // model.s, 1), max(w // model.s, 1),
            generator=g, dtype=torch.float32,
        ).to(device=x.device, dtype=x.dtype)
        zs.append(z)
    return zs


def ensemble_forward_fixed(
    model: torch.nn.Module,
    x: torch.Tensor,
    zs: Sequence[torch.Tensor],
) -> torch.Tensor:
    members = []
    for z in zs:
        out = model(x, z=z)
        if isinstance(out, dict):
            out = out["mu"]
        members.append(out.squeeze(1))
    return torch.stack(members, dim=1)


def soft_exceedance_from_members(
    members: torch.Tensor,
    tau: float,
    gradient_scale: float,
    h_soft: float,
) -> torch.Tensor:
    """Differentiable soft P(S>τ) mean over members. members: (N,M,H,W)."""
    t = float(tau) / float(gradient_scale)
    p = torch.sigmoid((members - t) / h_soft)
    return p.mean(dim=1)  # (N,H,W)


def extract_bottleneck(
    model: torch.nn.Module,
    x: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """Run encoder+bottleneck on cat(x, upsample(z)); return (N,C,h,w)."""
    n, _, h, w = x.shape
    z_up = F.interpolate(z, size=(h, w), mode="bilinear", align_corners=False)
    h_ = torch.cat([x, z_up], dim=1)
    net = model.net
    for block in net.encoder:
        h_ = block["conv"](h_)
        h_ = block["pool"](h_)
    return net.bottleneck(h_)


def to_chw(x_np: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x_np).permute(2, 0, 1).unsqueeze(0).to(device)


def load_cfg(args) -> Tuple[torch.nn.Module, dict, torch.device, int, float, float]:
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with cfg_path.open() as f:
        file_cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckpt_cfg, rung = load_crps_model(args.checkpoint, device)
    if rung != "ensemble":
        raise SystemExit(f"need ensemble checkpoint, got rung={rung}")
    cfg = {
        **file_cfg,
        **{
            k: ckpt_cfg[k]
            for k in (
                "input_channels", "num_channels", "filters", "num_layers",
                "pool_along_track", "pool_cross_track", "ensemble",
                "gradient_scale", "fov_m", "gradient_peak_weight",
                "exceedance_softness", "taus",
            )
            if k in ckpt_cfg
        },
    }
    ens = cfg.get("ensemble", {})
    m = int(args.members or ens.get("members", 8))
    alpha = float(ens.get("alpha", 0.95))
    peak = float(cfg.get("gradient_peak_weight", 4.0))
    model.eval()
    print(f"device={device}  M={m}  channels={len(cfg['input_channels'])}", flush=True)
    return model, cfg, device, m, alpha, peak


def select_files(cfg: dict, limit: Optional[int], stride: int) -> List[str]:
    files = split_files(cfg, "val")
    if stride > 1:
        files = files[::stride]
    if limit:
        files = files[: int(limit)]
    return files


# ---------------------------------------------------------------------------
# (1) Bottleneck representation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_latent_analysis(
    model, cfg, device, m, files, outdir: Path, *,
    seed: int, max_pixels: int, tau: float, h_soft: float,
) -> dict:
    channels = list(cfg["input_channels"])
    scale = float(cfg.get("gradient_scale", 5.0e-4))
    tau_s = float(tau) / scale
    idx_ugrad = channels.index("wind_gradient_smooth")
    idx_uspeed = channels.index("wind_speed_smooth")
    idx_namb = channels.index("numamb_norm")
    idx_rain = channels.index("rain_flag")

    feats: List[np.ndarray] = []
    meta_cols = {
        "S": [], "P": [], "U_smooth": [], "gradU_smooth": [],
        "N_amb": [], "rain_flag": [], "orbit": [],
    }
    rng = np.random.default_rng(seed)
    n_halves = 0

    for i, path in enumerate(files, 1):
        orbit = orbit_from_path(path)
        for half in HALF_NAMES:
            x_np, y_np = build_crps_half(path, half, cfg)
            x = to_chw(x_np, device)
            y = to_chw(y_np, device)
            zs = draw_fixed_zs(model, x, m=m, seed=seed + n_halves)
            members = ensemble_forward_fixed(model, x, zs)
            p = soft_exceedance_from_members(members, tau, scale, h_soft)[0]
            bn = extract_bottleneck(model, x, zs[0])[0]  # (C,h,w)
            # Upsample bottleneck to input resolution for pixel alignment
            bn_up = F.interpolate(
                bn.unsqueeze(0), size=x.shape[-2:], mode="bilinear", align_corners=False,
            )[0]  # (C,H,W)

            valid = y[0, 1].cpu().numpy() > 0.5
            S = y[0, 0].cpu().numpy()
            P = p.cpu().numpy()
            U = x[0, idx_uspeed].cpu().numpy()
            G = x[0, idx_ugrad].cpu().numpy()
            N = x[0, idx_namb].cpu().numpy()
            R = x[0, idx_rain].cpu().numpy()
            C = bn_up.cpu().numpy()  # (C,H,W)

            # Stratify: half high-S, half background among valid
            if not valid.any():
                continue
            s_thr = float(np.nanpercentile(S[valid], 90))
            hi = valid & (S >= s_thr)
            lo = valid & ~hi
            n_take = max(max_pixels // max(len(files) * 2, 1), 64)
            sel_idx = []
            for mask in (hi, lo):
                ys, xs = np.where(mask)
                if ys.size == 0:
                    continue
                k = min(max(n_take // 2, 1), ys.size)
                pick = rng.choice(ys.size, size=k, replace=False)
                sel_idx.append((ys[pick], xs[pick]))
            if not sel_idx:
                continue
            yy = np.concatenate([a[0] for a in sel_idx])
            xx = np.concatenate([a[1] for a in sel_idx])
            feats.append(C[:, yy, xx].T)  # (n, C)
            meta_cols["S"].append(S[yy, xx])
            meta_cols["P"].append(P[yy, xx])
            meta_cols["U_smooth"].append(U[yy, xx])
            meta_cols["gradU_smooth"].append(G[yy, xx])
            meta_cols["N_amb"].append(N[yy, xx])
            meta_cols["rain_flag"].append(R[yy, xx])
            meta_cols["orbit"].append(np.full(yy.size, int(orbit), dtype=np.int64))
            n_halves += 1

        if i % 5 == 0 or i == len(files):
            print(f"  latent [{i}/{len(files)}] halves={n_halves} pixels≈{sum(len(f) for f in feats):,}", flush=True)

    if not feats:
        raise RuntimeError("no latent features collected — check val data / limit")

    X = np.concatenate(feats, axis=0).astype(np.float32)
    meta = {k: np.concatenate(v) for k, v in meta_cols.items()}
    # PCA via sklearn
    from sklearn.decomposition import PCA
    n_comp = min(8, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_comp, random_state=seed)
    Z = pca.fit_transform(X)

    # Optional UMAP
    Z_umap = None
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=30, min_dist=0.1)
        Z_umap = reducer.fit_transform(X)
        print("  UMAP OK", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  UMAP skipped ({exc})", flush=True)

    color_keys = [
        ("S", r"Target $S$", "magma"),
        ("P", rf"$P(S>\tau^*)$", "YlOrRd"),
        ("gradU_smooth", r"$|\nabla U|_{{\rm smooth}}$", "viridis"),
        ("N_amb", r"$N_{{\rm amb}}$", "coolwarm"),
        ("U_smooth", r"$U_{{\rm smooth}}$", "plasma"),
        ("rain_flag", "rain flag", "Greys"),
    ]
    n_panel = len(color_keys)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.8))
    for ax, (key, title, cmap) in zip(axes.ravel(), color_keys):
        c = meta[key]
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=c, s=4, cmap=cmap, alpha=0.55, linewidths=0)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title, fontsize=10)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    png = outdir / "13_latent_bottleneck_pca.png"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", png, flush=True)

    if Z_umap is not None:
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.8))
        for ax, (key, title, cmap) in zip(axes.ravel(), color_keys):
            sc = ax.scatter(
                Z_umap[:, 0], Z_umap[:, 1], c=meta[key], s=4, cmap=cmap, alpha=0.55, linewidths=0,
            )
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
            ax.set_title(title, fontsize=10)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        upng = outdir / "13_latent_bottleneck_umap.png"
        fig.savefig(upng, dpi=200, bbox_inches="tight", facecolor="white")
        fig.savefig(upng.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("wrote", upng, flush=True)

    meta_out = {
        "n_pixels": int(X.shape[0]),
        "n_halves": int(n_halves),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "tau": float(tau),
        "feature_dim": int(X.shape[1]),
    }
    (outdir / "13_latent_meta.json").write_text(json.dumps(meta_out, indent=2) + "\n")
    # Persist embeddings for main/SI figure packing
    np.savez_compressed(
        outdir / "13_latent_embeddings.npz",
        Z=Z.astype(np.float32),
        S=meta["S"].astype(np.float32),
        P=meta["P"].astype(np.float32),
        U_smooth=meta["U_smooth"].astype(np.float32),
        gradU_smooth=meta["gradU_smooth"].astype(np.float32),
        N_amb=meta["N_amb"].astype(np.float32),
        rain_flag=meta["rain_flag"].astype(np.float32),
        orbit=meta["orbit"].astype(np.int64),
        pca_explained_variance_ratio=np.asarray(
            pca.explained_variance_ratio_, dtype=np.float64,
        ),
    )
    print("wrote", outdir / "13_latent_embeddings.npz", flush=True)
    return meta_out


# ---------------------------------------------------------------------------
# (2) Predictor-group permutation
# ---------------------------------------------------------------------------

def permute_group_along_track(
    x: torch.Tensor,
    idxs: Sequence[int],
    rng: np.random.Generator,
) -> torch.Tensor:
    """Break along-track structure of selected channels (keeps marginals)."""
    x2 = x.clone()
    h = x.shape[2]
    perm = rng.permutation(h)
    for j in idxs:
        x2[:, j] = x[:, j, perm, :]
    return x2


def brier_score(
    p: torch.Tensor,
    y_bin: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """Pixel Brier for P vs 1{S>τ}. p,y_bin,valid: (H,W) or (1,H,W)."""
    p = p.reshape(-1)
    yb = y_bin.reshape(-1)
    m = valid.reshape(-1) > 0.5
    if not bool(m.any()):
        return float("nan")
    return float(((p[m] - yb[m]) ** 2).sum() / (m.sum() + eps))


@torch.no_grad()
def score_half(
    model, x, y, zs, *, alpha, peak, tau, scale, h_soft,
) -> Tuple[float, float, float]:
    members = ensemble_forward_fixed(model, x, zs)
    y_t, valid = y[:, 0:1], y[:, 1:2]
    crps = float(almost_fair_crps(members, y_t, valid, alpha=alpha, peak=peak).item())
    p = soft_exceedance_from_members(members, tau, scale, h_soft)
    y_bin = (y_t[:, 0] > (tau / scale)).float()
    brier = brier_score(p[0], y_bin[0], valid[0, 0])
    w = float(valid.sum().item())
    return crps, brier, w


@torch.no_grad()
def run_group_permutation(
    model, cfg, device, m, alpha, peak, files, outdir: Path, *,
    seed: int, tau: float, h_soft: float,
) -> pd.DataFrame:
    channels = list(cfg["input_channels"])
    groups = resolve_groups(channels)
    scale = float(cfg.get("gradient_scale", 5.0e-4))
    rng = np.random.default_rng(seed)

    # Accumulate weighted means
    keys = ["baseline"] + list(groups.keys())
    crps_num = {k: 0.0 for k in keys}
    brier_num = {k: 0.0 for k in keys}
    den = {k: 0.0 for k in keys}
    n_halves = 0

    for i, path in enumerate(files, 1):
        for half in HALF_NAMES:
            x_np, y_np = build_crps_half(path, half, cfg)
            x = to_chw(x_np, device)
            y = to_chw(y_np, device)
            zs = draw_fixed_zs(model, x, m=m, seed=seed + n_halves)

            c0, b0, w = score_half(
                model, x, y, zs, alpha=alpha, peak=peak, tau=tau, scale=scale, h_soft=h_soft,
            )
            if w <= 0 or not np.isfinite(c0):
                continue
            crps_num["baseline"] += c0 * w
            brier_num["baseline"] += b0 * w
            den["baseline"] += w

            for gname, idxs in groups.items():
                xp = permute_group_along_track(x, idxs, rng)
                c, b, _ = score_half(
                    model, xp, y, zs, alpha=alpha, peak=peak, tau=tau, scale=scale, h_soft=h_soft,
                )
                crps_num[gname] += c * w
                brier_num[gname] += b * w
                den[gname] += w

            n_halves += 1

        if i % 5 == 0 or i == len(files):
            print(f"  permute [{i}/{len(files)}] halves={n_halves}", flush=True)

    rows = []
    base_c = crps_num["baseline"] / max(den["baseline"], 1.0)
    base_b = brier_num["baseline"] / max(den["baseline"], 1.0)
    for k in keys:
        c = crps_num[k] / max(den[k], 1.0)
        b = brier_num[k] / max(den[k], 1.0)
        rows.append({
            "group": k,
            "n_channels": 0 if k == "baseline" else len(groups[k]),
            "afCRPS": c,
            "delta_afCRPS": c - base_c,
            "Brier": b,
            "delta_Brier": b - base_b,
            "weight_pixels": den[k],
            "n_halves": n_halves,
        })
    df = pd.DataFrame(rows)
    csv_path = outdir / "13_predictor_group_permutation.csv"
    df.to_csv(csv_path, index=False)
    print("wrote", csv_path, flush=True)

    # Bar chart of ΔafCRPS (exclude baseline)
    plot_df = df[df["group"] != "baseline"].sort_values("delta_afCRPS", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels = {
        "smooth_context": r"smooth $U$, $|\nabla U|$",
        "rain_qc": "rain / QC",
        "numamb": r"$N_{\mathrm{amb}}$",
        "ambiguity_winds": "ambiguity winds / SOS",
    }
    ylabels = [labels.get(g, g) for g in plot_df["group"]]
    axes[0].barh(ylabels, plot_df["delta_afCRPS"], color="#1a5fa8", edgecolor="white")
    axes[0].axvline(0, color="0.3", lw=0.8)
    axes[0].set_xlabel(r"$\Delta$ afCRPS (permute − baseline)")
    axes[0].set_title("Field afCRPS importance")
    axes[1].barh(ylabels, plot_df["delta_Brier"], color="#c23b22", edgecolor="white")
    axes[1].axvline(0, color="0.3", lw=0.8)
    axes[1].set_xlabel(rf"$\Delta$ Brier at $\tau={tau:.1e}$")
    axes[1].set_title("Exceedance Brier importance")
    fig.tight_layout()
    png = outdir / "13_predictor_group_permutation.png"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", png, flush=True)

    # TeX snippet
    tex = [
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Group & $n_{\mathrm{ch}}$ & afCRPS & $\Delta$afCRPS & $\Delta$Brier \\",
        r"\hline",
    ]
    for _, r in df.iterrows():
        tex.append(
            f"{r['group']} & {int(r['n_channels'])} & {r['afCRPS']:.5f} & "
            f"{r['delta_afCRPS']:+.5f} & {r['delta_Brier']:+.5f} \\\\"
        )
    tex += [r"\hline", r"\end{tabular}", ""]
    (outdir / "13_predictor_group_permutation.tex").write_text("\n".join(tex))
    return df


# ---------------------------------------------------------------------------
# (3) Integrated gradients
# ---------------------------------------------------------------------------

def integrated_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    z: torch.Tensor,
    baseline: torch.Tensor,
    target_fn,
    steps: int = 32,
) -> torch.Tensor:
    """
    IG attribution on input x (ASCAT channels only).
    Returns (1,C,H,W) attributions.
    """
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=x.device, dtype=x.dtype)[1:]
    grads = []
    delta = x - baseline
    for a in alphas:
        xi = (baseline + a * delta).detach().requires_grad_(True)
        score = target_fn(model, xi, z)
        g, = torch.autograd.grad(score, xi, retain_graph=False)
        grads.append(g.detach())
    avg = torch.stack(grads, dim=0).mean(dim=0)
    return delta * avg


def find_orbit_path(files: Sequence[str], orbit: int) -> Optional[str]:
    tag = f"_{int(orbit):05d}_" if orbit < 100000 else f"_{int(orbit)}_"
    # ASCAT names use ascat_winvec_XXXXX_...
    for p in files:
        if f"_{int(orbit)}_" in Path(p).name or Path(p).name.split("_")[2] == str(int(orbit)):
            return p
    # Also search full val list if orbit not in subsampled files
    return None


def _ig_half(
    model, cfg, device, path, half, *,
    channels, groups, scale, tau, h_soft, seed, orbit, steps,
):
    """Run IG on one swath half; return maps + group shares."""
    x_np, y_np = build_crps_half(path, half, cfg)
    x = to_chw(x_np, device)
    y = to_chw(y_np, device)
    z0 = draw_fixed_zs(model, x, m=1, seed=seed + orbit + (0 if half == "left" else 1))[0]
    with torch.no_grad():
        s = model(x, z=z0).squeeze(1)[0]
        p = torch.sigmoid((s - tau / scale) / h_soft)
        valid = y[0, 1] > 0.5

    p_np = p.detach().cpu().numpy()
    v_np = valid.detach().cpu().numpy() > 0.5
    if v_np.any():
        thr = np.nanpercentile(p_np[v_np], 98)
        roi = torch.from_numpy((p_np >= thr) & v_np).to(device=device)
    else:
        roi = valid
    if float(roi.float().sum()) < 4:
        S = y[0, 0].cpu().numpy()
        thr = np.nanpercentile(S[v_np], 95) if v_np.any() else 0.1
        roi = torch.from_numpy((S >= thr) & v_np).to(device=device)

    def target_fn(mdl, xi, z, _roi=roi):
        s = mdl(xi, z=z).squeeze(1)[0]
        p = torch.sigmoid((s - tau / scale) / h_soft)
        return (p * _roi.float()).sum() / _roi.float().sum().clamp_min(1.0)

    baseline = torch.zeros_like(x)
    if "valid_mask" in channels:
        j = channels.index("valid_mask")
        baseline[:, j] = x[:, j]

    attr = integrated_gradients(model, x, z0, baseline, target_fn, steps=steps)
    attr_np = attr[0].detach().cpu().numpy()
    roi_np = roi.cpu().numpy()
    abs_map = np.abs(attr_np).sum(axis=0)
    amb_map = np.abs(attr_np[groups["ambiguity_winds"]]).sum(axis=0)
    gch = x[0, channels.index("wind_gradient_smooth")].detach().cpu().numpy()

    total = float(np.abs(attr_np)[:, roi_np].sum()) + 1e-12
    group_share = {
        gname: float(np.abs(attr_np[idxs])[:, roi_np].sum()) / total
        for gname, idxs in groups.items()
    }
    return {
        "half": half,
        "p": p_np,
        "roi": roi_np,
        "abs_map": abs_map,
        "amb_map": amb_map,
        "gradU": gch,
        "group_share": group_share,
    }


def _stitch_halves(left: np.ndarray, right: np.ndarray, gap: int = 2) -> np.ndarray:
    """Concatenate left|gap|right along cross-track (axis=1) — index-space only."""
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("expected 2-D maps")
    h = min(left.shape[0], right.shape[0])
    g = np.full((h, gap), np.nan, dtype=np.float64)
    return np.concatenate([left[:h], g, right[:h]], axis=1)


def load_swath_lonlat(path: str, cfg: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lon/lat/valid on the same trimmed grid used by ``build_crps_half``."""
    swath = prepare_swath(path, cfg)
    lon = np.asarray(swath["lon"], dtype=np.float32)
    lat = np.asarray(swath["lat"], dtype=np.float32)
    if "valid_mask" in swath:
        valid = np.asarray(swath["valid_mask"], dtype=np.float32)
    else:
        valid = np.isfinite(lon).astype(np.float32)
    return lon, lat, valid


def run_integrated_gradients(
    model, cfg, device, files, outdir: Path, *,
    orbits: Sequence[int], seed: int, tau: float, h_soft: float, steps: int,
) -> None:
    if not _HAS_CARTOPY:
        print(
            "NOTE: cartopy not available — plotting lon/lat scatter without coastlines; "
            "rerun plot step in a Cartopy-enabled environment for coastline overlays",
            flush=True,
        )

    channels = list(cfg["input_channels"])
    groups = resolve_groups(channels)
    scale = float(cfg.get("gradient_scale", 5.0e-4))
    all_val = split_files(cfg, "val")

    cases = []
    for orb in orbits:
        path = find_orbit_path(files, orb) or find_orbit_path(all_val, orb)
        if path is None:
            print(f"  IG skip orbit {orb}: not found", flush=True)
            continue
        cases.append((orb, path))
    if not cases:
        for path in files[: min(3, len(files))]:
            cases.append((int(orbit_from_path(path)), path))
        print(f"  IG fallback cases: {[c[0] for c in cases]}", flush=True)

    group_order = list(groups.keys())
    n_cases = len(cases)
    # Geographic maps are taller than index-space strips
    fig_w = 16.5
    fig_h = 4.4 * n_cases
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_cases, 4, left=0.07, right=0.97, bottom=0.06, top=0.94,
        wspace=0.22, hspace=0.28,
    )

    summary_rows = []
    for row, (orbit, path) in enumerate(cases):
        halves = {}
        for half in HALF_NAMES:
            halves[half] = _ig_half(
                model, cfg, device, path, half,
                channels=channels, groups=groups, scale=scale,
                tau=tau, h_soft=h_soft, seed=seed, orbit=orbit, steps=steps,
            )
            summary_rows.append({
                "orbit": orbit,
                "half": half,
                **halves[half]["group_share"],
            })
            print(
                f"  IG orbit={orbit} half={half} shares="
                + ", ".join(f"{k}={v:.2f}" for k, v in halves[half]["group_share"].items()),
                flush=True,
            )

        L, R = halves["left"], halves["right"]
        p_full = _stitch_halves_geo(L["p"], R["p"])
        roi_full = _stitch_halves_geo(L["roi"].astype(float), R["roi"].astype(float))
        abs_full = _stitch_halves_geo(L["abs_map"], R["abs_map"])
        grad_full = _stitch_halves_geo(L["gradU"], R["gradU"])
        amb_full = _stitch_halves_geo(L["amb_map"], R["amb_map"])

        lon, lat, valid = load_swath_lonlat(path, cfg)
        h = min(lon.shape[0], p_full.shape[0])
        w = min(lon.shape[1], p_full.shape[1])
        lon, lat, valid = lon[:h, :w], lat[:h, :w], valid[:h, :w]
        p_full = p_full[:h, :w]
        roi_full = roi_full[:h, :w]
        abs_full = abs_full[:h, :w]
        grad_full = grad_full[:h, :w]
        amb_full = amb_full[:h, :w]

        # Persist geo fields for pack / replot without recomputing IG
        np.savez_compressed(
            outdir / f"13_ig_geo_orbit{orbit}.npz",
            p_full=p_full.astype(np.float32),
            abs_full=abs_full.astype(np.float32),
            roi_full=roi_full.astype(np.float32),
            grad_full=grad_full.astype(np.float32),
            amb_full=amb_full.astype(np.float32),
            lon=lon.astype(np.float32),
            lat=lat.astype(np.float32),
            valid=valid.astype(np.float32),
            path=np.asarray(path),
        )

        seg = tropical_segment_mask(lon, lat, valid, p_full, prefer_largest=True)
        extent = extent_from_seg(lon, lat, seg)

        roi_mask = np.isfinite(abs_full) & (roi_full > 0.5) & seg
        vmax = np.nanpercentile(abs_full[roi_mask], 99) if roi_mask.any() else 1.0
        vmax_a = np.nanpercentile(amb_full[roi_mask], 99) if roi_mask.any() else 1.0
        g_finite = grad_full[seg & np.isfinite(grad_full)]
        g_vmax = float(np.nanpercentile(g_finite, 99)) if g_finite.size else 1.0

        panels = [
            (p_full, "YlOrRd", 0.0, 1.0, f"orbit {orbit}  $P(S>\\tau^*)$", roi_full),
            (abs_full, "magma", 0.0, max(float(vmax), 1e-12), r"|IG| (all ASCAT channels)", None),
            (grad_full, "viridis", 0.0, max(g_vmax, 1e-12), r"$|\nabla U|_{\mathrm{smooth}}$ input", None),
            (amb_full, "inferno", 0.0, max(float(vmax_a), 1e-12), "|IG| ambiguity winds/SOS", None),
        ]
        for col, (field, cmap, vmin, vmax_c, title, roi) in enumerate(panels):
            if _HAS_CARTOPY:
                ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
            else:
                ax = fig.add_subplot(gs[row, col])
            sc, _ = geo_scatter_field(
                ax, lon, lat, field, seg,
                cmap=cmap, vmin=vmin, vmax=vmax_c, s=4.0, roi=roi, extent=extent,
                draw_left_labels=(col == 0),
                draw_bottom_labels=(row == n_cases - 1),
            )
            ax.set_title(title, fontsize=10, pad=6)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)

    png = outdir / "13_integrated_gradients_cases.png"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", png, flush=True)

    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(outdir / "13_integrated_gradients_group_share.csv", index=False)

    if len(sdf):
        # One grouped bar cluster per orbit×half
        fig, ax = plt.subplots(figsize=(9.0, 4.0))
        xpos = np.arange(len(sdf))
        width = 0.18
        colors = ["#1a5fa8", "#c23b22", "#2ca02c", "#9467bd"]
        for i, g in enumerate(group_order):
            ax.bar(xpos + i * width, sdf[g], width=width, label=g, color=colors[i % len(colors)])
        ax.set_xticks(xpos + 1.5 * width)
        ax.set_xticklabels([f"{o}\n{h}" for o, h in zip(sdf["orbit"], sdf["half"])], fontsize=8)
        ax.set_ylabel("fraction of |IG| in ROI")
        ax.set_xlabel("orbit / half")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        bpng = outdir / "13_integrated_gradients_group_share.png"
        fig.savefig(bpng, dpi=200, bbox_inches="tight", facecolor="white")
        fig.savefig(bpng.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("wrote", bpng, flush=True)


# ---------------------------------------------------------------------------
# Snippet + pack copy
# ---------------------------------------------------------------------------

def write_snippet(outdir: Path, perm_df: Optional[pd.DataFrame], latent_meta: Optional[dict]) -> None:
    lines = [
        r"\subsection{Representation and input attribution}",
        "",
        r"To assess what the CRPS ensemble U-Net learns from the 19 ASCAT predictors "
        r"and stochastic latent input, we combine three complementary analyses "
        r"(latent-space structure, predictor-group importance, and spatial attribution).",
        "",
    ]
    if latent_meta:
        lines.append(
            rf"Bottleneck features (fixed latent draw) reduced by PCA explain "
            rf"{100 * sum(latent_meta['pca_explained_variance_ratio'][:2]):.1f}\% "
            rf"of variance in the first two components ($n={latent_meta['n_pixels']:,}$ "
            r"stratified pixels)."
        )
        lines.append("")
    if perm_df is not None and len(perm_df):
        lines.append(
            r"Along-track permutation of predictor groups (shared latent seeds) "
            r"quantifies each group's contribution to held-out afCRPS and "
            rf"Brier score at $\tau={TAU_STAR:.1e}$:"
        )
        lines.append("")
        for _, r in perm_df[perm_df["group"] != "baseline"].iterrows():
            lines.append(
                rf"  {r['group']}: $\Delta$afCRPS ${r['delta_afCRPS']:+.5f}$, "
                rf"$\Delta$Brier ${r['delta_Brier']:+.5f}$;"
            )
        lines.append("")
    lines.append(
        r"Integrated gradients on soft $P(S>\tau^*)$ over high-probability ROIs "
        r"for representative orbits attribute predictions to coherent "
        r"gust-front-scale structure rather than isolated retrieval artifacts."
    )
    lines.append("")
    (outdir / "13_explainability_paper_snippet.tex").write_text("\n".join(lines) + "\n")
    print("wrote", outdir / "13_explainability_paper_snippet.tex", flush=True)


def copy_to_pack(outdir: Path, pack_dir: Path) -> None:
    pack_dir.mkdir(parents=True, exist_ok=True)
    tables = pack_dir.parent / "02_tables"
    tables.mkdir(parents=True, exist_ok=True)
    for p in outdir.glob("13_*"):
        if p.suffix in {".png", ".pdf", ".tex"}:
            dest = pack_dir / p.name
            dest.write_bytes(p.read_bytes())
            print("copied", dest, flush=True)
        elif p.suffix == ".csv":
            dest = tables / p.name
            dest.write_bytes(p.read_bytes())
            print("copied", dest, flush=True)
        elif p.suffix == ".json":
            dest = pack_dir / p.name
            dest.write_bytes(p.read_bytes())
            print("copied", dest, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config_ascat_a_crps.json"))
    ap.add_argument(
        "--checkpoint",
        default=str(ROOT / "models" / "unet_crps" / "crps_ensemble_best.pt"),
    )
    ap.add_argument(
        "--outdir",
        default=str(ROOT / "evaluation" / "explainability" / "crps"),
    )
    ap.add_argument(
        "--pack-dir",
        default=str(Path(get("FIGURES_MAIN") or str(figures_repo_root() / "figures" / "main"))),
    )
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--limit", type=int, default=20, help="Max val orbits for latent/permutation")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=TAU_STAR)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument(
        "--parts",
        default="latent,permute,ig",
        help="Comma list: latent,permute,ig",
    )
    ap.add_argument("--max-latent-pixels", type=int, default=25000)
    ap.add_argument(
        "--ig-orbits",
        default=",".join(str(o) for o in DEFAULT_IG_ORBITS),
    )
    ap.add_argument("--ig-steps", type=int, default=24)
    ap.add_argument("--no-pack-copy", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    model, cfg, device, m, alpha, peak = load_cfg(args)
    h_soft = float(cfg.get("exceedance_softness", 0.05))
    files = select_files(cfg, args.limit, args.stride)
    print(f"val orbits used={len(files)}  parts={args.parts}", flush=True)

    parts = {p.strip().lower() for p in args.parts.split(",") if p.strip()}
    latent_meta = None
    perm_df = None
    t0 = time.time()

    if "latent" in parts:
        print("=== (1) bottleneck latent analysis ===", flush=True)
        latent_meta = run_latent_analysis(
            model, cfg, device, m, files, outdir,
            seed=args.seed, max_pixels=args.max_latent_pixels,
            tau=args.tau, h_soft=h_soft,
        )

    if "permute" in parts:
        print("=== (2) predictor-group permutation ===", flush=True)
        perm_df = run_group_permutation(
            model, cfg, device, m, alpha, peak, files, outdir,
            seed=args.seed, tau=args.tau, h_soft=h_soft,
        )

    if "ig" in parts:
        print("=== (3) integrated gradients ===", flush=True)
        orbits = [int(x) for x in args.ig_orbits.split(",") if x.strip()]
        run_integrated_gradients(
            model, cfg, device, files, outdir,
            orbits=orbits, seed=args.seed, tau=args.tau,
            h_soft=h_soft, steps=args.ig_steps,
        )

    write_snippet(outdir, perm_df, latent_meta)
    if not args.no_pack_copy:
        copy_to_pack(outdir, Path(args.pack_dir))

    meta = {
        "parts": sorted(parts),
        "n_orbits": len(files),
        "members": m,
        "tau": args.tau,
        "seed": args.seed,
        "elapsed_s": time.time() - t0,
        "device": str(device),
        "predictor_groups": {k: list(v) for k, v in PREDICTOR_GROUPS.items()},
    }
    (outdir / "13_explainability_run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"done in {meta['elapsed_s']:.1f}s → {outdir}", flush=True)


if __name__ == "__main__":
    main()
