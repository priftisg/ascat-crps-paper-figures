"""Checkpoint loaders for Stage 1 and Stage 2."""

import torch

from .model import build_full_swath_unet, build_stage1_unet


def _config(ckpt, path):
    config = ckpt.get("config")
    if config is None:
        raise ValueError(f"Checkpoint {path} has no embedded config")
    return config


def load_stage1_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = _config(ckpt, path)
    if ckpt.get("model_kind") != "stage1_multitask":
        raise ValueError(f"{path} is not a stage1_multitask checkpoint")
    model = build_stage1_unet(
        cfg["stage1_num_channels"], cfg["filters"], cfg["num_layers"],
        cfg["pool_along_track"], cfg["pool_cross_track"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg


def load_stage2_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = _config(ckpt, path)
    if ckpt.get("model_kind") not in {"stage2_proxy", "stage2_buoy_finetuned"}:
        raise ValueError(f"{path} is not a Stage-2 checkpoint")
    model = build_full_swath_unet(
        cfg["stage2_num_channels"], cfg["filters"], cfg["num_layers"],
        cfg["pool_along_track"], cfg["pool_cross_track"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg
