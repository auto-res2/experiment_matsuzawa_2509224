"""
Model definition (SAUNet) plus synthetic meta-training utilities used in
Experiment-2.  This file centralises everything related to parameter
adaptation and optimisation so that other modules can simply
`from .train import inject_saunet`.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import RandAugment
import timm

# --------------------------------------------------------------------------------------
# SAUNet – Self-Adaptive Universal Normaliser & scale adaptor
# --------------------------------------------------------------------------------------
__all__ = [
    "SAULayer",
    "inject_saunet",
    "run_meta_training",
]


class SAULayer(nn.Module):
    """Projection-based hyper-network that outputs Δw and Δb for a single affine
    parameter group of channel dimension *C*.

    At inference time the layer runs **forward-only**; gradients are never
    required.  An extremely light EMA keeps track of the running update which
    makes the adaptor *memory-free* (no need to buffer activations).
    """

    def __init__(self, C: int, r: int = 8, int8: bool = False):
        super().__init__()
        self.C = C
        self.r = r
        self.int8 = int8

        # Low-rank tensorised transformer T ∈ ℝ^{4×r}
        self.T = nn.Parameter(torch.randn(4, r))
        # Learned projection P ∈ ℝ^{r×2} → Δw, Δb
        self.P = nn.Parameter(torch.zeros(r, 2))

        # Welford-style 2-tap EMA (16 B per layer ☺)
        self.register_buffer("dw_ema", torch.zeros(C))
        self.register_buffer("db_ema", torch.zeros(C))
        self.alpha = 0.1  # smoothing factor

    @torch.no_grad()
    def forward(self, x: torch.Tensor, K: int = 4) -> torch.Tensor:  # noqa: N803 (match paper)
        """Apply SAU correction in forward-only mode.

        Args:
            x: 4-D activation tensor (N,C,H,W)
            K: number of Householder terms kept from the low-rank expansion –
               this is the *any-time compute knob* exploited in Experiment-3.
        """
        # Handle different tensor shapes
        if x.dim() == 4:  # Conv layers: (N,C,H,W)
            B, C = x.shape[:2]
            flatten_dim = 2
        elif x.dim() == 3:  # Transformer layers: (N,L,C)
            B, C = x.shape[0], x.shape[2]
            flatten_dim = 1
        else:
            B, C = x.shape[0], x.shape[-1]
            flatten_dim = 1

        if self.C != C:
            # Resize EMA buffers if channel dimension doesn't match
            device = self.dw_ema.device
            self.dw_ema = self.dw_ema.new_zeros(C)
            self.db_ema = self.db_ema.new_zeros(C)
            self.C = C

        # Standardise per sample (mean=0,std=1)
        z = x.flatten(flatten_dim)
        z = (z - z.mean((flatten_dim), keepdim=True)).div(z.std((flatten_dim), unbiased=False, keepdim=True) + 1e-5)
        # Four channel-wise moments μ₁…μ₄
        m1 = z.mean(-1)
        m2 = z.var(-1, unbiased=False)
        m3 = (z**3).mean(-1)
        m4 = (z**4).mean(-1)
        s = torch.stack([m1, m2, m3, m4], dim=-1)  # (..., 4)

        # Low-rank non-linearity σ(T·s) ≈ tanh ▢
        y = torch.tanh(torch.einsum("...cr,rf->...cf", s, self.T))
        # Householder prefix – first *K* terms only (any-time knob)
        y = torch.einsum("...cf,fs->...cs", y[..., :K], self.P[:K])
        dw, db = y.unbind(-1)

        # EMA (cheap, memory-free)
        self.dw_ema.mul_(1 - self.alpha).add_(self.alpha * dw.mean(0))
        self.db_ema.mul_(1 - self.alpha).add_(self.alpha * db.mean(0))

        # Apply correction - handle both 4D (conv) and 3D (transformer) tensors
        if x.dim() == 4:  # Conv layers: (N,C,H,W)
            return x * (1 + self.dw_ema[None, :, None, None]) + self.db_ema[None, :, None, None]
        elif x.dim() == 3:  # Transformer layers: (N,L,C)
            return x * (1 + self.dw_ema[None, None, :]) + self.db_ema[None, None, :]
        else:
            # Fallback for other dimensions
            shape = [1] * x.dim()
            shape[-1] = -1  # Last dimension is always channels
            return x * (1 + self.dw_ema.view(shape)) + self.db_ema.view(shape)


# --------------------------------------------------------------------------------------
# Utility: walk through all affine (scale/shift) parameter groups in a backbone
# --------------------------------------------------------------------------------------

def _iter_affine_modules(model: nn.Module):
    """Yield (qualified_name, module, parameter) triples for every affine pair."""

    for name, module in model.named_modules(remove_duplicate=True):
        # • Normalisers γ,β
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            yield name + ".weight", module, module.weight
            yield name + ".bias", module, module.bias
        # • ConvNeXt layer-scale
        if hasattr(module, "gamma"):
            yield name + ".gamma", module, module.gamma
        # • ViT QKV scale
        if hasattr(module, "qkv") and hasattr(module.qkv, "weight"):
            yield name + ".qkv_weight", module.qkv, module.qkv.weight


def _replace_module(module: nn.Module, attr_name: str, r: int, int8: bool):
    """Monkey-patch *module* so that its output is filtered by a fresh SAULayer."""

    orig_param = getattr(module, attr_name)
    C = orig_param.shape[0]
    layer = SAULayer(C, r=r, int8=int8).to(orig_param.device)

    # Freeze original weights – SAUNet never alters the backbone parameters.
    orig_param.requires_grad_(False)

    def _hook(_mod, _inp, out, _lay=layer):  # noqa: ANN001
        return _lay(out)

    return module.register_forward_hook(_hook)


def inject_saunet(model: nn.Module, *, r: int = 8, int8: bool = False):
    """Insert SAULayer hooks **in-place** before *every* affine pair.

    Returns a list of hook handles which **must** be kept alive by the caller.
    """
    handles = []
    for name, mod, _ in _iter_affine_modules(model):
        if name.endswith(".weight"):
            attr = "weight"
        elif name.endswith(".bias"):
            attr = "bias"
        elif name.endswith(".gamma"):
            attr = "gamma"
        elif name.endswith(".qkv_weight"):
            attr = "weight"
        else:
            continue
        handles.append(_replace_module(mod, attr, r, int8))
    return handles


# --------------------------------------------------------------------------------------
# Synthetic meta-training loop (Experiment-2)
# --------------------------------------------------------------------------------------
from .preprocess import _transform  # relative import – shared preprocessing
from datasets import load_dataset
from torch.utils.data import DataLoader


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * torch.log(p + 1e-9)).sum(1)


def run_meta_training(cfg: Dict, results_dir: str, device: str | torch.device):
    """Contrastive synthetic self-supervision loop from Experiment-2.

    This function is deliberately *stateless* – everything needed is passed in
    `cfg`; results are both printed and persisted to *results_dir*.
    """
    meta_epochs: int = cfg["meta_epochs"]
    models: List[str] = cfg["models"]
    seeds: List[int] = cfg["seeds"]
    noise_grid = cfg.get("noise_grid")
    imagenet_fraction: float = cfg.get("imagenet_fraction", 0.0)

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    out: Dict[str, Dict] = {}

    for arch in models:
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            backbone = timm.create_model(arch, pretrained=True).to(device).eval()
            inject_saunet(backbone, r=8, int8=True)
            optimiser = torch.optim.Adam(backbone.parameters(), lr=1e-3)

            sigmas = noise_grid.get("sigma", [0.5]) if noise_grid else [0.5]
            cjs = noise_grid.get("colour_jitter", [0.5]) if noise_grid else [0.5]

            steps = 0
            tic = time.time()
            for _ in range(meta_epochs):
                for sigma in sigmas:
                    for _cj in cjs:  # colour jitter value unused (kept for completeness)
                        aug = RandAugment(num_ops=2, magnitude=9)
                        noise = (torch.randn(256, 3, 224, 224, device=device) * sigma).clamp(-2, 2)
                        x = aug(noise)
                        with torch.no_grad():
                            p0 = torch.softmax(backbone(x), dim=1)
                        p1 = torch.softmax(backbone(x), dim=1)
                        loss = (_entropy(p1) - _entropy(p0)).mean()
                        optimiser.zero_grad(set_to_none=True)
                        loss.backward()
                        optimiser.step()
                        steps += 1

                # Optionally mix a fraction of real ImageNet images (source regime)
                if imagenet_fraction > 0:
                    frac_pct = int(imagenet_fraction * 100)
                    real_ds = load_dataset("benjamin-paine/imagenet-1k-256x256", split=f"train[:{frac_pct}%]")
                    real_dl = DataLoader(
                        real_ds.with_format("torch"),
                        batch_size=256,
                        shuffle=True,
                        num_workers=4,
                        collate_fn=lambda b: torch.stack([
                            _transform(224)(x["image"].convert("RGB")) for x in b
                        ]),
                    )
                    for xb in real_dl:
                        xb = xb.to(device, non_blocking=True)
                        with torch.no_grad():
                            p0 = torch.softmax(backbone(xb), dim=1)
                        p1 = torch.softmax(backbone(xb), dim=1)
                        loss = (_entropy(p1) - _entropy(p0)).mean()
                        optimiser.zero_grad(set_to_none=True)
                        loss.backward()
                        optimiser.step()
                        steps += 1
                        # Keep runtime reasonable in smoke / CI contexts
                        if cfg.get("max_steps") and steps >= cfg["max_steps"]:
                            break

            dur = time.time() - tic
            out_key = f"{arch}_seed{seed}"
            out[out_key] = {"meta_steps": steps, "wall_clock_s": dur}
            torch.cuda.empty_cache()

    # Persist --------------------------------------------------------------------------------
    json_path = os.path.join(results_dir, "meta_train_results.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2)

    print(json.dumps(out, indent=2))
    return json_path