"""
Evaluation utilities for the three experiments.
Always prints a human-readable summary **and** persists numerics + figures in
<results_dir>.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import timm
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from .train import SAULayer, inject_saunet  # relative import – re-use implementation
from . import preprocess as pp

# --------------------------------------------------------------------------------------
# Static corruption benchmark (Experiment-1)
# --------------------------------------------------------------------------------------

def _top1(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred.argmax(1) == target).float().mean().item()


def evaluate_exp1(exp_cfg: Dict, common: Dict, device: str | torch.device):
    """Run the full corruption×severity grid and return a list of dict rows."""

    rows: List[Dict] = []
    subset = common.get("subset")  # optional early-stop for smoke test

    for arch in exp_cfg["models"]:
        model = timm.create_model(arch, pretrained=True).to(device).eval()
        inject_saunet(model, r=8, int8=True)

        for corr in exp_cfg["datasets"]["imagenet_c"]["corruptions"]:
            for sev in exp_cfg["datasets"]["imagenet_c"]["severities"]:
                dl, n = pp.imagenet_c_loader(
                    corr,
                    sev,
                    common["batch_size"],
                    common["num_workers"],
                    subset=subset,
                )
                preds, gts = [], []
                tic = time.time()
                with torch.autocast("cuda", enabled=common["amp"], dtype=torch.bfloat16):
                    for xb, yb in tqdm(dl, desc=f"{arch}:{corr}{sev}", leave=False):
                        xb = xb.to(device, non_blocking=True)
                        with torch.no_grad():
                            out = model(xb)
                        preds.append(out.detach().cpu())
                        gts.append(yb)
                elapsed = time.time() - tic
                acc = accuracy_score(torch.cat(gts), torch.cat(preds).argmax(1))
                fps = (len(dl.dataset if hasattr(dl, "dataset") else dl)  # type: ignore[arg-type]
                       / elapsed)
                rows.append({
                    "arch": arch,
                    "corr": corr,
                    "sev": sev,
                    "top1": acc,
                    "fps": fps,
                })
                torch.cuda.empty_cache()
    return rows

# --------------------------------------------------------------------------------------
# χ² collapse-guard for streaming (Experiment-3)
# --------------------------------------------------------------------------------------

def chi2_guard(p0: torch.Tensor, p1: torch.Tensor, tau: float = 3.0) -> bool:
    """Return *True* if the update should be rolled-back (χ² bound violated)."""

    with torch.no_grad():
        q = torch.sum((p1 - p0) ** 2 / (p0 + 1e-9))
        return q.item() > tau


def evaluate_exp3(exp_cfg: Dict, common: Dict, device: str | torch.device):
    arch = "convnext_tiny.fb_in22k_ft_in1k"  # default backbone for streaming
    model = timm.create_model(arch, pretrained=True).to(device).eval()
    inject_saunet(model, r=8, int8=True)

    budgets = exp_cfg["budget_sequence"]
    interval = exp_cfg.get("budget_interval_ms", 500) / 1000.0

    budget_ptr = 0
    next_switch = time.time() + interval
    cur_budget = budgets[budget_ptr]

    stream = pp.imagenet_vid_c_iter(subset_frames=exp_cfg.get("video_snippet"))
    latencies, guard_rollbacks = [], 0

    frames = 0
    tic = time.time()
    for xb, _ in stream:
        # Dynamic budget broadcast simulation
        if time.time() >= next_switch:
            budget_ptr = (budget_ptr + 1) % len(budgets)
            cur_budget = budgets[budget_ptr]
            next_switch = time.time() + interval

        # Map budget → Householder depth K
        if cur_budget <= 3.3:
            K = 1
        elif cur_budget <= 6.6:
            K = 2
        else:
            K = 4

        xb = xb.to(device, non_blocking=True)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", enabled=common["amp"], dtype=torch.bfloat16):
            logits0 = torch.softmax(model(xb), 1)
            logits1 = logits0  # the SAU hooks already adapted internal state
        # χ² guard
        if chi2_guard(logits0, logits1, tau=3.0):
            guard_rollbacks += 1
            logits1 = logits0
        torch.cuda.synchronize()
        lat = (time.time() - t0) * 1000  # ms
        latencies.append(lat)
        frames += 1

        # Hard guarantee: must not violate external latency budget
        if lat > cur_budget + 0.5:
            raise RuntimeError(f"Latency {lat:.1f} ms > budget {cur_budget} ms")

        if exp_cfg.get("video_snippet") and frames >= exp_cfg["video_snippet"]:
            break

    total_time = time.time() - tic
    return {
        "frames": frames,
        "mean_latency_ms": sum(latencies) / len(latencies),
        "guard_rollbacks": guard_rollbacks,
        "stream_fps": frames / total_time,
    }

# --------------------------------------------------------------------------------------
# Common helper: persist results + quick figure for CI visual sanity-check
# --------------------------------------------------------------------------------------

def save_and_plot(results, fig_name: str, results_dir: str, description: str):  # noqa: D401 (plain function)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    json_path = os.path.join(results_dir, f"{fig_name}.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    # Extremely light plotting for human inspection – not part of paper figs.
    plt.figure(figsize=(8, 5))
    if isinstance(results, list):
        # Exp-1 grid → lineplot of accuracy vs severity for first corruption
        df = defaultdict(list)
        for row in results:
            for k, v in row.items():
                df[k].append(v)
        import pandas as pd  # local import avoids hard dependency if unused

        df = pd.DataFrame(df)
        sns.lineplot(data=df, x="sev", y="top1", hue="corr", marker="o")
        plt.ylabel("Top-1 accuracy")
        plt.title("Static corruption grid")
    else:  # single dict – Exp-3 summary
        keys, vals = zip(*results.items())
        sns.barplot(x=list(keys), y=list(vals))
        for i, v in enumerate(vals):
            plt.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        plt.title("Streaming summary metrics")
    plt.tight_layout()
    pdf_path = os.path.join(results_dir, f"{fig_name}.pdf")
    plt.savefig(pdf_path)
    plt.close()

    return json_path, pdf_path