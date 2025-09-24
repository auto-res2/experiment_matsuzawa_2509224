"""Entry-point orchestrating smoke vs full experiments.

Usage:
  uv run python -m src.main --smoke-test
  uv run python -m src.main --full-experiment
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap
from typing import Dict

import torch
import yaml

from .evaluate import evaluate_exp1, evaluate_exp3, save_and_plot
from .train import run_meta_training


# --------------------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------------------

def _pretty(obj):
    return json.dumps(obj, indent=2)


def _load_cfg(smoke: bool):
    cfg_file = "config/smoke_test.yaml" if smoke else "config/full_experiment.yaml"
    with open(cfg_file, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp), cfg_file


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():  # noqa: D401 (simple CLI)
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="run quick CI smoke-test")
    parser.add_argument("--full-experiment", action="store_true", help="run full paper experiments")
    args = parser.parse_args()

    smoke = args.smoke_test or not args.full_experiment
    cfg, cfg_path = _load_cfg(smoke)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    common = cfg["common"]

    print("=" * 80)
    print("SAUNet – unified forward-only adaptation experiments")
    print("Configuration :", cfg_path)
    print("Device        :", device)
    print("=" * 80)

    # ------------------------------------------------------------------ EXP-1
    if cfg["exp1"]["enable"]:
        print("\nExp-1  Breadth-of-Coverage  ", "=" * 47)
        res1 = evaluate_exp1(cfg["exp1"], common, device)
        j1, pdf1 = save_and_plot(res1, "exp1_accuracy", common["results_dir"], "Top-1 over corruption grid")
        print("Results (first 3 rows):", _pretty(res1[:3]), "...", sep="\n")
        print("Figure saved         :", Path(pdf1).name)

    # ------------------------------------------------------------------ EXP-2
    if cfg["exp2"]["enable"]:
        print("\nExp-2  Privacy-preserving meta-training  ", "=" * 34)
        meta_json = run_meta_training({**cfg["exp2"], **common}, common["results_dir"], device)
        print("Meta-training summary written to:", meta_json)

    # ------------------------------------------------------------------ EXP-3
    if cfg["exp3"]["enable"]:
        print("\nExp-3  Adaptive streaming & χ² guard  ", "=" * 34)
        res3 = evaluate_exp3(cfg["exp3"], common, device)
        j3, pdf3 = save_and_plot(res3, "exp3_stream", common["results_dir"], "Latency-aware streaming")
        print("Results:", _pretty(res3))
        print("Figure saved:", Path(pdf3).name)


if __name__ == "__main__":
    main()