import argparse
import os
from pathlib import Path

import yaml

from .preprocess import load_data
from .train import train
from .evaluate import evaluate


def _load_cfg(path: Path):
    with path.open() as fp:
        return yaml.safe_load(fp)


def run(cfg_path: Path, tag: str):
    cfg = _load_cfg(cfg_path)
    data = load_data(cfg)
    model, _ = train(data, cfg, tag=tag)
    evaluate(model, data, tag=tag)


def main():
    parser = argparse.ArgumentParser(description="SAFE-VALS experiment runner")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    group.add_argument("--full-experiment", action="store_true", help="Run full experiment only")
    args = parser.parse_args()

    cfg_dir = Path("config")
    smoke_cfg = cfg_dir / "smoke_test.yaml"
    full_cfg = cfg_dir / "full_experiment.yaml"

    if args.smoke_test:
        run(smoke_cfg, tag="smoke_test")
    elif args.full_experiment:
        run(full_cfg, tag="full_experiment")
    else:
        # Two-phase: smoke then full
        run(smoke_cfg, tag="smoke_test")
        run(full_cfg, tag="full_experiment")


if __name__ == "__main__":
    main()
