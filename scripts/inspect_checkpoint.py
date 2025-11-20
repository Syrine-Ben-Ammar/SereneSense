#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Utility to inspect a saved legacy checkpoint:
  - prints model metadata and stored metrics
  - verifies state dict integrity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.models.legacy import CNNMFCCModel, CRNNMFCCModel, LegacyModelConfig, LegacyModelType


def override_config(cfg: LegacyModelConfig, overrides: Dict[str, Any]) -> LegacyModelConfig:
    if not overrides:
        return cfg
    if "mfcc" in overrides:
        for key, value in overrides["mfcc"].items():
            setattr(cfg.mfcc, key, value)
    if "cnn" in overrides:
        for key, value in overrides["cnn"].items():
            setattr(cfg.cnn, key, value)
    if "crnn" in overrides:
        for key, value in overrides["crnn"].items():
            setattr(cfg.crnn, key, value)
    if "spec_augment" in overrides:
        for key, value in overrides["spec_augment"].items():
            setattr(cfg.spec_augment, key, value)
    return cfg


def load_overrides(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a legacy checkpoint file.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML overrides (applied on top of checkpoint config).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used when instantiating the model for validation.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to dump checkpoint metadata as JSON.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg: LegacyModelConfig = raw_checkpoint.get("config")
    if cfg is None:
        raise RuntimeError(
            "Checkpoint does not carry a serialized LegacyModelConfig; please re-train with the updated script."
        )
    cfg.device = args.device
    cfg = override_config(cfg, load_overrides(Path(args.config)) if args.config else {})

    model_cls = CNNMFCCModel if cfg.model_type == LegacyModelType.CNN else CRNNMFCCModel
    model = model_cls(cfg)
    model.load_state_dict(raw_checkpoint["model_state_dict"], strict=True)
    model.to(torch.device(args.device if torch.cuda.is_available() else "cpu"))

    summary = {
        "checkpoint": str(checkpoint_path),
        "model": model.model_name,
        "parameters": model.count_parameters(),
        "epoch": raw_checkpoint.get("epoch"),
        "metrics": raw_checkpoint.get("metrics", {}),
        "num_classes": (
            cfg.cnn.num_classes if cfg.model_type == LegacyModelType.CNN else cfg.crnn.num_classes
        ),
    }

    print(json.dumps(summary, indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Metadata written to {out_path}")


if __name__ == "__main__":
    main()
