#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Evaluation helper for legacy CNN/CRNN checkpoints on MAD splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from core.models.legacy import CNNMFCCModel, CRNNMFCCModel, LegacyModelConfig, LegacyModelType
from core.data.preprocessing.legacy_mfcc import LegacyMFCCPreprocessor

from train_legacy_model import H5pyDataset, build_label_mapping  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate legacy CNN/CRNN checkpoints on MAD.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint.")
    parser.add_argument(
        "--model",
        choices=["cnn", "crnn"],
        default="cnn",
        help="Model type to instantiate before loading checkpoint.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="validation",
        help="Which MAD split to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "models" / "legacy_cnn_mfcc.yaml"),
        help="Model config yaml (used for MFCC settings).",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Evaluation device.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "evaluations"),
        help="Where to store reports (JSON + confusion matrix).",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_model(model_kind: str, config_dict: Dict, device: str, num_classes: int) -> torch.nn.Module:
    cfg = LegacyModelConfig(
        model_type=LegacyModelType.CNN if model_kind == "cnn" else LegacyModelType.CRNN,
        device=device,
    )
    if "mfcc" in config_dict:
        for key, value in config_dict["mfcc"].items():
            setattr(cfg.mfcc, key, value)
    if model_kind == "cnn" and "cnn" in config_dict:
        for key, value in config_dict["cnn"].items():
            setattr(cfg.cnn, key, value)
    if model_kind == "crnn" and "crnn" in config_dict:
        for key, value in config_dict["crnn"].items():
            setattr(cfg.crnn, key, value)

    if model_kind == "cnn":
        cfg.cnn.num_classes = num_classes
        return CNNMFCCModel(cfg)
    cfg.crnn.num_classes = num_classes
    return CRNNMFCCModel(cfg)


def prepare_dataset(
    split: str,
    config: Dict,
    label_mapping: Dict[str, int],
    model_kind: str,
) -> Tuple[H5pyDataset, Tuple[int, int, int]]:
    data_dir = REPO_ROOT / "data" / "processed" / "mad"
    split_file = data_dir / split / f"{split}.h5"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing MAD split file: {split_file}")

    mfcc_cfg = config.get("mfcc", {})
    duration = mfcc_cfg.get(
        "target_duration", 3.0
    ) if model_kind == "cnn" else mfcc_cfg.get("crnn_duration", 4.0)
    preprocessor = LegacyMFCCPreprocessor(
        sample_rate=mfcc_cfg.get("sample_rate", 16000),
        duration=duration,
        n_mfcc=mfcc_cfg.get("n_mfcc", 40),
        n_mels=mfcc_cfg.get("n_mels", 64),
        n_fft=mfcc_cfg.get("n_fft", 1024),
        hop_length=mfcc_cfg.get("hop_length", 512),
        use_deltas=mfcc_cfg.get("use_deltas", True),
        use_delta_deltas=mfcc_cfg.get("use_delta_deltas", True),
        normalize=mfcc_cfg.get("normalize", True),
    )
    output_shape = preprocessor.output_shape
    expected_shape = (output_shape[2], output_shape[0], output_shape[1])
    dataset = H5pyDataset(
        split_file,
        expected_shape=expected_shape,
        preprocessor=preprocessor,
        label_mapping=label_mapping,
    )
    return dataset, expected_shape


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, any]:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for features, labels in tqdm(loader, desc="Evaluating", unit="batch"):
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    avg_loss = total_loss / total
    accuracy = correct / total

    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )  # micro/macro metrics
    cm = confusion_matrix(y_true, y_pred)
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
    }


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    data_dir = REPO_ROOT / "data" / "processed" / "mad"
    train_file = data_dir / "train" / "train.h5"
    val_file = data_dir / "validation" / "validation.h5"
    label_mapping = build_label_mapping([train_file, val_file])

    dataset, _ = prepare_dataset(args.split, config, label_mapping, args.model)
    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=use_cuda,
    )

    num_classes = len(label_mapping)
    model = build_model(args.model, config, args.device, num_classes)
    checkpoint_path = Path(args.checkpoint)
    model.load_checkpoint(checkpoint_path, strict=True)
    device = torch.device(args.device if use_cuda else "cpu")
    model.to(device)

    results = evaluate(model, loader, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{checkpoint_path.stem}_{args.split}_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "split": args.split,
                "loss": results["loss"],
                "accuracy": results["accuracy"],
                "classification_report": results["report"],
            },
            handle,
            indent=2,
        )
    cm_path = output_dir / f"{checkpoint_path.stem}_{args.split}_confusion.npy"
    np.save(cm_path, results["confusion_matrix"])
    print(f"Validation loss: {results['loss']:.4f} | Accuracy: {results['accuracy']:.4f}")
    print(f"Saved classification report to {report_path}")
    print(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()
