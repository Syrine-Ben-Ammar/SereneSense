#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Plot legacy training curves from JSON history files produced by train_legacy_model.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train/validation curves from history JSON files."
    )
    parser.add_argument(
        "--histories",
        nargs="+",
        required=True,
        help="One or more JSON history files (outputs/history/*.json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the plot (PNG/PDF). If omitted, the plot is shown interactively.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Legacy Training Curves",
        help="Figure title.",
    )
    return parser.parse_args()


def load_history(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_histories(history_paths: List[Path], title: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    for history_path in history_paths:
        history = load_history(history_path)
        label = history_path.stem
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label=f"{label} train")
        axes[0].plot(epochs, history["val_loss"], label=f"{label} val")
        axes[1].plot(epochs, history["val_accuracy"], label=f"{label}")
        best_epoch = history.get("best_epoch")
        best_acc = history.get("best_accuracy")
        if best_epoch and best_acc is not None:
            axes[1].scatter([best_epoch], [best_acc], marker="o", color=axes[1].lines[-1].get_color())

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].grid(True, alpha=0.3)

    axes[0].legend()
    axes[1].legend()
    fig.suptitle(title)
    return fig


def main() -> None:
    args = parse_args()
    history_paths = [Path(p).expanduser().resolve() for p in args.histories]
    fig = plot_histories(history_paths, args.title)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
