#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Offline MFCC feature caching for MAD/FSD50K HDF5 splits.

This script materialises MFCC tensors inside each HDF5 file so that
`train_legacy_model.py` can read spectrogram-like features directly
instead of recomputing them in every DataLoader worker.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.data.preprocessing.legacy_mfcc import LegacyMFCCPreprocessor  # noqa: E402


LOGGER = logging.getLogger("cache_mfcc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-compute MFCC tensors and store them in existing HDF5 splits."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="HDF5 files or glob patterns to process (e.g. data/processed/mad/train/train.h5).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "models" / "legacy_cnn_mfcc.yaml"),
        help="Model config file containing the MFCC settings (defaults to the CNN config).",
    )
    parser.add_argument(
        "--model-kind",
        choices=["cnn", "crnn"],
        default="cnn",
        help="Select which MFCC duration to use when reading the config file.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Override the target duration in seconds. If omitted, uses the config value.",
    )
    parser.add_argument(
        "--audio-key",
        type=str,
        default="audio",
        help="Dataset name inside the HDF5 file that stores raw waveforms.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="mfcc",
        help="Name for the cached MFCC dataset that will be created inside each file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of waveforms to convert per iteration.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Chunk size (first dimension) for the MFCC dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute MFCC caches even if the target dataset already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect files and report planned actions without writing any data.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def resolve_files(patterns: Iterable[str]) -> List[Path]:
    resolved: List[Path] = []
    for pattern in patterns:
        path = Path(pattern).expanduser()
        if path.is_file():
            resolved.append(path)
            continue
        if path.is_dir():
            resolved.extend(sorted(path.rglob("*.h5")))
            continue
        matches = sorted(path.parent.glob(path.name)) if any(ch in pattern for ch in "*?[]") else []
        if matches:
            resolved.extend(matches)
        else:
            LOGGER.warning("No files matched pattern: %s", pattern)
    return sorted({p.resolve() for p in resolved})


def load_mfcc_kwargs(config_path: Path, model_kind: str, duration_override: float | None) -> Dict[str, float]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    mfcc_cfg: Dict[str, float] = config.get("mfcc", {})
    duration_key = "target_duration" if model_kind == "cnn" else "crnn_duration"
    duration_default = 3.0 if model_kind == "cnn" else 4.0
    return {
        "sample_rate": mfcc_cfg.get("sample_rate", 16000),
        "duration": duration_override or mfcc_cfg.get(duration_key, duration_default),
        "n_mfcc": mfcc_cfg.get("n_mfcc", 40),
        "n_mels": mfcc_cfg.get("n_mels", 64),
        "n_fft": mfcc_cfg.get("n_fft", 1024),
        "hop_length": mfcc_cfg.get("hop_length", 512),
        "use_deltas": mfcc_cfg.get("use_deltas", True),
        "use_delta_deltas": mfcc_cfg.get("use_delta_deltas", True),
        "normalize": mfcc_cfg.get("normalize", True),
    }


def ensure_logger(level: str) -> None:
    LOGGER.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)


def process_file(
    file_path: Path,
    preprocessor: LegacyMFCCPreprocessor,
    audio_key: str,
    dataset_name: str,
    batch_size: int,
    chunk_size: int,
    overwrite: bool,
    dry_run: bool,
) -> None:
    open_mode = "r" if dry_run else "r+"
    with h5py.File(file_path, open_mode) as handle:
        if audio_key not in handle:
            raise KeyError(f"Dataset '{audio_key}' not present in {file_path}")

        if dataset_name in handle:
            if overwrite:
                if dry_run:
                    LOGGER.info("[%s] would remove existing '%s' dataset", file_path.name, dataset_name)
                else:
                    del handle[dataset_name]
                    LOGGER.info("[%s] removed existing '%s' dataset", file_path.name, dataset_name)
            else:
                LOGGER.info("[%s] cache already exists, skipping. Use --overwrite to recompute.", file_path.name)
                return

        audio_ds = handle[audio_key]
        num_samples, num_points = audio_ds.shape
        if num_points != preprocessor.n_samples:
            LOGGER.warning(
                "[%s] audio length (%d) differs from MFCC target (%d). Samples will be padded or trimmed.",
                file_path.name,
                num_points,
                preprocessor.n_samples,
            )

        channels = preprocessor.output_shape[2]
        freq = preprocessor.output_shape[0]
        time_bins = preprocessor.output_shape[1]

        if dry_run:
            LOGGER.info(
                "[%s] would create dataset '%s' with shape (%d, %d, %d, %d)",
                file_path.name,
                dataset_name,
                num_samples,
                channels,
                freq,
                time_bins,
            )
            return

        chunk = max(1, min(chunk_size, num_samples))
        cache_ds = handle.create_dataset(
            dataset_name,
            shape=(num_samples, channels, freq, time_bins),
            dtype="float32",
            chunks=(chunk, channels, freq, time_bins),
            compression="gzip",
            compression_opts=4,
        )
        cache_ds.attrs["source"] = audio_key
        cache_ds.attrs["description"] = "Legacy MFCC cache (channels, freq, time)"

        LOGGER.info(
            "[%s] caching MFCC tensors into '%s' (%d samples)...",
            file_path.name,
            dataset_name,
            num_samples,
        )
        steps = math.ceil(num_samples / batch_size)
        progress = tqdm(total=num_samples, desc=file_path.name, unit="sample")
        for step in range(steps):
            start = step * batch_size
            end = min(start + batch_size, num_samples)
            batch = audio_ds[start:end]
            features = np.empty((end - start, channels, freq, time_bins), dtype=np.float32)
            for idx, waveform in enumerate(batch):
                mfcc = preprocessor.process_audio(waveform)
                features[idx] = np.transpose(mfcc, (2, 0, 1))
            cache_ds[start:end] = features
            progress.update(end - start)
        progress.close()
        LOGGER.info("[%s] completed MFCC cache.", file_path.name)


def main() -> None:
    args = parse_args()
    ensure_logger(args.log_level)

    target_files = resolve_files(args.files)
    if not target_files:
        LOGGER.error("No HDF5 files to process. Please double-check the --files argument.")
        sys.exit(1)

    mfcc_kwargs = load_mfcc_kwargs(Path(args.config), args.model_kind, args.duration)
    preprocessor = LegacyMFCCPreprocessor(**mfcc_kwargs)
    LOGGER.info(
        "Using MFCC config: sr=%d, duration=%.2fs, n_mfcc=%d, n_mels=%d, hop=%d, deltas=%s/%s",
        mfcc_kwargs["sample_rate"],
        mfcc_kwargs["duration"],
        mfcc_kwargs["n_mfcc"],
        mfcc_kwargs["n_mels"],
        mfcc_kwargs["hop_length"],
        "on" if mfcc_kwargs["use_deltas"] else "off",
        "on" if mfcc_kwargs["use_delta_deltas"] else "off",
    )

    for file_path in target_files:
        try:
            process_file(
                file_path=file_path,
                preprocessor=preprocessor,
                audio_key=args.audio_key,
                dataset_name=args.dataset_name,
                batch_size=max(1, args.batch_size),
                chunk_size=max(1, args.chunk_size),
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            LOGGER.exception("Failed to process %s: %s", file_path, exc)
            sys.exit(1)


if __name__ == "__main__":
    main()
