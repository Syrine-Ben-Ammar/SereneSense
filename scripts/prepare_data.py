#!/usr/bin/env python3
"""
Dataset preparation pipeline for SereneSense.

Features:
    - Audio validation (sample rate, channels, duration)
    - Resampling to 16 kHz mono with optional duration normalization
    - Silence removal and amplitude normalization
    - Stratified train/validation/test split with configurable ratios
    - Optional data augmentation for offline expansion
    - Chunked HDF5 and Zarr export for fast training data access
    - JSON manifests and dataset-level statistics
    - Multi-process workers with progress tracking
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import dataclasses
import json
import logging
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import librosa
import numpy as np
import soundfile as sf
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import zarr
except Exception:  # pragma: no cover - optional dependency
    zarr = None


LOGGER_NAME = "serenesense.data_prep"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_DURATION = 10.0
DEFAULT_CHANNELS = 1
EPS = 1e-10


@dataclass
class AudioSample:
    """Represents a single raw audio file entry."""

    path: Path
    label: str
    dataset: str
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def uid(self) -> str:
        base = self.path.stem
        return f"{self.dataset}_{self.index:08d}_{base}"


@dataclass
class ProcessedSample:
    """Processed audio ready for storage."""

    audio: np.ndarray
    label: str
    sample_id: str
    source_path: str
    metadata: Dict[str, Any]


@dataclass
class ProcessingConfig:
    """Configuration governing audio processing behaviour."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    duration: float = DEFAULT_DURATION
    normalize: bool = True
    silence_trim: bool = True
    silence_db: float = 30.0
    augmentations: Sequence[str] = dataclasses.field(default_factory=list)
    target_format: str = "wav"
    chunk_size: int = 64
    seed: int = 42
    create_zarr: bool = False

    @property
    def target_num_samples(self) -> int:
        return int(self.sample_rate * self.duration)


def setup_logger(log_dir: Path, verbose: bool = False) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"prepare_data_{timestamp}.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(file_handler)

    logger.debug("Logging initialized: %s", log_path)
    return logger


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def canonical_label(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def pad_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
    if audio.shape[-1] == target_length:
        return audio
    if audio.shape[-1] > target_length:
        return audio[..., :target_length]
    pad_width = target_length - audio.shape[-1]
    return np.pad(audio, (0, pad_width), mode="constant")


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    if not audio.size:
        return audio
    mean = float(np.mean(audio))
    std = float(np.std(audio))
    if std < EPS:
        std = 1.0
    return (audio - mean) / std


def trim_silence(audio: np.ndarray, top_db: float) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    if trimmed.size == 0:
        return audio
    return trimmed


def apply_augmentation(
    audio: np.ndarray,
    augmentation: str,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if augmentation == "gaussian_noise":
        noise_level = 0.005 * rng.random()
        return audio + rng.normal(0.0, noise_level, size=audio.shape)
    if augmentation == "time_shift":
        shift = int(len(audio) * rng.uniform(-0.1, 0.1))
        return np.roll(audio, shift)
    if augmentation == "pitch_shift":
        steps = rng.uniform(-2.0, 2.0)
        return librosa.effects.pitch_shift(audio, sample_rate, steps)
    if augmentation == "time_stretch":
        rate = rng.uniform(0.9, 1.1)
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        return pad_or_trim(stretched, len(audio))
    return audio


def process_audio_sample(sample: AudioSample, config: ProcessingConfig) -> List[ProcessedSample]:
    rng = np.random.default_rng(config.seed + sample.index)
    try:
        info = sf.info(sample.path)
    except Exception as exc:
        raise RuntimeError(f"Failed to inspect {sample.path}: {exc}") from exc
    if info.samplerate <= 0:
        raise RuntimeError(f"Invalid sample rate detected for {sample.path}: {info.samplerate}")
    if info.channels <= 0:
        raise RuntimeError(f"Invalid channel count for {sample.path}: {info.channels}")

    try:
        waveform, sr = librosa.load(sample.path, sr=None, mono=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {sample.path}: {exc}") from exc

    original_channels = 1 if waveform.ndim == 1 else waveform.shape[0]
    if waveform.ndim > 1:
        waveform = librosa.to_mono(waveform)

    if sr != config.sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=config.sample_rate)

    if config.silence_trim:
        waveform = trim_silence(waveform, config.silence_db)

    waveform = pad_or_trim(waveform, config.target_num_samples)

    results: List[ProcessedSample] = []
    variants = [("orig", waveform)]

    for aug in config.augmentations:
        augmented = apply_augmentation(waveform, aug, config.sample_rate, rng)
        augmented = pad_or_trim(augmented, config.target_num_samples)
        variants.append((aug, augmented))

    for suffix, audio in variants:
        processed = audio.astype(np.float32)
        if config.normalize:
            processed = normalize_audio(processed)
        sample_id = sample.uid if suffix == "orig" else f"{sample.uid}__{suffix}"
        metadata = dict(sample.metadata)
        metadata.update(
            {
                "original_sample_rate": sr,
                "original_channels": original_channels,
                "original_duration_seconds": float(info.duration),
                "processed_sample_rate": config.sample_rate,
                "duration_seconds": len(processed) / config.sample_rate,
                "augmentation": suffix,
            }
        )
        results.append(
            ProcessedSample(
                audio=processed,
                label=sample.label,
                sample_id=sample_id,
                source_path=str(sample.path),
                metadata=metadata,
            )
        )

    return results


class StatisticsAccumulator:
    """Accumulates descriptive statistics over processed samples."""

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = math.inf
        self.max = -math.inf
        self.duration_sum = 0.0
        self.class_counts: Counter[str] = Counter()
        self.per_class_duration: Counter[str] = Counter()

    def update(self, sample: ProcessedSample) -> None:
        audio = sample.audio
        self.count += 1
        self.sum += float(np.sum(audio))
        self.sumsq += float(np.sum(np.square(audio)))
        self.min = min(self.min, float(np.min(audio)))
        self.max = max(self.max, float(np.max(audio)))
        duration = float(sample.metadata.get("duration_seconds", len(audio)))
        self.duration_sum += duration
        self.class_counts[sample.label] += 1
        self.per_class_duration[sample.label] += duration

    def to_dict(self) -> Dict[str, Any]:
        mean = self.sum / max(self.count, 1)
        mean_square = self.sumsq / max(self.count, 1)
        variance = max(mean_square - mean**2, 0.0)
        std = math.sqrt(variance)
        class_distribution = {
            label: {
                "count": count,
                "percentage": (count / self.count) * 100 if self.count else 0.0,
                "avg_duration": (self.per_class_duration[label] / count if count else 0.0),
            }
            for label, count in self.class_counts.items()
        }
        return {
            "total_samples": self.count,
            "mean": mean,
            "std": std,
            "min": float(self.min if self.count else 0.0),
            "max": float(self.max if self.count else 0.0),
            "average_duration": self.duration_sum / max(self.count, 1),
            "class_distribution": class_distribution,
        }


class HDF5SplitWriter:
    """Handles chunked writing of processed audio into HDF5 format."""

    def __init__(self, path: Path, target_samples: int, chunk_size: int) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, "w")
        self.audio_ds = self.file.create_dataset(
            "audio",
            shape=(0, target_samples),
            maxshape=(None, target_samples),
            dtype="float32",
            chunks=(chunk_size, target_samples),
            compression="gzip",
            compression_opts=4,
        )
        self.labels_ds = self.file.create_dataset(
            "labels",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype("utf-8", length=32),
            chunks=(chunk_size,),
        )
        self.ids_ds = self.file.create_dataset(
            "sample_ids",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype("utf-8", length=64),
            chunks=(chunk_size,),
        )
        self._count = 0
        self._manifest: List[Dict[str, Any]] = []

    def append(self, sample: ProcessedSample) -> None:
        idx = self._count
        self.audio_ds.resize(idx + 1, axis=0)
        self.labels_ds.resize(idx + 1, axis=0)
        self.ids_ds.resize(idx + 1, axis=0)
        self.audio_ds[idx] = sample.audio
        self.labels_ds[idx] = sample.label
        self.ids_ds[idx] = sample.sample_id
        self._manifest.append(
            {
                "sample_id": sample.sample_id,
                "label": sample.label,
                "source_path": sample.source_path,
                **sample.metadata,
            }
        )
        self._count += 1

    def close(self) -> List[Dict[str, Any]]:
        self.file.flush()
        self.file.close()
        return self._manifest


class ZarrSplitWriter:
    """Optional Zarr writer mirroring HDF5 output."""

    def __init__(self, path: Path, target_samples: int, chunk_size: int) -> None:
        if zarr is None:
            raise RuntimeError("Zarr is not installed; install zarr>=2.16.0 to enable.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.group = zarr.open(str(path), mode="w")
        self.audio = self.group.create_dataset(
            "audio",
            shape=(0, target_samples),
            maxshape=(None, target_samples),
            chunks=(chunk_size, target_samples),
            dtype="float32",
            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
        )
        self.labels = self.group.create_dataset(
            "labels",
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_size,),
            dtype="U32",
        )
        self.ids = self.group.create_dataset(
            "sample_ids",
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_size,),
            dtype="U64",
        )
        self._count = 0

    def append(self, sample: ProcessedSample) -> None:
        idx = self._count
        self.audio.resize(idx + 1, axis=0)
        self.labels.resize(idx + 1, axis=0)
        self.ids.resize(idx + 1, axis=0)
        self.audio[idx] = sample.audio
        self.labels[idx] = sample.label
        self.ids[idx] = sample.sample_id
        self._count += 1


class BaseDatasetPreparer:
    """Base class for dataset-specific collection logic."""

    NAME = "base"

    def __init__(self, config: Dict[str, Any], data_root: Path, args: argparse.Namespace) -> None:
        self.config = config
        self.data_root = data_root
        self.args = args

    def collect_samples(self) -> List[AudioSample]:
        raise NotImplementedError

    def _iter_audio_files(self, base_dir: Path) -> Iterable[Path]:
        for extension in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
            yield from base_dir.rglob(extension)


class MADPreparer(BaseDatasetPreparer):
    NAME = "mad"

    def __init__(self, config: Dict[str, Any], data_root: Path, args: argparse.Namespace) -> None:
        super().__init__(config=config, data_root=data_root, args=args)
        extraction_cfg = self.config.get("source", {}).get("extraction", {})
        target_dir = Path(extraction_cfg.get("target_dir", "data/raw/mad"))
        self.base_path = target_dir if target_dir.is_absolute() else self.data_root.parent / target_dir
        self._label_lookup = self._load_csv_label_lookup(self.base_path)
        self.label_metadata = self._build_label_metadata()

    def _build_label_metadata(self) -> Dict[str, Any]:
        class_defs = self.config.get("classes", {}).get("definitions", {})
        class_names = {
            str(int(idx)): canonical_label(info.get("name", str(idx)))
            for idx, info in class_defs.items()
        }
        if not class_names and self._label_lookup:
            # Fall back to discovered labels if config is missing
            discovered = {str(label_id) for label_id in self._label_lookup.values()}
            class_names = {label_id: label_id for label_id in discovered}
        return {
            "class_names": class_names,
            "num_classes": len(class_names),
        }

    def _load_csv_label_lookup(self, base_path: Path) -> Dict[str, int]:
        lookup: Dict[str, int] = {}
        csv_files = ["training.csv", "validation.csv", "test.csv"]
        for csv_name in csv_files:
            csv_path = base_path / csv_name
            if not csv_path.exists():
                continue
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "path" not in reader.fieldnames or "label" not in reader.fieldnames:
                    continue
                for row in reader:
                    rel_path = (row.get("path") or "").strip()
                    label_value = row.get("label")
                    if not rel_path or label_value is None:
                        continue
                    try:
                        label_id = int(label_value)
                    except ValueError:
                        continue
                    resolved = (base_path / rel_path.replace("/", os.sep)).resolve()
                    lookup[str(resolved)] = label_id
        if not lookup:
            logging.getLogger(__name__).warning(
                "MAD label lookup could not be built from CSV files. "
                "Falling back to directory names (may yield incorrect labels)."
            )
        return lookup

    def collect_samples(self) -> List[AudioSample]:
        base_path = self.base_path
        samples: List[AudioSample] = []
        id_to_name = self.label_metadata.get("class_names", {})
        for idx, file_path in enumerate(self._iter_audio_files(base_path)):
            lookup_key = str(file_path.resolve())
            label_id = self._label_lookup.get(lookup_key)
            if label_id is None:
                logging.getLogger(__name__).warning(
                    "Skipping %s because no label was found in MAD CSV metadata.", file_path
                )
                continue
            label_str = str(label_id)
            class_name = id_to_name.get(label_str, label_str)
            samples.append(
                AudioSample(
                    path=file_path,
                    label=label_str,
                    dataset=self.NAME,
                    index=idx,
                    metadata={
                        "relative_path": str(file_path.relative_to(base_path)),
                        "class_name": class_name,
                    },
                )
            )
        return samples


class FSD50KPreparer(BaseDatasetPreparer):
    NAME = "fsd50k"

    def collect_samples(self) -> List[AudioSample]:
        extraction_cfg = self.config.get("source", {}).get("extraction", {})
        base_dir = Path(extraction_cfg.get("target_dir", "data/raw/fsd50k"))
        base_path = base_dir if base_dir.is_absolute() else self.data_root.parent / base_dir
        mapping_file = self.config.get("integration", {}).get("mapping_file")
        label_mapper = load_mapping(mapping_file) if mapping_file else {}

        subsets = self.config.get("subsets", {})
        selected_subset = (
            self.args.subsets[0]
            if self.args.subsets
            else self.config.get("integration", {}).get("default_subset", "dev")
        )
        subset_cfg = subsets.get(selected_subset)
        if not subset_cfg:
            raise RuntimeError(f"Unknown FSD50K subset: {selected_subset}")

        ground_truth_path = base_path / subset_cfg.get("ground_truth", "")

        # Check if structured directory exists, otherwise use base_path directly
        audio_dir_name = subset_cfg.get("audio_dir", "")
        audio_dir = base_path / audio_dir_name
        if not audio_dir.exists() and base_path.exists():
            # Fallback: audio files might be directly in base_path
            audio_dir = base_path
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Expected audio directory not found: {base_path / audio_dir_name}. "
                f"Using base directory: {audio_dir}"
            )

        if not ground_truth_path.exists():
            # Fallback: if ground truth CSV is missing, infer labels from filename or use 'unknown'
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Ground truth CSV not found: {ground_truth_path}. "
                f"Will attempt to infer labels from filenames or use 'unknown' label."
            )
            return self._collect_samples_without_metadata(audio_dir, label_mapper)

        metadata = load_fsd50k_metadata(ground_truth_path)
        samples: List[AudioSample] = []
        for idx, (filename, labels) in enumerate(metadata.items()):
            normalized_name = filename.strip()
            if not normalized_name.lower().endswith(".wav"):
                normalized_name = f"{normalized_name}.wav"
            file_path = audio_dir / normalized_name
            if not file_path.exists():
                continue
            canonical = map_label(labels, label_mapper)
            samples.append(
                AudioSample(
                    path=file_path,
                    label=canonical,
                    dataset=self.NAME,
                    index=idx,
                    metadata={"subset": selected_subset, "labels": list(labels)},
                )
            )
        return samples

    def _collect_samples_without_metadata(
        self, audio_dir: Path, label_mapper: Dict[str, str]
    ) -> List[AudioSample]:
        """Fallback method when ground truth CSV is missing."""
        samples: List[AudioSample] = []
        for idx, file_path in enumerate(self._iter_audio_files(audio_dir)):
            # Try to infer label from parent directory or use 'unknown'
            label = canonical_label(file_path.parent.name)
            if label == audio_dir.name.lower():
                # If parent is the base directory, use 'unknown'
                label = "unknown"
            mapped = label_mapper.get(label, label)
            samples.append(
                AudioSample(
                    path=file_path,
                    label=mapped,
                    dataset=self.NAME,
                    index=idx,
                    metadata={"inferred_label": True, "source": "filename"},
                )
            )
        return samples


class AudioSetPreparer(BaseDatasetPreparer):
    NAME = "audioset"

    def collect_samples(self) -> List[AudioSample]:
        source_cfg = self.config.get("source", {})
        base_dir = source_cfg.get("local", {}).get("base_dir", "data/raw/audioset")
        base_path = (
            Path(base_dir) if Path(base_dir).is_absolute() else self.data_root.parent / base_dir
        )
        samples: List[AudioSample] = []
        for idx, file_path in enumerate(self._iter_audio_files(base_path)):
            label = canonical_label(file_path.stem.split("_")[-1])
            samples.append(
                AudioSample(
                    path=file_path,
                    label=label,
                    dataset=self.NAME,
                    index=idx,
                    metadata={"subset": file_path.parent.name},
                )
            )
        return samples


class VGGSoundPreparer(BaseDatasetPreparer):
    NAME = "vggsound"

    def collect_samples(self) -> List[AudioSample]:
        source_cfg = self.config.get("source", {})
        base_dir = source_cfg.get("local", {}).get("base_dir", "data/raw/vggsound")
        base_path = (
            Path(base_dir) if Path(base_dir).is_absolute() else self.data_root.parent / base_dir
        )
        mapping_file = self.config.get("integration", {}).get("mapping_file")
        label_mapper = load_mapping(mapping_file) if mapping_file else {}
        samples: List[AudioSample] = []
        for idx, file_path in enumerate(self._iter_audio_files(base_path)):
            raw_label = canonical_label(file_path.stem.split("_")[-1])
            mapped = label_mapper.get(raw_label, raw_label)
            samples.append(
                AudioSample(
                    path=file_path,
                    label=mapped,
                    dataset=self.NAME,
                    index=idx,
                    metadata={"relative_path": str(file_path.relative_to(base_path))},
                )
            )
        return samples


PREPARER_REGISTRY = {
    "mad": MADPreparer,
    "fsd50k": FSD50KPreparer,
    "audioset": AudioSetPreparer,
    "vggsound": VGGSoundPreparer,
}

DEFAULT_CONFIG_MAP = {
    "mad": Path("configs/data/mad_dataset.yaml"),
    "fsd50k": Path("configs/data/fsd50k.yaml"),
    "audioset": Path("configs/data/audioset.yaml"),
    "vggsound": Path("configs/data/vggsound.yaml"),
}


def load_mapping(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    mapping_path = Path(path)
    if not mapping_path.is_absolute():
        mapping_path = Path.cwd() / mapping_path
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    result = {}
    for label, cfg in data.get("serenesense_labels", {}).items():
        for key in ("synonyms", "include"):
            for synonym in cfg.get(key, []) or []:
                result[canonical_label(synonym)] = canonical_label(label)
    return result


def map_label(labels: Iterable[str], mapping: Dict[str, str]) -> str:
    for label in labels:
        canonical = canonical_label(label)
        if canonical in mapping:
            return mapping[canonical]
    return canonical_label(next(iter(labels)))


def load_fsd50k_metadata(csv_path: Path) -> Dict[str, List[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"FSD50K metadata file not found: {csv_path}")
    metadata: Dict[str, List[str]] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filename = row.get("fname")
            labels = row.get("labels", "")
            if not filename:
                continue
            label_list = [label.strip() for label in labels.split(",") if label.strip()]
            metadata[filename] = label_list or ["unknown"]
    return metadata


def stratified_split(
    samples: List[AudioSample],
    ratios: Dict[str, float],
    random_state: int = 42,
) -> Dict[str, List[AudioSample]]:
    labels = [sample.label for sample in samples]
    train_ratio = ratios.get("train", 0.7)
    val_ratio = ratios.get("validation", ratios.get("val", 0.15))
    test_ratio = ratios.get("test", 0.15)
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-3):
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total

    # Check if stratification is possible - each class needs at least 2 samples
    from collections import Counter
    label_counts = Counter(labels)
    min_samples_per_class = min(label_counts.values())

    # If any class has fewer than 2 samples, filter them out with a warning
    if min_samples_per_class < 2:
        import logging
        logger = logging.getLogger(__name__)
        rare_classes = [label for label, count in label_counts.items() if count < 2]
        logger.warning(f"Filtering out {len(rare_classes)} classes with < 2 samples: {rare_classes}")
        logger.warning(f"Total samples before filtering: {len(samples)}")

        # Filter samples
        filtered_samples = [s for s in samples if s.label not in rare_classes]
        logger.warning(f"Total samples after filtering: {len(filtered_samples)}")

        if len(filtered_samples) < 10:
            raise ValueError(f"Too few samples remaining after filtering rare classes: {len(filtered_samples)}")

        samples = filtered_samples
        labels = [sample.label for sample in samples]
        label_counts = Counter(labels)
        min_samples_per_class = min(label_counts.values())

    # Use stratified split
    train_samples, temp_samples, train_labels, temp_labels = train_test_split(
        samples,
        labels,
        test_size=1 - train_ratio,
        stratify=labels,
        random_state=random_state,
    )

    # Check temp_samples for classes with < 2 samples before second split
    temp_label_counts = Counter(temp_labels)
    temp_min_samples = min(temp_label_counts.values()) if temp_label_counts else 0

    if temp_min_samples < 2:
        import logging
        logger = logging.getLogger(__name__)
        temp_rare_classes = [label for label, count in temp_label_counts.items() if count < 2]
        logger.warning(f"After first split, {len(temp_rare_classes)} classes have < 2 samples in temp set")
        logger.warning(f"Filtering these from temp set: {temp_rare_classes}")
        logger.warning(f"Temp samples before filtering: {len(temp_samples)}")

        # Filter temp samples
        filtered_temp = [s for s in temp_samples if s.label not in temp_rare_classes]
        logger.warning(f"Temp samples after filtering: {len(filtered_temp)}")

        if len(filtered_temp) < 10:
            raise ValueError(f"Too few temp samples remaining after filtering: {len(filtered_temp)}")

        temp_samples = filtered_temp
        temp_labels = [sample.label for sample in temp_samples]

    val_share = val_ratio / (val_ratio + test_ratio)
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=1 - val_share,
        stratify=temp_labels,
        random_state=random_state,
    )
    return {"train": train_samples, "validation": val_samples, "test": test_samples}


def _process_sample_worker(sample_and_config: Tuple[AudioSample, ProcessingConfig]) -> Tuple[AudioSample, Optional[List[ProcessedSample]], Optional[str]]:
    """Worker function for multiprocessing that processes a single audio sample.

    Args:
        sample_and_config: Tuple of (AudioSample, ProcessingConfig)

    Returns:
        Tuple of (original_sample, processed_results or None, error_message or None)
    """
    sample, config = sample_and_config
    try:
        results = process_audio_sample(sample, config)
        return (sample, results, None)
    except Exception as exc:
        return (sample, None, str(exc))


def process_split(
    split_name: str,
    samples: List[AudioSample],
    config: ProcessingConfig,
    output_root: Path,
    logger: logging.Logger,
    max_workers: int = 4,
    label_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not samples:
        logger.warning("No samples provided for split '%s'", split_name)
        return {"manifest": [], "statistics": {}}

    hdf5_path = output_root / f"{split_name}.h5"
    manifest_path = output_root / f"manifest_{split_name}.json"
    zarr_path = output_root / f"{split_name}.zarr"

    writer = HDF5SplitWriter(hdf5_path, config.target_num_samples, config.chunk_size)
    zarr_writer = None
    if config.create_zarr:
        try:
            zarr_writer = ZarrSplitWriter(zarr_path, config.target_num_samples, config.chunk_size)
        except Exception as exc:
            logger.warning("Unable to create Zarr output: %s", exc)

    stats = StatisticsAccumulator()

    # Prepare arguments for worker pool
    worker_args = [(sample, config) for sample in samples]

    # Ensure max_workers is reasonable for the system
    cpu_count = os.cpu_count() or 2
    effective_max_workers = min(max_workers, max(1, cpu_count - 1))
    logger.info("Using %d worker processes for %s split (CPU count: %d)", effective_max_workers, split_name, cpu_count)

    progress = tqdm(total=len(samples), desc=f"Processing {split_name}", unit="sample", leave=False)
    with concurrent.futures.ProcessPoolExecutor(max_workers=effective_max_workers) as executor:
        futures = {executor.submit(_process_sample_worker, args): args[0] for args in worker_args}
        for future in concurrent.futures.as_completed(futures):
            original_sample = futures[future]
            try:
                original_sample, results, error_msg = future.result()
                if error_msg:
                    logger.error("Processing failed for %s: %s", original_sample.path, error_msg)
                    progress.update(1)
                    continue
                if results:
                    for processed in results:
                        writer.append(processed)
                        if zarr_writer:
                            zarr_writer.append(processed)
                        stats.update(processed)
            except Exception as exc:
                logger.error("Unexpected error processing %s: %s", original_sample.path, exc)
            progress.update(1)
    progress.close()

    manifest = writer.close()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    if label_metadata:
        split_label_counts = Counter(sample.label for sample in samples)
        metadata_payload = {
            "split": split_name,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "num_samples": len(samples),
            "label_counts": {label: count for label, count in split_label_counts.items()},
            "class_names": label_metadata.get("class_names", {}),
            "num_classes": label_metadata.get(
                "num_classes", len(label_metadata.get("class_names", {}))
            ),
        }
        metadata_path = output_root / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as meta_handle:
            json.dump(metadata_payload, meta_handle, indent=2)

    return {"manifest": manifest, "statistics": stats.to_dict()}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SereneSense data preparation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="+", help="Datasets to prepare (default: all supported)"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/data"),
        help="Directory containing dataset configs",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"), help="Raw data root directory"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed"),
        help="Destination root for processed data",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path("logs"), help="Directory for preparation logs"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Target sample rate (Hz)"
    )
    parser.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION, help="Target duration in seconds"
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=DEFAULT_CHANNELS,
        choices=(1, 2),
        help="Number of channels after processing",
    )
    parser.add_argument(
        "--no-normalize", action="store_true", help="Disable waveform normalization"
    )
    parser.add_argument("--no-silence-trim", action="store_true", help="Disable silence trimming")
    parser.add_argument(
        "--silence-db", type=float, default=30.0, help="Silence threshold in dB for trimming"
    )
    parser.add_argument(
        "--augmentations",
        nargs="*",
        default=[],
        help="Augmentations to apply (gaussian_noise, time_shift, pitch_shift, time_stretch)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=64, help="Chunk size for HDF5/Zarr datasets"
    )
    parser.add_argument(
        "--create-zarr", action="store_true", help="Emit Zarr archives alongside HDF5"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--subsets", nargs="*", help="Dataset-specific subset selection (e.g., dev, eval)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel worker processes (default: 4, recommended for Windows)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without processing")
    parser.add_argument("--verbose", action="store_true", help="Increase logging verbosity")
    return parser.parse_args(argv)


def resolve_datasets(names: Optional[List[str]]) -> List[str]:
    if not names:
        return sorted(PREPARER_REGISTRY.keys())
    resolved = []
    aliases = {
        "mad": "mad",
        "fsd50k": "fsd50k",
        "fsd": "fsd50k",
        "audioset": "audioset",
        "audio_set": "audioset",
        "vggsound": "vggsound",
        "vgg_sound": "vggsound",
    }
    for name in names:
        key = name.strip().lower()
        if key not in aliases:
            raise ValueError(f"Unknown dataset: {name}")
        resolved.append(aliases[key])
    return resolved


def load_dataset_config(dataset: str, config_dir: Path) -> Dict[str, Any]:
    default_path = DEFAULT_CONFIG_MAP.get(dataset)
    if default_path is None:
        raise ValueError(f"No configuration mapping for dataset '{dataset}'")
    config_path = config_dir / default_path.name
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return load_yaml(config_path)


def prepare_dataset(
    dataset: str,
    args: argparse.Namespace,
    config: Dict[str, Any],
    processing_cfg: ProcessingConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    preparer_cls = PREPARER_REGISTRY[dataset]
    preparer = preparer_cls(config=config, data_root=args.data_root, args=args)
    samples = preparer.collect_samples()
    if not samples:
        logger.warning("No samples discovered for dataset '%s'", dataset)
        return {}

    logger.info("Collected %d samples for %s", len(samples), dataset)

    ratios = config.get("splits", {}).get(
        "ratios", {"train": 0.7, "validation": 0.15, "test": 0.15}
    )
    splits = stratified_split(samples, ratios, random_state=processing_cfg.seed)

    dataset_output_root = args.output_root / dataset
    dataset_output_root.mkdir(parents=True, exist_ok=True)

    dataset_stats: Dict[str, Any] = {}
    label_metadata = getattr(preparer, "label_metadata", None)
    for split_name, split_samples in splits.items():
        split_dir = dataset_output_root / split_name
        if args.dry_run:
            logger.info(
                "[dry-run] Would process %d samples for %s/%s",
                len(split_samples),
                dataset,
                split_name,
            )
            continue
        result = process_split(
            split_name,
            split_samples,
            processing_cfg,
            split_dir,
            logger,
            args.max_workers,
            label_metadata=label_metadata,
        )
        dataset_stats[split_name] = result.get("statistics", {})

    if args.dry_run:
        return {}

    stats_path = dataset_output_root / "statistics.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset_stats, handle, indent=2)

    logger.info("Completed processing for %s. Statistics stored at %s", dataset, stats_path)
    return dataset_stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = setup_logger(args.log_dir, verbose=args.verbose)

    processing_cfg = ProcessingConfig(
        sample_rate=args.sample_rate,
        channels=args.channels,
        duration=args.duration,
        normalize=not args.no_normalize,
        silence_trim=not args.no_silence_trim,
        silence_db=args.silence_db,
        augmentations=args.augmentations or [],
        chunk_size=args.chunk_size,
        seed=args.seed,
        create_zarr=args.create_zarr,
    )

    datasets = resolve_datasets(args.datasets)
    config_dir = args.config_dir if args.config_dir.is_absolute() else Path.cwd() / args.config_dir
    all_stats: Dict[str, Any] = {}

    for dataset in datasets:
        logger.info("=== Preparing %s ===", dataset)
        config = load_dataset_config(dataset, config_dir)
        stats = prepare_dataset(dataset, args, config, processing_cfg, logger)
        if stats:
            all_stats[dataset] = stats

    if not args.dry_run and all_stats:
        summary_path = args.output_root / "preparation_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(all_stats, handle, indent=2)
        logger.info("Preparation summary written to %s", summary_path)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
