#!/usr/bin/env python3
"""
Comprehensive dataset download manager for SereneSense.

Supports resilient downloads with resume capability, checksum verification,
structured logging, and dataset-specific workflows for MAD, AudioSet, FSD50K,
and VGGSound. Integrates with configuration files under ``configs/data`` and
produces logs under ``logs/``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
import yaml
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


LOGGER_NAME = "serenesense.downloads"
CANONICAL_LABELS = {
    0: "helicopter",
    1: "fighter_aircraft",
    2: "military_vehicle",
    3: "truck",
    4: "footsteps",
    5: "speech",
    6: "background",
}


class DownloadError(RuntimeError):
    """Raised when a download fails."""


class ChecksumError(DownloadError):
    """Raised when checksum verification fails."""


class ExtractionError(RuntimeError):
    """Raised when archive extraction fails."""


class DatasetConfigError(RuntimeError):
    """Raised when the dataset configuration is invalid or missing."""


@dataclass
class DownloadTask:
    """Metadata describing a download operation."""

    url: str
    destination: Path
    checksum: Optional[str] = None
    size_bytes: Optional[int] = None
    description: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    resume: bool = True


def setup_logging(log_dir: Path, verbose: bool = False) -> logging.Logger:
    """Configure application logging with console and file handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"download_{timestamp}.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_level = logging.INFO if not verbose else logging.DEBUG
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_format = logging.Formatter(
        fmt="%(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    logger.debug("Logging initialized. Log file: %s", log_path)
    return logger


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""
    if not path.exists():
        raise DatasetConfigError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_checksum(value: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Split a checksum string into algorithm and hash."""
    if not value:
        return None, None
    if ":" not in value:
        raise DatasetConfigError(f"Checksum must use '<algo>:<hash>' format: {value}")
    algo, digest = value.split(":", 1)
    return algo.lower(), digest.strip()


class DownloadManager:
    """Coordinates resumable downloads, checksum validation, and extraction."""

    def __init__(
        self,
        data_root: Path,
        cache_root: Path,
        log_dir: Path,
        resume: bool = True,
        skip_checksum: bool = False,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.data_root = self._resolve_path(data_root)
        self.cache_root = self._resolve_path(cache_root)
        self.log_dir = self._resolve_path(log_dir)
        self.resume = resume
        self.skip_checksum = skip_checksum
        self.dry_run = dry_run
        self.logger = setup_logging(self.log_dir, verbose=verbose)
        self.session = self._create_session()
        self._register_signal_handlers()

    def _resolve_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return self.project_root / path

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD"}),
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _register_signal_handlers(self) -> None:
        def handler(signum, _frame):
            self.logger.warning("Received signal %s, attempting graceful shutdown.", signum)
            self.close()
            sys.exit(1)

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def resolve(self, path_like: str | Path) -> Path:
        """Resolve a path relative to the project root."""
        path = Path(path_like)
        if path.is_absolute():
            return path
        return self.project_root / path

    def resolve_data_path(self, path_like: str | Path) -> Path:
        """Resolve a path under the configured data root."""
        path = Path(path_like)
        if path.is_absolute():
            return path
        if str(path).startswith("data/"):
            return self.project_root / path
        return self.data_root / path

    def resolve_cache_path(self, dataset: str, filename: str) -> Path:
        dataset_cache = self.cache_root / dataset
        dataset_cache.mkdir(parents=True, exist_ok=True)
        return dataset_cache / filename

    def download(self, task: DownloadTask) -> Path:
        """Download a file with resume support and checksum verification."""
        destination = task.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = destination.with_suffix(destination.suffix + ".part")

        if destination.exists():
            if self.skip_checksum or not task.checksum:
                self.logger.info("File already present, skipping: %s", destination)
                return destination
            algo, digest = parse_checksum(task.checksum)
            if algo and digest and self.verify_checksum(destination, algo, digest):
                self.logger.info("File already present and verified: %s", destination)
                return destination
            self.logger.warning("Existing file failed checksum; re-downloading: %s", destination)
            destination.unlink()

        if tmp_path.exists() and not self.resume:
            self.logger.warning("Removing leftover partial file: %s", tmp_path)
            tmp_path.unlink()

        if self.dry_run:
            self.logger.info("[dry-run] Would download %s -> %s", task.url, destination)
            return destination

        resume_pos = tmp_path.stat().st_size if tmp_path.exists() else 0
        headers = dict(task.headers)
        if resume_pos and self.resume:
            headers["Range"] = f"bytes={resume_pos}-"

        self.logger.info(
            "Downloading %s%s",
            task.description + " " if task.description else "",
            task.url,
        )

        try:
            with self.session.get(task.url, stream=True, headers=headers, timeout=60) as response:
                if response.status_code not in (200, 206):
                    raise DownloadError(
                        f"Failed to download {task.url} (status {response.status_code})"
                    )

                total = task.size_bytes
                if "Content-Length" in response.headers:
                    length = int(response.headers["Content-Length"])
                    total = length + resume_pos if response.status_code == 206 else length

                mode = "ab" if resume_pos and self.resume else "wb"
                with tmp_path.open(mode) as handle, tqdm(
                    total=total,
                    initial=resume_pos if mode == "ab" else 0,
                    unit="B",
                    unit_scale=True,
                    desc=task.description or destination.name,
                    leave=False,
                ) as progress:
                    for chunk in response.iter_content(chunk_size=2**20):
                        if chunk:
                            handle.write(chunk)
                            progress.update(len(chunk))
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise DownloadError(f"Failed to download {task.url}: {exc}") from exc

        tmp_path.rename(destination)

        if self.skip_checksum or not task.checksum:
            return destination

        algo, digest = parse_checksum(task.checksum)
        if not algo or not digest:
            return destination

        if not self.verify_checksum(destination, algo, digest):
            destination.unlink(missing_ok=True)
            raise ChecksumError(f"Checksum mismatch for {destination} ({algo})")

        self.logger.debug("Checksum verified for %s (%s)", destination, algo)
        return destination

    def verify_checksum(self, path: Path, algorithm: str, expected: str) -> bool:
        hasher = hashlib.new(algorithm)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(2**20), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        return digest.lower() == expected.lower()

    def extract(self, archive: Path, target_dir: Path, fmt: Optional[str] = None) -> None:
        if self.dry_run:
            self.logger.info("[dry-run] Would extract %s -> %s", archive, target_dir)
            return

        target_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Extracting %s -> %s", archive, target_dir)
        try:
            shutil.unpack_archive(str(archive), str(target_dir), format=fmt)
        except Exception as exc:
            raise ExtractionError(f"Failed to extract {archive}: {exc}") from exc

    def close(self) -> None:
        self.session.close()


class BaseDatasetDownloader:
    """Common base class for dataset downloaders."""

    NAME: str = "base"

    def __init__(self, manager: DownloadManager, config: Dict[str, Any], args: argparse.Namespace):
        self.manager = manager
        self.config = config
        self.args = args
        self.logger = logging.getLogger(f"{LOGGER_NAME}.{self.NAME}")

    @property
    def selected_classes(self) -> Optional[Sequence[str]]:
        if not self.args.classes:
            return None
        return [c.lower() for c in self.args.classes]

    @staticmethod
    def canonical_label(label: str) -> str:
        return label.strip().lower().replace(" ", "_")

    def download(self) -> None:
        raise NotImplementedError


class ArchiveDatasetDownloader(BaseDatasetDownloader):
    """Downloader for archive-based datasets."""

    DOWNLOAD_KEY = "download"
    EXTRACTION_KEY = "extraction"

    def download_archive(self, dataset: str) -> None:
        source = self.config.get("source", {})
        download_cfg = source.get(self.DOWNLOAD_KEY)
        extraction_cfg = source.get(self.EXTRACTION_KEY)
        if not download_cfg:
            raise DatasetConfigError(f"Missing {self.DOWNLOAD_KEY} configuration for {dataset}")

        url = download_cfg["url"]

        # Check if this is a Kaggle dataset
        if url.startswith("kagglehub:"):
            # Use newer kagglehub library (recommended)
            archive_path = self._download_from_kagglehub(url, dataset, download_cfg)
        elif url.startswith("kaggle:"):
            # Use older kaggle API library (legacy support)
            archive_path = self._download_from_kaggle(url, dataset, download_cfg)
        else:
            filename = download_cfg.get("filename") or Path(url).name
            checksum = download_cfg.get("checksum")
            size_mb = download_cfg.get("size_mb")
            size_bytes = int(size_mb * 1024 * 1024) if size_mb else None

            destination = self.manager.resolve_cache_path(dataset, filename)
            task = DownloadTask(
                url=url,
                destination=destination,
                checksum=checksum,
                size_bytes=size_bytes,
                description=f"{dataset.upper()} archive",
            )
            archive_path = self.manager.download(task)

        if extraction_cfg:
            target_dir = self.manager.resolve_data_path(
                extraction_cfg.get("target_dir", f"data/raw/{dataset}")
            )

            # Check if archive_path is a directory (already extracted by kagglehub)
            if archive_path.is_dir():
                self.logger.info(f"Dataset already extracted, copying from {archive_path} to {target_dir}")
                if not self.manager.dry_run:
                    import shutil
                    # Copy the entire directory tree
                    if target_dir.exists():
                        self.logger.info(f"Target directory exists, updating files...")
                    else:
                        target_dir.parent.mkdir(parents=True, exist_ok=True)

                    # Use copytree with dirs_exist_ok to handle existing directories
                    shutil.copytree(archive_path, target_dir, dirs_exist_ok=True)
                    self.logger.info(f"Copied dataset to {target_dir}")
                else:
                    self.logger.info(f"[dry-run] Would copy {archive_path} to {target_dir}")
            else:
                # It's a zip file, extract normally
                fmt = extraction_cfg.get("archive_format")
                self.manager.extract(archive_path, target_dir, fmt=fmt)
                if extraction_cfg.get("preserve_structure", True):
                    self.logger.debug("Preserved archive structure for %s", dataset)

    def _download_from_kaggle(self, kaggle_url: str, dataset: str, download_cfg: dict) -> Path:
        """Download dataset from Kaggle using kaggle API."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            raise RuntimeError(
                "Kaggle API not installed. Install with: pip install kaggle"
            )

        # Parse Kaggle URL: kaggle:owner/dataset-name
        kaggle_path = kaggle_url.replace("kaggle:", "").strip()

        self.logger.info(f"Downloading {dataset} from Kaggle: {kaggle_path}")

        if self.manager.dry_run:
            self.logger.info(f"[dry-run] Would download from Kaggle: {kaggle_path}")
            # Return a dummy path for dry run
            return self.manager.resolve_cache_path(dataset, f"{dataset}.zip")

        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()

        # Download to cache directory
        cache_dir = self.manager.cache_root / dataset
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Downloading to: {cache_dir}")

        # Download the dataset
        api.dataset_download_files(
            kaggle_path,
            path=str(cache_dir),
            unzip=False
        )

        # Find the downloaded zip file
        zip_files = list(cache_dir.glob("*.zip"))
        if not zip_files:
            raise DownloadError(f"No zip file found after Kaggle download in {cache_dir}")

        archive_path = zip_files[0]
        self.logger.info(f"Downloaded: {archive_path}")

        return archive_path

    def _download_from_kagglehub(self, kaggle_url: str, dataset: str, download_cfg: dict) -> Path:
        """Download dataset from Kaggle using kagglehub library (newer, simpler API)."""
        try:
            import kagglehub
        except ImportError:
            raise RuntimeError(
                "kagglehub not installed. Install with: pip install kagglehub"
            )

        # Parse Kaggle URL: kagglehub:username/dataset-name
        kaggle_handle = kaggle_url.replace("kagglehub:", "").strip()

        self.logger.info(f"Downloading {dataset} from Kaggle Hub: {kaggle_handle}")

        if self.manager.dry_run:
            self.logger.info(f"[dry-run] Would download from Kaggle Hub: {kaggle_handle}")
            # Return a dummy path for dry run
            return self.manager.resolve_cache_path(dataset, f"{dataset}.zip")

        # kagglehub downloads to its own cache (~/.cache/kagglehub/)
        # and returns the path to the downloaded directory
        try:
            self.logger.info(f"Downloading via kagglehub (may use cache)...")
            downloaded_path = kagglehub.dataset_download(kaggle_handle)
            self.logger.info(f"Downloaded to: {downloaded_path}")

            # kagglehub returns a directory path, we need to find the zip file
            # or handle the extracted files
            downloaded_path = Path(downloaded_path)

            # Check if it's a directory with a zip file
            if downloaded_path.is_dir():
                zip_files = list(downloaded_path.glob("*.zip"))
                if zip_files:
                    # Copy the zip to our project cache
                    cache_dir = self.manager.cache_root / dataset
                    cache_dir.mkdir(parents=True, exist_ok=True)

                    dest_zip = cache_dir / zip_files[0].name
                    if not dest_zip.exists():
                        self.logger.info(f"Copying {zip_files[0]} to {dest_zip}")
                        import shutil
                        shutil.copy2(zip_files[0], dest_zip)
                    else:
                        self.logger.info(f"Zip already exists in cache: {dest_zip}")

                    return dest_zip
                else:
                    # No zip file found - kagglehub has already extracted the dataset
                    # Copy the extracted files directly to the target directory
                    self.logger.info(f"Dataset already extracted by kagglehub at {downloaded_path}")
                    self.logger.info(f"Will copy directly to target directory (skipping extraction step)")

                    # Return None to signal that extraction should be skipped
                    # The extraction_cfg check will handle copying the files
                    return downloaded_path
            elif downloaded_path.is_file() and downloaded_path.suffix == ".zip":
                # It's already a zip file, copy to our cache
                cache_dir = self.manager.cache_root / dataset
                cache_dir.mkdir(parents=True, exist_ok=True)

                dest_zip = cache_dir / downloaded_path.name
                if not dest_zip.exists():
                    self.logger.info(f"Copying {downloaded_path} to {dest_zip}")
                    import shutil
                    shutil.copy2(downloaded_path, dest_zip)
                else:
                    self.logger.info(f"Zip already exists in cache: {dest_zip}")

                return dest_zip
            else:
                # Unexpected format
                self.logger.warning(f"Unexpected download format: {downloaded_path}")
                return downloaded_path

        except Exception as exc:
            raise DownloadError(f"kagglehub download failed for {kaggle_handle}: {exc}") from exc


class MADDownloader(ArchiveDatasetDownloader):
    NAME = "mad"

    def download(self) -> None:
        self.logger.info("Preparing to download MAD dataset")
        self.download_archive(self.NAME)
        self._verify_structure()

    def _verify_structure(self) -> None:
        verification_cfg = self.config.get("source", {}).get("verification", {})
        if not verification_cfg:
            return
        if self.args.dry_run:
            self.logger.info("[dry-run] Would verify MAD dataset structure.")
            return
        if not verification_cfg.get("verify_structure", False):
            return
        extraction_cfg = self.config["source"]["extraction"]
        target_dir = self.manager.resolve_data_path(
            extraction_cfg.get("target_dir", "data/raw/mad")
        )
        expected_items = verification_cfg.get("expected_items", [])
        missing = [item for item in expected_items if not (target_dir / item).exists()]
        if missing:
            self.logger.warning("MAD verification found missing items: %s", ", ".join(missing))
        else:
            self.logger.info("MAD dataset structure verified at %s", target_dir)


class FSD50KDownloader(ArchiveDatasetDownloader):
    NAME = "fsd50k"

    def download(self) -> None:
        self.logger.info("Preparing to download FSD50K dataset")
        self.download_archive(self.NAME)
        self._verify_contents()

    def _verify_contents(self) -> None:
        verification_cfg = self.config.get("source", {}).get("verification", {})
        if self.args.dry_run or not verification_cfg.get("verify_structure", False):
            return
        extraction_cfg = self.config["source"]["extraction"]
        target_dir = self.manager.resolve_data_path(
            extraction_cfg.get("target_dir", "data/raw/fsd50k")
        )
        expected_dirs = verification_cfg.get("structure", {}).get("expected_dirs", [])
        required_files = verification_cfg.get("structure", {}).get("required_files", [])
        missing_dirs = [d for d in expected_dirs if not (target_dir / d).exists()]
        missing_files = [f for f in required_files if not (target_dir / f).exists()]
        if missing_dirs or missing_files:
            if missing_dirs:
                self.logger.warning("Missing directories: %s", ", ".join(missing_dirs))
            if missing_files:
                self.logger.warning("Missing files: %s", ", ".join(missing_files))
        else:
            self.logger.info("FSD50K dataset verified at %s", target_dir)


class YouTubeDatasetDownloader(BaseDatasetDownloader):
    """Base class for datasets requiring YouTube downloads via yt-dlp."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.yt_dlp = self._resolve_executable(self.args.yt_dlp_path)

    @staticmethod
    def _resolve_executable(explicit: Optional[str]) -> str:
        if explicit:
            path = shutil.which(explicit)
            if path:
                return path
            raise RuntimeError(f"yt-dlp executable '{explicit}' not found in PATH")
        for candidate in ("yt-dlp", "youtube-dl"):
            path = shutil.which(candidate)
            if path:
                return path
        raise RuntimeError(
            "yt-dlp (or youtube-dl) is required but not installed. "
            "Install with `pip install yt-dlp`."
        )

    def _download_clip(
        self,
        entry: Dict[str, Any],
        output_dir: Path,
        audio_format: str,
        sample_rate: int,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ytid = entry["youtube_id"]
        start = entry["start"]
        end = entry["end"]
        label = entry.get("label", "clip")
        stem = f"{ytid}_{start:.3f}-{end:.3f}_{label}"
        output_path = output_dir / f"{stem}.{audio_format}"
        if output_path.exists() and not self.args.force_redownload:
            return

        if self.args.dry_run:
            self.logger.info("[dry-run] Would download clip %s", stem)
            return

        url = f"https://www.youtube.com/watch?v={ytid}"
        cmd = [
            self.yt_dlp,
            url,
            "--quiet",
            "--no-warnings",
            "--prefer-ffmpeg",
            "--extract-audio",
            "--audio-format",
            audio_format,
            "--output",
            str(output_path.with_suffix(".%(ext)s")),
            "--download-sections",
            f"*{start}-{end}",
            "--postprocessor-args",
            f"-ar {sample_rate}",
        ]
        if self.args.proxy:
            cmd.extend(["--proxy", self.args.proxy])
        if self.args.yt_dlp_args:
            cmd.extend(self.args.yt_dlp_args)

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            raise DownloadError(f"yt-dlp failed for {url}: {exc}") from exc

    def _download_clips_parallel(
        self,
        entries: List[Dict[str, Any]],
        output_dir: Path,
        audio_format: str,
        sample_rate: int,
    ) -> None:
        if not entries:
            self.logger.info("No clips to download for %s", self.NAME)
            return

        max_workers = min(self.args.max_workers, len(entries))
        progress = tqdm(
            total=len(entries),
            desc=f"{self.NAME} clips",
            unit="clip",
            leave=False,
        )

        def worker(entry: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Exception]]:
            try:
                self._download_clip(entry, output_dir, audio_format, sample_rate)
                return entry, None
            except Exception as exc:
                return entry, exc

        failures: List[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(worker, entry): entry for entry in entries}
            for future in concurrent.futures.as_completed(future_map):
                entry, err = future.result()
                if err:
                    clip_id = entry["youtube_id"]
                    failures.append(f"{clip_id} ({err})")
                progress.update(1)
        progress.close()

        if failures:
            self.logger.warning("Failed downloads (%d): %s", len(failures), "; ".join(failures))
        else:
            self.logger.info("All clips downloaded successfully for %s", self.NAME)


class AudioSetDownloader(YouTubeDatasetDownloader):
    NAME = "audioset"

    def download(self) -> None:
        self.logger.info("Preparing to download AudioSet")
        metadata_dir = self._metadata_directory()
        subset_records = self._download_metadata(metadata_dir)
        if self.args.metadata_only:
            self.logger.info("Metadata-only download requested; skipping audio clips.")
            return
        filtered_records = self._filter_records(subset_records)
        limited_records = (
            filtered_records[: self.args.max_clips] if self.args.max_clips else filtered_records
        )
        output_dir = self.manager.resolve_data_path(
            self.config.get("source", {}).get("local", {}).get("base_dir", "data/raw/audioset")
        )
        audio_format = self.config.get("source", {}).get("youtube", {}).get("format", "wav")
        sample_rate = self.config.get("source", {}).get("youtube", {}).get("sample_rate", 16000)
        self._download_clips_parallel(limited_records, output_dir, audio_format, sample_rate)

    def _metadata_directory(self) -> Path:
        meta_dir = (
            self.config.get("source", {})
            .get("local", {})
            .get("metadata_dir", "data/metadata/audioset")
        )
        return self.manager.resolve_data_path(meta_dir)

    def _download_metadata(self, metadata_dir: Path) -> List[Dict[str, Any]]:
        metadata_dir.mkdir(parents=True, exist_ok=True)
        subsets_cfg = self.config.get("subsets", {})
        requested = self.args.subsets or [
            name for name, cfg in subsets_cfg.items() if cfg.get("enabled", True)
        ]
        records: List[Dict[str, Any]] = []
        for subset in requested:
            cfg = subsets_cfg.get(subset)
            if not cfg:
                self.logger.warning("Unknown AudioSet subset requested: %s", subset)
                continue
            csv_url = cfg.get("csv_url")
            filename = Path(csv_url).name if csv_url else f"{subset}.csv"
            destination = metadata_dir / filename
            if csv_url:
                task = DownloadTask(
                    url=csv_url,
                    destination=destination,
                    description=f"AudioSet {subset} metadata",
                )
                self.manager.download(task)
            subset_records = self._parse_audioset_csv(destination, subset)
            records.extend(subset_records)
            self.logger.info("Loaded %d records for subset '%s'", len(subset_records), subset)
        return records

    def _parse_audioset_csv(self, path: Path, subset: str) -> List[Dict[str, Any]]:
        if not path.exists():
            raise DownloadError(f"AudioSet metadata file missing: {path}")
        parsed: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)  # header
            for row in reader:
                if len(row) < 4:
                    continue
                youtube_id = row[0]
                start = float(row[1])
                end = float(row[2])
                labels = row[3].strip('"').split(",")
                parsed.append(
                    {
                        "subset": subset,
                        "youtube_id": youtube_id,
                        "start": start,
                        "end": end,
                        "labels": labels,
                        "label": self._canonical_label(labels),
                    }
                )
        return parsed

    def _canonical_label(self, labels: Sequence[str]) -> str:
        mapping = self.config.get("classes", {}).get("mapping", {}).get("target_classes", {})
        label_lookup: Dict[str, str] = {}
        for idx_str, cfg in mapping.items():
            canonical_label = CANONICAL_LABELS.get(int(idx_str), f"class_{idx_str}")
            for source_id in cfg.get("source_ids", []):
                label_lookup[source_id] = canonical_label
        for label in labels:
            label = label.strip()
            if label in label_lookup:
                return label_lookup[label]
        return "unlabeled"

    def _filter_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        requested = self.selected_classes
        if not requested:
            return records
        filtered = [record for record in records if record["label"] in requested]
        self.logger.info(
            "Filtered AudioSet records: %d -> %d using classes %s",
            len(records),
            len(filtered),
            ", ".join(requested),
        )
        return filtered


class VGGSoundDownloader(YouTubeDatasetDownloader):
    NAME = "vggsound"

    def download(self) -> None:
        self.logger.info("Preparing to download VGGSound")
        metadata_dir = self._metadata_directory()
        metadata_path = metadata_dir / "files.csv"
        source_cfg = self.config.get("source", {})
        metadata_cfg = source_cfg.get("metadata", {})

        task = DownloadTask(
            url=metadata_cfg.get("csv_url"),
            destination=metadata_path,
            checksum=metadata_cfg.get("checksum"),
            size_bytes=(
                int(metadata_cfg.get("size_mb", 0) * 1024 * 1024)
                if metadata_cfg.get("size_mb")
                else None
            ),
            description="VGGSound metadata",
        )
        self.manager.download(task)
        records = self._parse_metadata(metadata_path)
        filtered = self._filter_records(records)
        limited = filtered[: self.args.max_clips] if self.args.max_clips else filtered

        if self.args.metadata_only:
            self.logger.info("Metadata-only download requested; skipping audio clips.")
            return

        download_cfg = source_cfg.get("download", {})
        output_dir = self.manager.resolve_data_path(
            source_cfg.get("local", {}).get("base_dir", "data/raw/vggsound")
        )
        audio_format = download_cfg.get("audio_format", "wav")
        sample_rate = download_cfg.get("sample_rate", 16000)
        self._download_clips_parallel(limited, output_dir, audio_format, sample_rate)

    def _metadata_directory(self) -> Path:
        meta_dir = (
            self.config.get("source", {})
            .get("local", {})
            .get("metadata_dir", "data/metadata/vggsound")
        )
        path = self.manager.resolve_data_path(meta_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _parse_metadata(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise DownloadError(f"VGGSound metadata file missing: {path}")
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label = row.get("label", "unknown")
                start = float(row.get("start_seconds", 0))
                end = float(row.get("end_seconds", start + 10))
                records.append(
                    {
                        "youtube_id": row.get("youtube_id") or row.get("video_id"),
                        "start": start,
                        "end": end,
                        "label": self.canonical_label(label),
                        "labels": [label],
                    }
                )
        self.logger.info("Parsed %d VGGSound records", len(records))
        return records

    def _filter_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        requested = self.selected_classes
        if not requested:
            return records
        filtered = [record for record in records if record["label"] in requested]
        self.logger.info(
            "Filtered VGGSound records: %d -> %d using classes %s",
            len(records),
            len(filtered),
            ", ".join(requested),
        )
        return filtered


DATASET_REGISTRY = {
    "mad": MADDownloader,
    "audioset": AudioSetDownloader,
    "fsd50k": FSD50KDownloader,
    "vggsound": VGGSoundDownloader,
}

DEFAULT_CONFIG_MAP = {
    "mad": Path("configs/data/mad_dataset.yaml"),
    "audioset": Path("configs/data/audioset.yaml"),
    "fsd50k": Path("configs/data/fsd50k.yaml"),
    "vggsound": Path("configs/data/vggsound.yaml"),
}


def normalize_dataset_name(name: str) -> str:
    canonical = name.strip().lower()
    aliases = {
        "mad": "mad",
        "military": "mad",
        "audioset": "audioset",
        "audio_set": "audioset",
        "fsd50k": "fsd50k",
        "fsd": "fsd50k",
        "vggsound": "vggsound",
        "vgg_sound": "vggsound",
    }
    if canonical not in aliases:
        raise DatasetConfigError(f"Unknown dataset: {name}")
    return aliases[canonical]


def discover_available_datasets() -> List[str]:
    return sorted(DATASET_REGISTRY.keys())


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SereneSense dataset downloader",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Datasets to download (default: all available)",
    )
    parser.add_argument(
        "--classes", nargs="*", help="Canonical classes to download (e.g., helicopter truck)"
    )
    parser.add_argument("--subsets", nargs="*", help="Subsets to download (dataset-specific)")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/data"),
        help="Directory containing dataset configs",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"), help="Root directory for dataset storage"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/downloads"),
        help="Directory for cached archives",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path("logs"), help="Directory for log files"
    )
    parser.add_argument(
        "--yt-dlp-path", type=str, help="Custom path or alias for yt-dlp executable"
    )
    parser.add_argument("--proxy", type=str, help="Proxy URL for yt-dlp downloads")
    parser.add_argument(
        "--yt-dlp-args", nargs=argparse.REMAINDER, help="Additional arguments forwarded to yt-dlp"
    )
    parser.add_argument("--max-clips", type=int, help="Limit number of clips per dataset")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum concurrent downloads for clip-based datasets",
    )
    parser.add_argument(
        "--metadata-only", action="store_true", help="Only fetch metadata (skip media downloads)"
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force re-download even if clips already exist",
    )
    parser.add_argument(
        "--resume/--no-resume", dest="resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--skip-checksum", action="store_true", help="Skip checksum verification")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate configuration without downloading"
    )
    parser.add_argument("--verbose", action="store_true", help="Increase console logging verbosity")
    parser.add_argument("--version", action="version", version="SereneSense Downloader 1.0.0")
    return parser.parse_args(argv)


def load_dataset_config(name: str, args: argparse.Namespace) -> Dict[str, Any]:
    config_dir = args.config_dir
    config_dir = config_dir if config_dir.is_absolute() else Path.cwd() / config_dir
    default_path = DEFAULT_CONFIG_MAP.get(name)
    config_path = config_dir / default_path.name if default_path else None
    if not config_path or not config_path.exists():
        raise DatasetConfigError(f"Configuration for dataset '{name}' not found in {config_dir}")
    config = load_yaml(config_path)
    if not config:
        raise DatasetConfigError(f"Configuration file is empty: {config_path}")
    return config


def instantiate_downloader(
    dataset: str,
    manager: DownloadManager,
    args: argparse.Namespace,
) -> BaseDatasetDownloader:
    config = load_dataset_config(dataset, args)
    downloader_cls = DATASET_REGISTRY[dataset]
    return downloader_cls(manager=manager, config=config, args=args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    requested = (
        discover_available_datasets()
        if not args.datasets
        else [normalize_dataset_name(d) for d in args.datasets]
    )
    manager = DownloadManager(
        data_root=args.data_root,
        cache_root=args.cache_dir,
        log_dir=args.log_dir,
        resume=args.resume,
        skip_checksum=args.skip_checksum,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    manager.logger.info("Datasets requested: %s", ", ".join(requested))
    exit_code = 0
    for dataset in requested:
        try:
            downloader = instantiate_downloader(dataset, manager, args)
            downloader.download()
        except Exception as exc:
            manager.logger.exception("Failed to download %s: %s", dataset, exc)
            exit_code = 1

    manager.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
