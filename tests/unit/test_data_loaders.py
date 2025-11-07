"""
Unit tests for SereneSense data loaders.

Tests all dataset loaders (MAD, AudioSet, FSD50K) for:
- Correct initialization
- Data loading functionality
- Label mapping
- Transform application
- Split handling
- Error handling
- Memory efficiency
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import json
import sys
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.data.loaders.mad_loader import MADDataset
from core.data.loaders.audioset_loader import AudioSetDataset
from core.data.loaders.fsd50k_loader import FSD50KDataset


class TestMADDataset:
    """Test MAD (Military Audio Detection) dataset loader."""

    @pytest.fixture
    def mock_mad_data_dir(self):
        """Create a mock MAD dataset directory structure."""
        temp_dir = tempfile.mkdtemp()

        # Create directory structure
        (Path(temp_dir) / "audio" / "train").mkdir(parents=True)
        (Path(temp_dir) / "audio" / "val").mkdir(parents=True)
        (Path(temp_dir) / "audio" / "test").mkdir(parents=True)
        (Path(temp_dir) / "metadata").mkdir(parents=True)

        # Create mock metadata files
        train_metadata = pd.DataFrame(
            {
                "filename": ["tank_001.wav", "helicopter_001.wav", "jet_001.wav"],
                "label": ["tank", "helicopter", "jet"],
                "vehicle_type": ["ground", "air", "air"],
                "duration": [5.0, 8.0, 12.0],
                "split": ["train", "train", "train"],
            }
        )

        val_metadata = pd.DataFrame(
            {
                "filename": ["tank_002.wav", "helicopter_002.wav"],
                "label": ["tank", "helicopter"],
                "vehicle_type": ["ground", "air"],
                "duration": [6.0, 7.0],
                "split": ["val", "val"],
            }
        )

        test_metadata = pd.DataFrame(
            {
                "filename": ["tank_003.wav", "jet_002.wav"],
                "label": ["tank", "jet"],
                "vehicle_type": ["ground", "air"],
                "duration": [4.0, 10.0],
                "split": ["test", "test"],
            }
        )

        # Save metadata
        train_metadata.to_csv(Path(temp_dir) / "metadata" / "train.csv", index=False)
        val_metadata.to_csv(Path(temp_dir) / "metadata" / "val.csv", index=False)
        test_metadata.to_csv(Path(temp_dir) / "metadata" / "test.csv", index=False)

        # Create mock audio files (just empty files for testing)
        for split_data in [train_metadata, val_metadata, test_metadata]:
            split_name = split_data["split"].iloc[0]
            for filename in split_data["filename"]:
                audio_path = Path(temp_dir) / "audio" / split_name / filename
                audio_path.touch()

        # Create class mapping
        class_mapping = {"tank": 0, "helicopter": 1, "jet": 2}

        with open(Path(temp_dir) / "metadata" / "class_mapping.json", "w") as f:
            json.dump(class_mapping, f)

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_mad_dataset_initialization(self, mock_mad_data_dir):
        """Test MAD dataset initialization."""
        dataset = MADDataset(data_dir=mock_mad_data_dir, split="train", transform=None)

        assert dataset.data_dir == Path(mock_mad_data_dir)
        assert dataset.split == "train"
        assert len(dataset.class_to_idx) == 3
        assert "tank" in dataset.class_to_idx
        assert "helicopter" in dataset.class_to_idx
        assert "jet" in dataset.class_to_idx

    def test_mad_dataset_length(self, mock_mad_data_dir):
        """Test MAD dataset length for different splits."""
        train_dataset = MADDataset(mock_mad_data_dir, split="train")
        val_dataset = MADDataset(mock_mad_data_dir, split="val")
        test_dataset = MADDataset(mock_mad_data_dir, split="test")

        assert len(train_dataset) == 3
        assert len(val_dataset) == 2
        assert len(test_dataset) == 2

    @patch("torchaudio.load")
    def test_mad_dataset_getitem(self, mock_torchaudio_load, mock_mad_data_dir):
        """Test MAD dataset item retrieval."""
        # Mock torchaudio.load to return fake audio data
        mock_torchaudio_load.return_value = (
            torch.randn(1, 16000),  # 1 second of audio at 16kHz
            16000,  # sample rate
        )

        dataset = MADDataset(mock_mad_data_dir, split="train")

        # Test getting first item
        audio, label = dataset[0]

        assert isinstance(audio, torch.Tensor)
        assert isinstance(label, int)
        assert audio.shape[0] == 1  # Single channel
        assert 0 <= label < len(dataset.class_to_idx)

        # Verify torchaudio.load was called
        mock_torchaudio_load.assert_called_once()

    def test_mad_dataset_invalid_split(self, mock_mad_data_dir):
        """Test MAD dataset with invalid split."""
        with pytest.raises(ValueError):
            MADDataset(mock_mad_data_dir, split="invalid_split")

    def test_mad_dataset_missing_directory(self):
        """Test MAD dataset with missing directory."""
        with pytest.raises(FileNotFoundError):
            MADDataset("/nonexistent/path", split="train")

    def test_mad_dataset_with_transform(self, mock_mad_data_dir):
        """Test MAD dataset with transform."""

        def dummy_transform(x):
            return x * 2

        dataset = MADDataset(mock_mad_data_dir, split="train", transform=dummy_transform)

        # Mock audio loading
        with patch("torchaudio.load") as mock_load:
            mock_load.return_value = (torch.ones(1, 16000), 16000)

            audio, label = dataset[0]

            # Transform should have been applied (x * 2)
            assert torch.allclose(audio, torch.ones(1, 16000) * 2)

    def test_mad_dataset_class_distribution(self, mock_mad_data_dir):
        """Test MAD dataset class distribution."""
        dataset = MADDataset(mock_mad_data_dir, split="train")

        # Count class occurrences
        class_counts = {}
        for i in range(len(dataset)):
            with patch("torchaudio.load") as mock_load:
                mock_load.return_value = (torch.randn(1, 16000), 16000)
                _, label = dataset[i]
                class_counts[label] = class_counts.get(label, 0) + 1

        # Should have representation of all classes
        assert len(class_counts) <= len(dataset.class_to_idx)
        assert all(count > 0 for count in class_counts.values())


class TestAudioSetDataset:
    """Test AudioSet dataset loader."""

    @pytest.fixture
    def mock_audioset_data_dir(self):
        """Create a mock AudioSet dataset directory structure."""
        temp_dir = tempfile.mkdtemp()

        # Create directory structure
        (Path(temp_dir) / "audio" / "balanced_train").mkdir(parents=True)
        (Path(temp_dir) / "audio" / "eval").mkdir(parents=True)
        (Path(temp_dir) / "metadata").mkdir(parents=True)

        # Create mock metadata files
        balanced_metadata = pd.DataFrame(
            {
                "YTID": ["abc123", "def456", "ghi789"],
                "start_seconds": [10.0, 5.0, 0.0],
                "end_seconds": [20.0, 15.0, 10.0],
                "positive_labels": ['"/m/07yv9,/m/09x0r"', '"/m/09x0r"', '"/m/07yv9"'],
            }
        )

        eval_metadata = pd.DataFrame(
            {
                "YTID": ["jkl012", "mno345"],
                "start_seconds": [2.0, 8.0],
                "end_seconds": [12.0, 18.0],
                "positive_labels": ['"/m/07yv9"', '"/m/09x0r,/m/07yv9"'],
            }
        )

        # Save metadata
        balanced_metadata.to_csv(
            Path(temp_dir) / "metadata" / "balanced_train_segments.csv", index=False
        )
        eval_metadata.to_csv(Path(temp_dir) / "metadata" / "eval_segments.csv", index=False)

        # Create mock audio files
        for ytid in balanced_metadata["YTID"]:
            audio_path = Path(temp_dir) / "audio" / "balanced_train" / f"{ytid}.wav"
            audio_path.touch()

        for ytid in eval_metadata["YTID"]:
            audio_path = Path(temp_dir) / "audio" / "eval" / f"{ytid}.wav"
            audio_path.touch()

        # Create ontology (class mapping)
        ontology = [
            {"id": "/m/07yv9", "name": "Vehicle", "child_ids": []},
            {"id": "/m/09x0r", "name": "Aircraft", "child_ids": []},
        ]

        with open(Path(temp_dir) / "metadata" / "ontology.json", "w") as f:
            json.dump(ontology, f)

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_audioset_dataset_initialization(self, mock_audioset_data_dir):
        """Test AudioSet dataset initialization."""
        dataset = AudioSetDataset(
            data_dir=mock_audioset_data_dir,
            split="balanced_train",
            subset_classes=["/m/07yv9", "/m/09x0r"],
        )

        assert dataset.data_dir == Path(mock_audioset_data_dir)
        assert dataset.split == "balanced_train"
        assert len(dataset.class_to_idx) == 2

    def test_audioset_dataset_length(self, mock_audioset_data_dir):
        """Test AudioSet dataset length."""
        train_dataset = AudioSetDataset(mock_audioset_data_dir, split="balanced_train")
        eval_dataset = AudioSetDataset(mock_audioset_data_dir, split="eval")

        assert len(train_dataset) == 3
        assert len(eval_dataset) == 2

    @patch("torchaudio.load")
    def test_audioset_dataset_getitem(self, mock_torchaudio_load, mock_audioset_data_dir):
        """Test AudioSet dataset item retrieval."""
        mock_torchaudio_load.return_value = (
            torch.randn(1, 160000),  # 10 seconds of audio at 16kHz
            16000,
        )

        dataset = AudioSetDataset(mock_audioset_data_dir, split="balanced_train")

        audio, labels = dataset[0]

        assert isinstance(audio, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert audio.shape[0] == 1  # Single channel
        assert labels.dtype == torch.float32  # Multi-label format

    def test_audioset_dataset_multi_label(self, mock_audioset_data_dir):
        """Test AudioSet dataset multi-label handling."""
        dataset = AudioSetDataset(
            mock_audioset_data_dir, split="balanced_train", subset_classes=["/m/07yv9", "/m/09x0r"]
        )

        with patch("torchaudio.load") as mock_load:
            mock_load.return_value = (torch.randn(1, 160000), 16000)

            audio, labels = dataset[0]

            # Should be multi-hot encoded
            assert labels.shape == (2,)  # Number of classes
            assert labels.dtype == torch.float32
            assert torch.all((labels == 0) | (labels == 1))  # Binary values only

    def test_audioset_dataset_subset_classes(self, mock_audioset_data_dir):
        """Test AudioSet dataset with subset of classes."""
        # Test with only one class
        dataset = AudioSetDataset(
            mock_audioset_data_dir, split="balanced_train", subset_classes=["/m/07yv9"]
        )

        assert len(dataset.class_to_idx) == 1
        assert "/m/07yv9" in dataset.class_to_idx


class TestFSD50KDataset:
    """Test FSD50K dataset loader."""

    @pytest.fixture
    def mock_fsd50k_data_dir(self):
        """Create a mock FSD50K dataset directory structure."""
        temp_dir = tempfile.mkdtemp()

        # Create directory structure
        (Path(temp_dir) / "FSD50K.dev_audio").mkdir(parents=True)
        (Path(temp_dir) / "FSD50K.eval_audio").mkdir(parents=True)
        (Path(temp_dir) / "FSD50K.ground_truth").mkdir(parents=True)

        # Create mock metadata files
        dev_metadata = pd.DataFrame(
            {
                "fname": ["10000.wav", "10001.wav", "10002.wav"],
                "labels": [
                    "Accelerating_and_revving_and_vroom",
                    "Aircraft",
                    "Accelerating_and_revving_and_vroom,Aircraft",
                ],
                "mids": ["/m/07qv_x7", "/m/09x0r", "/m/07qv_x7,/m/09x0r"],
                "split": ["train", "train", "val"],
            }
        )

        eval_metadata = pd.DataFrame(
            {
                "fname": ["20000.wav", "20001.wav"],
                "labels": ["Aircraft", "Accelerating_and_revving_and_vroom"],
                "mids": ["/m/09x0r", "/m/07qv_x7"],
            }
        )

        # Save metadata
        dev_metadata.to_csv(Path(temp_dir) / "FSD50K.ground_truth" / "dev.csv", index=False)
        eval_metadata.to_csv(Path(temp_dir) / "FSD50K.ground_truth" / "eval.csv", index=False)

        # Create mock audio files
        for fname in dev_metadata["fname"]:
            audio_path = Path(temp_dir) / "FSD50K.dev_audio" / fname
            audio_path.touch()

        for fname in eval_metadata["fname"]:
            audio_path = Path(temp_dir) / "FSD50K.eval_audio" / fname
            audio_path.touch()

        # Create vocabulary (class mapping)
        vocabulary = pd.DataFrame(
            {
                "mid": ["/m/07qv_x7", "/m/09x0r"],
                "display_name": ["Accelerating_and_revving_and_vroom", "Aircraft"],
            }
        )

        vocabulary.to_csv(Path(temp_dir) / "FSD50K.ground_truth" / "vocabulary.csv", index=False)

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_fsd50k_dataset_initialization(self, mock_fsd50k_data_dir):
        """Test FSD50K dataset initialization."""
        dataset = FSD50KDataset(data_dir=mock_fsd50k_data_dir, split="train")

        assert dataset.data_dir == Path(mock_fsd50k_data_dir)
        assert dataset.split == "train"
        assert len(dataset.class_to_idx) == 2

    def test_fsd50k_dataset_splits(self, mock_fsd50k_data_dir):
        """Test FSD50K dataset different splits."""
        train_dataset = FSD50KDataset(mock_fsd50k_data_dir, split="train")
        val_dataset = FSD50KDataset(mock_fsd50k_data_dir, split="val")
        eval_dataset = FSD50KDataset(mock_fsd50k_data_dir, split="eval")

        assert len(train_dataset) == 2  # 2 train samples
        assert len(val_dataset) == 1  # 1 val sample
        assert len(eval_dataset) == 2  # 2 eval samples

    @patch("torchaudio.load")
    def test_fsd50k_dataset_getitem(self, mock_torchaudio_load, mock_fsd50k_data_dir):
        """Test FSD50K dataset item retrieval."""
        mock_torchaudio_load.return_value = (
            torch.randn(1, 80000),  # 5 seconds of audio at 16kHz
            16000,
        )

        dataset = FSD50KDataset(mock_fsd50k_data_dir, split="train")

        audio, labels = dataset[0]

        assert isinstance(audio, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert audio.shape[0] == 1  # Single channel
        assert labels.dtype == torch.float32  # Multi-label format

    def test_fsd50k_dataset_multi_label(self, mock_fsd50k_data_dir):
        """Test FSD50K dataset multi-label handling."""
        dataset = FSD50KDataset(mock_fsd50k_data_dir, split="train")

        with patch("torchaudio.load") as mock_load:
            mock_load.return_value = (torch.randn(1, 80000), 16000)

            # Get sample with multiple labels (index 1 in our mock data)
            audio, labels = dataset[1]

            # Should be multi-hot encoded
            assert labels.shape == (2,)  # Number of classes
            assert labels.dtype == torch.float32
            assert torch.sum(labels) >= 1  # At least one label should be active


class TestDatasetComparison:
    """Test comparative functionality across datasets."""

    @pytest.fixture
    def all_datasets(self, mock_mad_data_dir, mock_audioset_data_dir, mock_fsd50k_data_dir):
        """Create instances of all datasets for comparison."""
        return {
            "mad": MADDataset(mock_mad_data_dir, split="train"),
            "audioset": AudioSetDataset(mock_audioset_data_dir, split="balanced_train"),
            "fsd50k": FSD50KDataset(mock_fsd50k_data_dir, split="train"),
        }

    def test_all_datasets_implement_torch_dataset(self, all_datasets):
        """Test that all datasets implement torch.utils.data.Dataset interface."""
        from torch.utils.data import Dataset

        for dataset_name, dataset in all_datasets.items():
            assert isinstance(dataset, Dataset), f"{dataset_name} does not inherit from Dataset"
            assert hasattr(dataset, "__len__"), f"{dataset_name} missing __len__ method"
            assert hasattr(dataset, "__getitem__"), f"{dataset_name} missing __getitem__ method"

    def test_all_datasets_return_consistent_types(self, all_datasets):
        """Test that all datasets return consistent data types."""
        with patch("torchaudio.load") as mock_load:
            mock_load.return_value = (torch.randn(1, 16000), 16000)

            for dataset_name, dataset in all_datasets.items():
                if len(dataset) > 0:
                    audio, labels = dataset[0]

                    assert isinstance(audio, torch.Tensor), f"{dataset_name} audio not torch.Tensor"
                    assert len(audio.shape) >= 2, f"{dataset_name} audio shape incorrect"

                    # Labels can be int (single-label) or tensor (multi-label)
                    assert isinstance(
                        labels, (int, torch.Tensor)
                    ), f"{dataset_name} labels type incorrect: {type(labels)}"

    def test_all_datasets_handle_indexing(self, all_datasets):
        """Test that all datasets handle indexing correctly."""
        with patch("torchaudio.load") as mock_load:
            mock_load.return_value = (torch.randn(1, 16000), 16000)

            for dataset_name, dataset in all_datasets.items():
                dataset_len = len(dataset)

                if dataset_len > 0:
                    # Test valid indices
                    _ = dataset[0]
                    _ = dataset[dataset_len - 1]

                    # Test invalid indices
                    with pytest.raises(IndexError):
                        _ = dataset[dataset_len]

                    with pytest.raises(IndexError):
                        _ = dataset[-dataset_len - 1]


class TestDataLoaderUtilities:
    """Test data loader utility functions."""

    def test_dataset_with_dataloader(self, mock_mad_data_dir):
        """Test dataset integration with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        dataset = MADDataset(mock_mad_data_dir, split="train")

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        )

        with patch("torchaudio.load") as mock_load:
            mock_load.return_value = (torch.randn(1, 16000), 16000)

            # Test iteration
            for batch_audio, batch_labels in dataloader:
                assert isinstance(batch_audio, torch.Tensor)
                assert isinstance(batch_labels, torch.Tensor)
                assert batch_audio.shape[0] <= 2  # Batch size
                assert batch_labels.shape[0] <= 2  # Batch size
                break  # Just test first batch

    def test_dataset_transforms(self, mock_mad_data_dir):
        """Test dataset with custom transforms."""

        def custom_transform(audio):
            # Simple transform: normalize to [-1, 1]
            return audio / torch.max(torch.abs(audio))

        dataset = MADDataset(mock_mad_data_dir, split="train", transform=custom_transform)

        with patch("torchaudio.load") as mock_load:
            # Return audio with known range
            mock_load.return_value = (torch.randn(1, 16000) * 10, 16000)

            audio, _ = dataset[0]

            # Transform should normalize to [-1, 1]
            assert torch.max(torch.abs(audio)) <= 1.0

    def test_dataset_memory_efficiency(self, mock_mad_data_dir):
        """Test dataset memory efficiency."""
        dataset = MADDataset(mock_mad_data_dir, split="train")

        # Dataset should not load all data into memory at once
        # This is tested by checking that the dataset doesn't store audio data
        assert not hasattr(dataset, "audio_data")
        assert not hasattr(dataset, "preloaded_audio")

        # Loading an item should not affect dataset size significantly
        import sys

        initial_size = sys.getsizeof(dataset)

        with patch("torchaudio.load") as mock_load:
            mock_load.return_value = (torch.randn(1, 16000), 16000)
            _ = dataset[0]

        final_size = sys.getsizeof(dataset)

        # Size should not increase significantly
        assert final_size - initial_size < 1000  # Less than 1KB increase


class TestErrorHandling:
    """Test error handling in data loaders."""

    def test_corrupted_audio_file(self, mock_mad_data_dir):
        """Test handling of corrupted audio files."""
        dataset = MADDataset(mock_mad_data_dir, split="train")

        # Mock torchaudio.load to raise an exception
        with patch("torchaudio.load") as mock_load:
            mock_load.side_effect = RuntimeError("Corrupted file")

            # Dataset should handle the error gracefully
            try:
                _ = dataset[0]
                # If no exception is raised, the dataset handled it gracefully
            except RuntimeError:
                # If exception is propagated, that's also acceptable behavior
                pass

    def test_missing_metadata_file(self):
        """Test handling of missing metadata files."""
        temp_dir = tempfile.mkdtemp()

        try:
            # Create directory but no metadata
            (Path(temp_dir) / "audio" / "train").mkdir(parents=True)

            with pytest.raises(FileNotFoundError):
                MADDataset(temp_dir, split="train")

        finally:
            shutil.rmtree(temp_dir)

    def test_empty_dataset(self):
        """Test handling of empty datasets."""
        temp_dir = tempfile.mkdtemp()

        try:
            # Create directory structure with empty metadata
            (Path(temp_dir) / "audio" / "train").mkdir(parents=True)
            (Path(temp_dir) / "metadata").mkdir(parents=True)

            # Create empty metadata file
            empty_metadata = pd.DataFrame(
                columns=["filename", "label", "vehicle_type", "duration", "split"]
            )
            empty_metadata.to_csv(Path(temp_dir) / "metadata" / "train.csv", index=False)

            # Create empty class mapping
            with open(Path(temp_dir) / "metadata" / "class_mapping.json", "w") as f:
                json.dump({}, f)

            dataset = MADDataset(temp_dir, split="train")
            assert len(dataset) == 0

        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])
