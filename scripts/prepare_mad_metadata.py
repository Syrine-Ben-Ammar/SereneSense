"""
Prepare MAD dataset metadata.csv from training.csv and test.csv files.

This script combines the training and test CSV files into a single metadata.csv
file that the MAD data loader expects.
"""

import pandas as pd
import argparse
from pathlib import Path
import logging

# MAD dataset class mapping (from mad_loader.py)
MAD_CLASSES = {
    "Helicopter": 0,
    "Fighter Aircraft": 1,
    "Military Vehicle": 2,
    "Truck": 3,
    "Foot Movement": 4,
    "Speech": 5,
    "Background": 6,
}

# Inverse mapping: label_id -> class_name
LABEL_TO_CLASS = {v: k for k, v in MAD_CLASSES.items()}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_metadata(data_dir: Path, train_val_split: float = 0.85):
    """
    Prepare metadata.csv from training.csv and test.csv.

    Args:
        data_dir: Path to MAD dataset directory
        train_val_split: Fraction of training data to use for training (rest for validation)
    """
    logger.info(f"Preparing metadata for MAD dataset at: {data_dir}")

    # Read CSV files
    training_csv = data_dir / "training.csv"
    test_csv = data_dir / "test.csv"

    if not training_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {training_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    logger.info(f"Reading training data from: {training_csv}")
    train_df = pd.read_csv(training_csv)
    logger.info(f"Reading test data from: {test_csv}")
    test_df = pd.read_csv(test_csv)

    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Test samples: {len(test_df)}")

    # Process training data
    train_df = train_df.rename(columns={"path": "filename", "label": "class_id"})
    train_df["class"] = train_df["class_id"].map(LABEL_TO_CLASS)

    # Split training data into train and validation
    # Use stratified sampling to maintain class distribution
    from sklearn.model_selection import train_test_split

    train_indices, val_indices = train_test_split(
        range(len(train_df)),
        test_size=1 - train_val_split,
        stratify=train_df["class_id"],
        random_state=42
    )

    train_df.loc[train_indices, "split"] = "train"
    train_df.loc[val_indices, "split"] = "val"

    logger.info(f"Train split: {len(train_indices)} samples")
    logger.info(f"Validation split: {len(val_indices)} samples")

    # Process test data
    test_df = test_df.rename(columns={"path": "filename", "label": "class_id"})
    test_df["class"] = test_df["class_id"].map(LABEL_TO_CLASS)
    test_df["split"] = "test"

    # Combine all data
    metadata_df = pd.concat([train_df, test_df], ignore_index=True)

    # Select only the required columns
    metadata_df = metadata_df[["filename", "class", "class_id", "split"]]

    # Save metadata.csv
    output_path = data_dir / "metadata.csv"
    metadata_df.to_csv(output_path, index=False)
    logger.info(f"Metadata saved to: {output_path}")

    # Print statistics
    logger.info("\n=== Dataset Statistics ===")
    logger.info(f"Total samples: {len(metadata_df)}")
    logger.info("\nSplit distribution:")
    logger.info(metadata_df["split"].value_counts())
    logger.info("\nClass distribution:")
    logger.info(metadata_df.groupby(["split", "class"]).size().unstack(fill_value=0))

    # Verify audio files exist
    logger.info("\nVerifying audio files...")
    audio_dir = data_dir / "audio"
    missing_files = []
    for idx, row in metadata_df.iterrows():
        audio_path = audio_dir / row["filename"]
        if not audio_path.exists():
            missing_files.append(row["filename"])

    if missing_files:
        logger.warning(f"Found {len(missing_files)} missing audio files:")
        for f in missing_files[:10]:  # Show first 10
            logger.warning(f"  - {f}")
        if len(missing_files) > 10:
            logger.warning(f"  ... and {len(missing_files) - 10} more")
    else:
        logger.info("All audio files verified successfully!")

    return metadata_df


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MAD dataset metadata.csv from training.csv and test.csv"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/mad",
        help="Path to MAD dataset directory (default: data/raw/mad)"
    )
    parser.add_argument(
        "--train-val-split",
        type=float,
        default=0.85,
        help="Fraction of training data to use for training vs validation (default: 0.85)"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    prepare_metadata(data_dir, args.train_val_split)
    logger.info("\nMetadata preparation complete!")


if __name__ == "__main__":
    main()
