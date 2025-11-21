"""
SereneSense Data Loaders Module
Comprehensive dataset loaders for military vehicle sound detection.

This module provides dataset loaders for:
- MAD (Military Audio Dataset): 8,075 military vehicle samples
- AudioSet: 2.08M general audio samples for pre-training
- FSD50K: 51,197 sound event samples for transfer learning

All loaders support:
- Automatic download and preprocessing
- Train/validation/test splits
- Data augmentation pipelines
- Efficient caching mechanisms
- Configuration-driven setup
"""

from .mad_loader import (
    MADDataset,
    MADConfig,
    MADDataModule,
    create_mad_dataloader
)

try:
    from .audioset_loader import (
        AudioSetDataset,
        AudioSetConfig,
        AudioSetDataModule,
        create_audioset_dataloader
    )
    # Backward compatibility alias
    AudioSetDataLoader = AudioSetDataModule
except ImportError:
    AudioSetDataset = None
    AudioSetConfig = None
    AudioSetDataModule = None
    AudioSetDataLoader = None
    create_audioset_dataloader = None

try:
    from .fsd50k_loader import (
        FSD50KDataset,
        FSD50KConfig,
        FSD50KDataModule,
        create_fsd50k_dataloader
    )
    # Backward compatibility alias
    FSD50KDataLoader = FSD50KDataModule
except ImportError:
    FSD50KDataset = None
    FSD50KConfig = None
    FSD50KDataModule = None
    FSD50KDataLoader = None
    create_fsd50k_dataloader = None

# Supported datasets
SUPPORTED_DATASETS = {
    'mad': {
        'dataset_class': MADDataset,
        'config_class': MADConfig,
        'datamodule_class': MADDataModule,
        'create_function': create_mad_dataloader,
        'description': 'Military Audio Dataset with 8,075 samples across 7 military classes',
        'license': 'CC BY 4.0',
        'samples': 8075,
        'classes': 7,
        'duration_hours': 12.0
    }
}

if AudioSetDataset is not None:
    SUPPORTED_DATASETS['audioset'] = {
        'dataset_class': AudioSetDataset,
        'config_class': AudioSetConfig,
        'datamodule_class': AudioSetDataModule,
        'create_function': create_audioset_dataloader,
        'description': 'Google AudioSet with 2.08M samples across 632 classes',
        'license': 'CC BY 4.0',
        'samples': 2084000,
        'classes': 632,
        'duration_hours': 5800.0
    }

if FSD50KDataset is not None:
    SUPPORTED_DATASETS['fsd50k'] = {
        'dataset_class': FSD50KDataset,
        'config_class': FSD50KConfig,
        'datamodule_class': FSD50KDataModule,
        'create_function': create_fsd50k_dataloader,
        'description': 'Freesound Dataset 50K with 51,197 samples across 200 classes',
        'license': 'CC BY 4.0',
        'samples': 51197,
        'classes': 200,
        'duration_hours': 108.0
    }


def get_dataset_info(dataset_name: str) -> dict:
    """
    Get information about a supported dataset.
    
    Args:
        dataset_name: Name of the dataset ('mad', 'audioset', 'fsd50k')
        
    Returns:
        Dictionary with dataset information
        
    Raises:
        ValueError: If dataset is not supported
    """
    if dataset_name not in SUPPORTED_DATASETS:
        available = list(SUPPORTED_DATASETS.keys())
        raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {available}")
    
    return SUPPORTED_DATASETS[dataset_name]


def list_datasets() -> list:
    """
    List all supported datasets.
    
    Returns:
        List of supported dataset names
    """
    return list(SUPPORTED_DATASETS.keys())


def create_dataloader(dataset_name: str, config_path: str, **kwargs):
    """
    Create a dataloader for any supported dataset.
    
    Args:
        dataset_name: Name of the dataset
        config_path: Path to configuration file
        **kwargs: Additional arguments for dataloader creation
        
    Returns:
        DataLoader instance
        
    Raises:
        ValueError: If dataset is not supported
    """
    dataset_info = get_dataset_info(dataset_name)
    create_function = dataset_info['create_function']
    
    if create_function is None:
        raise RuntimeError(f"Dataset '{dataset_name}' loader not available")
    
    return create_function(config_path, **kwargs)


__all__ = [
    # MAD Dataset
    'MADDataset',
    'MADConfig',
    'MADDataModule',
    'create_mad_dataloader',

    # AudioSet Dataset (if available)
    'AudioSetDataset',
    'AudioSetConfig',
    'AudioSetDataModule',
    'AudioSetDataLoader',  # Alias for backward compatibility
    'create_audioset_dataloader',
    
    # FSD50K Dataset (if available)
    'FSD50KDataset',
    'FSD50KConfig',
    'FSD50KDataModule',
    'FSD50KDataLoader',  # Alias for backward compatibility
    'create_fsd50k_dataloader',
    
    # Utility functions
    'get_dataset_info',
    'list_datasets',
    'create_dataloader',
    'SUPPORTED_DATASETS'
]