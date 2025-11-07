"""
Configuration Parser for SereneSense

This module provides comprehensive configuration parsing and validation
for the SereneSense military vehicle sound detection system.

Features:
- YAML configuration file parsing
- Configuration validation and schema checking
- Environment variable interpolation
- Configuration merging and inheritance
- Type checking and conversion
- Configuration templates and defaults

Example:
    >>> from core.utils.config_parser import ConfigParser, load_config
    >>> 
    >>> # Load configuration
    >>> config = load_config("configs/models/audioMAE.yaml")
    >>> 
    >>> # Use ConfigParser for advanced features
    >>> parser = ConfigParser()
    >>> config = parser.load_and_validate("configs/training/base.yaml")
    >>> 
    >>> # Merge configurations
    >>> base_config = load_config("configs/training/base.yaml")
    >>> model_config = load_config("configs/models/audioMAE.yaml")
    >>> merged = parser.merge_configs(base_config, model_config)
"""

import os
import re
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple
from dataclasses import dataclass
import logging
from copy import deepcopy

# Type definitions
ConfigDict = Dict[str, Any]
ConfigValue = Union[str, int, float, bool, List, Dict]

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a configuration validation error."""

    path: str
    message: str
    expected_type: Optional[str] = None
    actual_value: Any = None


class ConfigValidator:
    """
    Configuration validator with schema checking capabilities.
    """

    def __init__(self):
        """Initialize the validator."""
        self.errors: List[ValidationError] = []

    def validate_type(self, value: Any, expected_type: type, path: str) -> bool:
        """
        Validate that a value matches the expected type.

        Args:
            value: Value to validate
            expected_type: Expected type
            path: Configuration path for error reporting

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(value, expected_type):
            self.errors.append(
                ValidationError(
                    path=path,
                    message=f"Expected {expected_type.__name__}, got {type(value).__name__}",
                    expected_type=expected_type.__name__,
                    actual_value=value,
                )
            )
            return False
        return True

    def validate_range(
        self,
        value: Union[int, float],
        min_val: Optional[Union[int, float]] = None,
        max_val: Optional[Union[int, float]] = None,
        path: str = "",
    ) -> bool:
        """
        Validate that a numeric value is within a specified range.

        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            path: Configuration path for error reporting

        Returns:
            True if valid, False otherwise
        """
        if min_val is not None and value < min_val:
            self.errors.append(
                ValidationError(
                    path=path,
                    message=f"Value {value} is below minimum {min_val}",
                    actual_value=value,
                )
            )
            return False

        if max_val is not None and value > max_val:
            self.errors.append(
                ValidationError(
                    path=path,
                    message=f"Value {value} is above maximum {max_val}",
                    actual_value=value,
                )
            )
            return False

        return True

    def validate_choices(self, value: Any, choices: List[Any], path: str = "") -> bool:
        """
        Validate that a value is in a list of allowed choices.

        Args:
            value: Value to validate
            choices: List of allowed values
            path: Configuration path for error reporting

        Returns:
            True if valid, False otherwise
        """
        if value not in choices:
            self.errors.append(
                ValidationError(
                    path=path,
                    message=f"Value '{value}' not in allowed choices: {choices}",
                    actual_value=value,
                )
            )
            return False
        return True

    def validate_required_keys(
        self, config: ConfigDict, required_keys: List[str], path: str = ""
    ) -> bool:
        """
        Validate that required keys are present in configuration.

        Args:
            config: Configuration dictionary
            required_keys: List of required keys
            path: Configuration path for error reporting

        Returns:
            True if all required keys present, False otherwise
        """
        valid = True
        for key in required_keys:
            if key not in config:
                self.errors.append(
                    ValidationError(
                        path=f"{path}.{key}" if path else key,
                        message=f"Required key '{key}' is missing",
                    )
                )
                valid = False
        return valid

    def clear_errors(self) -> None:
        """Clear all validation errors."""
        self.errors.clear()

    def get_error_summary(self) -> str:
        """
        Get a formatted summary of validation errors.

        Returns:
            Formatted error summary
        """
        if not self.errors:
            return "No validation errors"

        summary = f"Found {len(self.errors)} validation error(s):\n"
        for i, error in enumerate(self.errors, 1):
            summary += f"  {i}. {error.path}: {error.message}\n"

        return summary


class ConfigParser:
    """
    Comprehensive configuration parser for SereneSense.

    Handles YAML parsing, validation, environment variable interpolation,
    and configuration merging.
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the configuration parser.

        Args:
            base_dir: Base directory for relative path resolution
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.validator = ConfigValidator()
        self.env_pattern = re.compile(r"\$\{([^}]+)\}")

    def load_yaml(self, file_path: Union[str, Path]) -> ConfigDict:
        """
        Load a YAML configuration file.

        Args:
            file_path: Path to YAML file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        file_path = Path(file_path)

        # Resolve relative paths
        if not file_path.is_absolute():
            file_path = self.base_dir / file_path

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if config is None:
                config = {}

            logger.debug(f"Loaded configuration from {file_path}")
            return config

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading {file_path}: {e}")
            raise

    def save_yaml(self, config: ConfigDict, file_path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.

        Args:
            config: Configuration dictionary
            file_path: Output file path
        """
        file_path = Path(file_path)

        # Resolve relative paths
        if not file_path.is_absolute():
            file_path = self.base_dir / file_path

        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)

            logger.debug(f"Saved configuration to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {e}")
            raise

    def interpolate_environment_variables(self, config: ConfigDict) -> ConfigDict:
        """
        Replace environment variable placeholders in configuration.

        Supports syntax: ${VAR_NAME} and ${VAR_NAME:default_value}

        Args:
            config: Configuration dictionary

        Returns:
            Configuration with environment variables interpolated
        """

        def interpolate_value(value: Any) -> Any:
            if isinstance(value, str):
                return self._interpolate_string(value)
            elif isinstance(value, dict):
                return {k: interpolate_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [interpolate_value(item) for item in value]
            else:
                return value

        return interpolate_value(config)

    def _interpolate_string(self, text: str) -> str:
        """
        Interpolate environment variables in a string.

        Args:
            text: String with possible environment variable placeholders

        Returns:
            String with environment variables replaced
        """

        def replace_var(match):
            var_expr = match.group(1)

            # Check for default value
            if ":" in var_expr:
                var_name, default = var_expr.split(":", 1)
            else:
                var_name, default = var_expr, None

            # Get environment variable
            value = os.environ.get(var_name.strip())

            if value is not None:
                return value
            elif default is not None:
                return default
            else:
                logger.warning(f"Environment variable '{var_name}' not found")
                return match.group(0)  # Return original placeholder

        return self.env_pattern.sub(replace_var, text)

    def merge_configs(self, *configs: ConfigDict) -> ConfigDict:
        """
        Merge multiple configuration dictionaries.

        Later configurations override earlier ones. Nested dictionaries
        are merged recursively.

        Args:
            *configs: Configuration dictionaries to merge

        Returns:
            Merged configuration dictionary
        """
        if not configs:
            return {}

        result = deepcopy(configs[0])

        for config in configs[1:]:
            result = self._deep_merge(result, config)

        return result

    def _deep_merge(self, base: ConfigDict, override: ConfigDict) -> ConfigDict:
        """
        Recursively merge two configuration dictionaries.

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        result = deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)

        return result

    def validate_model_config(self, config: ConfigDict) -> bool:
        """
        Validate model configuration structure and values.

        Args:
            config: Model configuration dictionary

        Returns:
            True if valid, False otherwise
        """
        self.validator.clear_errors()

        # Check required top-level keys
        required_keys = ["model", "training", "hardware"]
        if not self.validator.validate_required_keys(config, required_keys):
            return False

        # Validate model section
        model_config = config.get("model", {})
        if isinstance(model_config, dict):
            model_required = ["name", "type", "architecture"]
            self.validator.validate_required_keys(model_config, model_required, "model")

        # Validate training section
        training_config = config.get("training", {})
        if isinstance(training_config, dict):
            training_required = ["epochs", "batch_size", "optimizer", "lr_scheduler"]
            self.validator.validate_required_keys(training_config, training_required, "training")

            # Validate numeric ranges
            if "epochs" in training_config:
                self.validator.validate_range(training_config["epochs"], 1, 1000, "training.epochs")

            if "batch_size" in training_config:
                self.validator.validate_range(
                    training_config["batch_size"], 1, 1024, "training.batch_size"
                )

        # Validate hardware section
        hardware_config = config.get("hardware", {})
        if isinstance(hardware_config, dict):
            if "device" in hardware_config:
                allowed_devices = ["auto", "cuda", "cpu", "mps"]
                self.validator.validate_choices(
                    hardware_config["device"], allowed_devices, "hardware.device"
                )

        return len(self.validator.errors) == 0

    def validate_data_config(self, config: ConfigDict) -> bool:
        """
        Validate data configuration structure and values.

        Args:
            config: Data configuration dictionary

        Returns:
            True if valid, False otherwise
        """
        self.validator.clear_errors()

        # Check required top-level keys
        required_keys = ["dataset", "source", "splits"]
        if not self.validator.validate_required_keys(config, required_keys):
            return False

        # Validate dataset section
        dataset_config = config.get("dataset", {})
        if isinstance(dataset_config, dict):
            dataset_required = ["name", "statistics"]
            self.validator.validate_required_keys(dataset_config, dataset_required, "dataset")

        # Validate splits section
        splits_config = config.get("splits", {})
        if isinstance(splits_config, dict):
            if "ratios" in splits_config:
                ratios = splits_config["ratios"]
                if isinstance(ratios, dict):
                    # Check that ratios sum to approximately 1.0
                    total = sum(ratios.values())
                    if abs(total - 1.0) > 0.01:
                        self.validator.errors.append(
                            ValidationError(
                                path="splits.ratios",
                                message=f"Split ratios sum to {total:.3f}, expected ~1.0",
                            )
                        )

        return len(self.validator.errors) == 0

    def get_config_template(self, config_type: str) -> ConfigDict:
        """
        Get a configuration template for the specified type.

        Args:
            config_type: Type of configuration (model, training, data, etc.)

        Returns:
            Configuration template dictionary
        """
        templates = {
            "model": {
                "model": {"name": "placeholder", "type": "placeholder", "architecture": {}},
                "training": {
                    "epochs": 100,
                    "batch_size": 32,
                    "optimizer": {"type": "AdamW", "lr": 1e-4},
                    "lr_scheduler": {"type": "cosine_annealing"},
                },
                "hardware": {"device": "auto", "mixed_precision": True},
            },
            "training": {
                "experiment": {"name": "experiment", "description": "Training experiment"},
                "training": {"epochs": 100, "batch_size": 32},
                "optimizer": {"type": "AdamW", "lr": 1e-4},
                "lr_scheduler": {"type": "cosine_annealing"},
            },
            "data": {
                "dataset": {"name": "placeholder", "statistics": {}},
                "source": {"download": {}},
                "splits": {
                    "strategy": "stratified",
                    "ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
                },
            },
        }

        return templates.get(config_type, {})

    def load_and_validate(
        self, file_path: Union[str, Path], config_type: Optional[str] = None
    ) -> ConfigDict:
        """
        Load and validate a configuration file.

        Args:
            file_path: Path to configuration file
            config_type: Type of configuration for validation

        Returns:
            Validated configuration dictionary

        Raises:
            ValueError: If validation fails
        """
        # Load configuration
        config = self.load_yaml(file_path)

        # Interpolate environment variables
        config = self.interpolate_environment_variables(config)

        # Validate if type specified
        if config_type:
            if config_type == "model":
                valid = self.validate_model_config(config)
            elif config_type == "data":
                valid = self.validate_data_config(config)
            else:
                logger.warning(f"Unknown config type '{config_type}', skipping validation")
                valid = True

            if not valid:
                error_summary = self.validator.get_error_summary()
                raise ValueError(f"Configuration validation failed:\n{error_summary}")

        return config

    def create_config_from_template(
        self, config_type: str, output_path: Union[str, Path], **kwargs
    ) -> ConfigDict:
        """
        Create a configuration file from a template.

        Args:
            config_type: Type of configuration template
            output_path: Output file path
            **kwargs: Template parameter overrides

        Returns:
            Created configuration dictionary
        """
        # Get template
        template = self.get_config_template(config_type)

        # Apply overrides
        if kwargs:
            template = self.merge_configs(template, kwargs)

        # Save to file
        self.save_yaml(template, output_path)

        logger.info(f"Created {config_type} configuration template at {output_path}")

        return template


# Convenience functions
def load_config(file_path: Union[str, Path], base_dir: Optional[str] = None) -> ConfigDict:
    """
    Load a configuration file.

    Args:
        file_path: Path to configuration file
        base_dir: Base directory for relative path resolution

    Returns:
        Configuration dictionary
    """
    parser = ConfigParser(base_dir)
    return parser.load_yaml(file_path)


def load_and_merge_configs(
    *file_paths: Union[str, Path], base_dir: Optional[str] = None
) -> ConfigDict:
    """
    Load and merge multiple configuration files.

    Args:
        *file_paths: Paths to configuration files
        base_dir: Base directory for relative path resolution

    Returns:
        Merged configuration dictionary
    """
    parser = ConfigParser(base_dir)
    configs = [parser.load_yaml(path) for path in file_paths]
    return parser.merge_configs(*configs)


def validate_config(config: ConfigDict, config_type: str) -> Tuple[bool, List[str]]:
    """
    Validate a configuration dictionary.

    Args:
        config: Configuration dictionary
        config_type: Type of configuration

    Returns:
        Tuple of (is_valid, error_messages)
    """
    parser = ConfigParser()

    if config_type == "model":
        valid = parser.validate_model_config(config)
    elif config_type == "data":
        valid = parser.validate_data_config(config)
    else:
        return True, []  # Unknown type, assume valid

    error_messages = [f"{error.path}: {error.message}" for error in parser.validator.errors]

    return valid, error_messages


def interpolate_env_vars(config: ConfigDict) -> ConfigDict:
    """
    Interpolate environment variables in configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configuration with environment variables interpolated
    """
    parser = ConfigParser()
    return parser.interpolate_environment_variables(config)


def create_config_template(config_type: str, output_path: Union[str, Path], **kwargs) -> ConfigDict:
    """
    Create a configuration template file.

    Args:
        config_type: Type of configuration template
        output_path: Output file path
        **kwargs: Template parameter overrides

    Returns:
        Created configuration dictionary
    """
    parser = ConfigParser()
    return parser.create_config_from_template(config_type, output_path, **kwargs)


# Configuration utilities for specific use cases
class TrainingConfigBuilder:
    """
    Builder class for training configurations.
    """

    def __init__(self):
        """Initialize the builder."""
        self.config = {
            "experiment": {"name": "experiment"},
            "training": {},
            "optimizer": {},
            "lr_scheduler": {},
            "hardware": {},
        }

    def set_experiment_name(self, name: str) -> "TrainingConfigBuilder":
        """Set experiment name."""
        self.config["experiment"]["name"] = name
        return self

    def set_epochs(self, epochs: int) -> "TrainingConfigBuilder":
        """Set number of training epochs."""
        self.config["training"]["epochs"] = epochs
        return self

    def set_batch_size(self, batch_size: int) -> "TrainingConfigBuilder":
        """Set batch size."""
        self.config["training"]["batch_size"] = batch_size
        return self

    def set_optimizer(self, optimizer_type: str, **kwargs) -> "TrainingConfigBuilder":
        """Set optimizer configuration."""
        self.config["optimizer"]["type"] = optimizer_type
        self.config["optimizer"].update(kwargs)
        return self

    def set_lr_scheduler(self, scheduler_type: str, **kwargs) -> "TrainingConfigBuilder":
        """Set learning rate scheduler configuration."""
        self.config["lr_scheduler"]["type"] = scheduler_type
        self.config["lr_scheduler"].update(kwargs)
        return self

    def set_device(self, device: str) -> "TrainingConfigBuilder":
        """Set device configuration."""
        self.config["hardware"]["device"] = device
        return self

    def build(self) -> ConfigDict:
        """Build the configuration."""
        return deepcopy(self.config)

    def save(self, file_path: Union[str, Path]) -> ConfigDict:
        """Save configuration to file."""
        config = self.build()
        parser = ConfigParser()
        parser.save_yaml(config, file_path)
        return config
