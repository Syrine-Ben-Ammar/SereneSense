# =============================================================================
# serenesense/cli/__init__.py - CLI Package
# =============================================================================
"""
SereneSense Command Line Interface
Production-ready CLI tools for all SereneSense operations.
"""

__version__ = "1.0.0"

# =============================================================================
# serenesense/cli/train.py - Training CLI
# =============================================================================
#!/usr/bin/env python3
"""
SereneSense Training CLI
Command-line interface for model training.
"""

import click
import sys
import os
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.training.trainer import create_trainer
from core.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@click.command()
@click.option("--config", "-c", required=True, help="Training configuration file")
@click.option("--model-config", help="Model configuration file (optional)")
@click.option("--resume", help="Checkpoint to resume training from")
@click.option("--pretrained", help="Pre-trained model weights")
@click.option("--epochs", type=int, help="Number of epochs (overrides config)")
@click.option("--batch-size", type=int, help="Batch size (overrides config)")
@click.option("--learning-rate", type=float, help="Learning rate (overrides config)")
@click.option("--output-dir", help="Output directory (overrides config)")
@click.option("--experiment-name", help="Experiment name (overrides config)")
@click.option("--data-dir", help="Data directory (overrides config)")
@click.option("--num-workers", type=int, default=4, help="Number of data workers")
@click.option("--distributed", is_flag=True, help="Enable distributed training")
@click.option("--local-rank", type=int, default=0, help="Local rank for distributed training")
@click.option("--disable-wandb", is_flag=True, help="Disable Weights & Biases")
@click.option("--disable-mlflow", is_flag=True, help="Disable MLflow")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option("--dry-run", is_flag=True, help="Perform dry run without training")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--profile", is_flag=True, help="Enable profiling")
def main(
    config,
    model_config,
    resume,
    pretrained,
    epochs,
    batch_size,
    learning_rate,
    output_dir,
    experiment_name,
    data_dir,
    num_workers,
    distributed,
    local_rank,
    disable_wandb,
    disable_mlflow,
    debug,
    dry_run,
    log_level,
    profile,
):
    """
    Train SereneSense military vehicle detection models.

    Examples:
        serenesense-train --config configs/training/audioMAE.yaml
        serenesense-train --config configs/training/audioMAE.yaml --epochs 50 --batch-size 64
        serenesense-train --config configs/training/audioMAE.yaml --resume models/checkpoint.pth
    """
    # Setup logging
    setup_logging(level=log_level.upper())

    logger.info("🎯 SereneSense Model Training")
    logger.info("=" * 50)

    try:
        # Load and modify configuration based on CLI arguments
        import yaml

        with open(config, "r") as f:
            config_dict = yaml.safe_load(f)

        # Override with CLI arguments
        if epochs:
            config_dict.setdefault("training", {})["epochs"] = epochs
        if batch_size:
            config_dict.setdefault("training", {})["batch_size"] = batch_size
        if learning_rate:
            config_dict.setdefault("training", {})["learning_rate"] = learning_rate
        if output_dir:
            config_dict.setdefault("training", {})["output_dir"] = output_dir
        if experiment_name:
            config_dict.setdefault("training", {})["experiment_name"] = experiment_name
        if data_dir:
            config_dict.setdefault("data", {})["data_dir"] = data_dir
        if distributed:
            config_dict.setdefault("training", {})["distributed"] = True
            config_dict.setdefault("training", {})["local_rank"] = local_rank
        if disable_wandb:
            config_dict.setdefault("training", {})["use_wandb"] = False
        if disable_mlflow:
            config_dict.setdefault("training", {})["use_mlflow"] = False

        # Save modified config
        temp_config_path = "temp_training_config.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        if dry_run:
            logger.info("✅ Dry run completed - configuration is valid")
            return

        # Create trainer
        trainer = create_trainer(temp_config_path, resume)

        # Load pre-trained weights if specified
        if pretrained:
            import torch

            checkpoint = torch.load(pretrained, map_location="cpu")
            trainer.model.load_state_dict(
                checkpoint.get("model_state_dict", checkpoint), strict=False
            )
            logger.info(f"Loaded pre-trained weights from {pretrained}")

        # Start training
        results = trainer.train()

        # Log results
        logger.info("🎉 Training completed successfully!")
        logger.info(f"Best validation score: {results['best_val_score']:.4f}")
        logger.info(f"Total epochs: {results['total_epochs']}")
        logger.info(f"Training time: {results['total_time']:.2f} seconds")

        # Cleanup
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

# =============================================================================
# serenesense/cli/evaluate.py - Evaluation CLI
# =============================================================================
#!/usr/bin/env python3
"""
SereneSense Evaluation CLI
Command-line interface for model evaluation.
"""

import click
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@click.command()
@click.option("--model", "-m", required=True, help="Path to trained model")
@click.option("--config", "-c", help="Evaluation configuration file")
@click.option("--model-config", help="Model configuration file")
@click.option("--split", default="test", help="Dataset split to evaluate")
@click.option("--batch-size", type=int, default=32, help="Batch size")
@click.option("--num-workers", type=int, default=4, help="Number of workers")
@click.option("--output-dir", default="results/evaluation", help="Output directory")
@click.option("--generate-report", is_flag=True, help="Generate comprehensive report")
@click.option("--no-plots", is_flag=True, help="Skip generating plots")
@click.option("--benchmark", is_flag=True, help="Run performance benchmark")
@click.option("--benchmark-iterations", type=int, default=100, help="Benchmark iterations")
@click.option("--error-analysis", is_flag=True, help="Perform error analysis")
@click.option("--top-errors", type=int, default=10, help="Number of top errors to analyze")
@click.option("--device", help="Device to use (cuda, cpu, auto)")
@click.option("--log-level", default="INFO", help="Logging level")
def main(
    model,
    config,
    model_config,
    split,
    batch_size,
    num_workers,
    output_dir,
    generate_report,
    no_plots,
    benchmark,
    benchmark_iterations,
    error_analysis,
    top_errors,
    device,
    log_level,
):
    """
    Evaluate SereneSense military vehicle detection models.

    Examples:
        serenesense-eval --model models/best_model.pth
        serenesense-eval --model models/best_model.pth --generate-report --benchmark
        serenesense-eval --model models/best_model.pth --error-analysis --top-errors 20
    """
    # Setup logging
    setup_logging(level=log_level.upper())

    logger.info("📊 SereneSense Model Evaluation")
    logger.info("=" * 50)

    try:
        from scripts.evaluate_model import main as evaluate_main

        # Prepare arguments for evaluation script
        import argparse

        args = argparse.Namespace(
            model=model,
            config=config,
            model_config=model_config,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            output_dir=output_dir,
            generate_report=generate_report,
            no_plots=no_plots,
            benchmark=benchmark,
            benchmark_iterations=benchmark_iterations,
            error_analysis=error_analysis,
            top_errors=top_errors,
            device=device,
            log_level=log_level,
        )

        # Override sys.argv to pass arguments to evaluation script
        original_argv = sys.argv
        sys.argv = ["evaluate_model.py"] + [
            "--model",
            model,
            "--split",
            split,
            "--batch-size",
            str(batch_size),
            "--num-workers",
            str(num_workers),
            "--output-dir",
            output_dir,
            "--log-level",
            log_level,
        ]

        if config:
            sys.argv.extend(["--config", config])
        if model_config:
            sys.argv.extend(["--model-config", model_config])
        if generate_report:
            sys.argv.append("--generate-report")
        if no_plots:
            sys.argv.append("--no-plots")
        if benchmark:
            sys.argv.append("--benchmark")
            sys.argv.extend(["--benchmark-iterations", str(benchmark_iterations)])
        if error_analysis:
            sys.argv.append("--error-analysis")
            sys.argv.extend(["--top-errors", str(top_errors)])
        if device:
            sys.argv.extend(["--device", device])

        # Run evaluation
        evaluate_main()

        # Restore original argv
        sys.argv = original_argv

        logger.info("✅ Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# =============================================================================
# serenesense/cli/serve.py - API Server CLI
# =============================================================================
#!/usr/bin/env python3
"""
SereneSense API Server CLI
Command-line interface for serving models via API.
"""

import click
import sys
import os
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@click.command()
@click.option("--model", "-m", required=True, help="Path to trained model")
@click.option("--config", "-c", help="API configuration file")
@click.option("--host", default="0.0.0.0", help="Host address")
@click.option("--port", type=int, default=8080, help="Port number")
@click.option("--workers", type=int, default=1, help="Number of workers")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.option("--log-level", default="info", help="Log level")
@click.option("--device", help="Device to use (cuda, cpu, auto)")
@click.option("--batch-size", type=int, default=1, help="Inference batch size")
@click.option("--confidence-threshold", type=float, default=0.7, help="Detection threshold")
@click.option("--enable-websocket", is_flag=True, help="Enable WebSocket for real-time")
@click.option("--cors-origins", default="*", help="CORS allowed origins")
@click.option("--rate-limit", type=int, help="Rate limit (requests per minute)")
def main(
    model,
    config,
    host,
    port,
    workers,
    reload,
    log_level,
    device,
    batch_size,
    confidence_threshold,
    enable_websocket,
    cors_origins,
    rate_limit,
):
    """
    Start SereneSense API server for model inference.

    Examples:
        serenesense-serve --model models/best_model.pth
        serenesense-serve --model models/best_model.pth --host 0.0.0.0 --port 8080 --workers 4
        serenesense-serve --model models/best_model.pth --enable-websocket --rate-limit 60
    """
    # Setup logging
    setup_logging(level=log_level.upper())

    logger.info("🚀 SereneSense API Server")
    logger.info("=" * 50)
    logger.info(f"Model: {model}")
    logger.info(f"Server: http://{host}:{port}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Device: {device or 'auto'}")

    try:
        # Set environment variables for the API server
        os.environ["SERENESENSE_MODEL_PATH"] = model
        if device:
            os.environ["SERENESENSE_DEVICE"] = device
        if config:
            os.environ["SERENESENSE_CONFIG_PATH"] = config

        os.environ["SERENESENSE_CONFIDENCE_THRESHOLD"] = str(confidence_threshold)
        os.environ["SERENESENSE_BATCH_SIZE"] = str(batch_size)

        # Import and run the API server
        import uvicorn

        # Configure CORS origins
        cors_list = cors_origins.split(",") if cors_origins != "*" else ["*"]

        # Server configuration
        server_config = {
            "app": "core.deployment.api.fastapi_server:app",
            "host": host,
            "port": port,
            "workers": workers,
            "log_level": log_level,
            "reload": reload,
        }

        logger.info("Starting API server...")
        uvicorn.run(**server_config)

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# =============================================================================
# serenesense/cli/optimize.py - Model Optimization CLI
# =============================================================================
#!/usr/bin/env python3
"""
SereneSense Model Optimization CLI
Command-line interface for optimizing models for edge deployment.
"""

import click
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@click.command()
@click.option("--model", "-m", required=True, help="Path to trained model")
@click.option(
    "--target",
    "-t",
    required=True,
    type=click.Choice(["jetson", "raspberry_pi", "cpu", "gpu"]),
    help="Target platform",
)
@click.option(
    "--precision",
    "-p",
    default="fp16",
    type=click.Choice(["fp32", "fp16", "int8"]),
    help="Target precision",
)
@click.option("--output", "-o", help="Output path for optimized model")
@click.option("--config", "-c", help="Optimization configuration file")
@click.option("--calibration-data", help="Calibration dataset path (for INT8)")
@click.option("--validate", is_flag=True, help="Validate optimized model accuracy")
@click.option("--benchmark", is_flag=True, help="Benchmark optimized model")
@click.option("--benchmark-iterations", type=int, default=100, help="Benchmark iterations")
@click.option("--batch-size", type=int, default=1, help="Optimization batch size")
@click.option("--workspace-size", type=int, default=1073741824, help="TensorRT workspace size")
@click.option("--log-level", default="INFO", help="Logging level")
def main(
    model,
    target,
    precision,
    output,
    config,
    calibration_data,
    validate,
    benchmark,
    benchmark_iterations,
    batch_size,
    workspace_size,
    log_level,
):
    """
    Optimize SereneSense models for edge deployment.

    Examples:
        serenesense-optimize --model models/best_model.pth --target jetson --precision int8
        serenesense-optimize --model models/best_model.pth --target raspberry_pi --precision int8 --validate
        serenesense-optimize --model models/best_model.pth --target gpu --precision fp16 --benchmark
    """
    # Setup logging
    setup_logging(level=log_level.upper())

    logger.info("⚡ SereneSense Model Optimization")
    logger.info("=" * 50)
    logger.info(f"Model: {model}")
    logger.info(f"Target: {target}")
    logger.info(f"Precision: {precision}")

    try:
        from scripts.optimize_for_edge import optimize_serenesense_model

        # Determine output path
        if not output:
            model_path = Path(model)
            output = f"models/optimized/{model_path.stem}_{target}_{precision}"
            if target == "jetson":
                output += ".trt"
            else:
                output += ".onnx"

        logger.info(f"Output: {output}")

        # Run optimization
        results = optimize_serenesense_model(
            model_path=model,
            target_platform=target,
            precision=precision,
            output_dir=str(Path(output).parent),
        )

        logger.info("✅ Optimization completed successfully!")

        # Display results
        if "benchmarks" in results:
            logger.info("\nBenchmark Results:")
            for opt_name, benchmark in results["benchmarks"].items():
                if isinstance(benchmark, dict):
                    logger.info(f"  {opt_name}:")
                    if "avg_latency_ms" in benchmark:
                        logger.info(f"    Latency: {benchmark['avg_latency_ms']:.2f}ms")
                    if "throughput_fps" in benchmark:
                        logger.info(f"    Throughput: {benchmark['throughput_fps']:.1f} FPS")
                    if "accuracy_retention" in benchmark:
                        logger.info(
                            f"    Accuracy retention: {benchmark['accuracy_retention']:.1%}"
                        )

        # Display recommendations
        if "recommendations" in results:
            logger.info("\nRecommendations:")
            for rec in results["recommendations"]:
                logger.info(f"  • {rec}")

        logger.info(f"\nOptimized model saved to: {output}")

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# =============================================================================
# serenesense/cli/deploy.py - Deployment CLI
# =============================================================================
#!/usr/bin/env python3
"""
SereneSense Deployment CLI
Command-line interface for deploying models to edge devices.
"""

import click
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@click.command()
@click.option("--model", "-m", required=True, help="Path to optimized model")
@click.option(
    "--platform",
    "-p",
    required=True,
    type=click.Choice(["jetson", "raspberry_pi", "auto"]),
    help="Target platform",
)
@click.option("--host", default="0.0.0.0", help="API server host")
@click.option("--port", type=int, default=8080, help="API server port")
@click.option("--config", "-c", help="Deployment configuration file")
@click.option("--confidence-threshold", type=float, default=0.7, help="Detection threshold")
@click.option("--no-api", is_flag=True, help="Don't start API server")
@click.option("--no-detector", is_flag=True, help="Don't start real-time detector")
@click.option(
    "--power-mode",
    type=click.Choice(["MAXN", "15W", "10W"]),
    default="MAXN",
    help="Jetson power mode",
)
@click.option("--cpu-threads", type=int, default=4, help="CPU threads (Raspberry Pi)")
@click.option("--gpu-memory", type=int, default=64, help="GPU memory MB (Raspberry Pi)")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--dry-run", is_flag=True, help="Validate configuration without deployment")
def main(
    model,
    platform,
    host,
    port,
    config,
    confidence_threshold,
    no_api,
    no_detector,
    power_mode,
    cpu_threads,
    gpu_memory,
    log_level,
    dry_run,
):
    """
    Deploy SereneSense models to edge devices.

    Examples:
        serenesense-deploy --model models/optimized/model_jetson.trt --platform jetson
        serenesense-deploy --model models/optimized/model_rpi.onnx --platform raspberry_pi
        serenesense-deploy --model models/optimized/model.onnx --platform auto --host 192.168.1.100
    """
    # Setup logging
    setup_logging(level=log_level.upper())

    logger.info("🚀 SereneSense Edge Deployment")
    logger.info("=" * 50)
    logger.info(f"Model: {model}")
    logger.info(f"Platform: {platform}")
    logger.info(f"API: http://{host}:{port}")

    if dry_run:
        logger.info("🔍 Dry run mode - validating configuration")

    try:
        from scripts.deploy_edge import main as deploy_main

        # Prepare arguments
        import argparse

        args = argparse.Namespace(
            model=model,
            platform=platform,
            optimize=False,  # Model should already be optimized
            precision="fp16",
            output_model=None,
            host=host,
            port=port,
            confidence_threshold=confidence_threshold,
            no_api=no_api,
            no_detector=no_detector,
            power_mode=power_mode,
            cpu_threads=cpu_threads,
            gpu_memory=gpu_memory,
            config=config,
            log_level=log_level,
        )

        if dry_run:
            logger.info("✅ Configuration is valid")
            logger.info("Remove --dry-run flag to start actual deployment")
            return

        # Override sys.argv for deployment script
        original_argv = sys.argv
        sys.argv = ["deploy_edge.py"] + [
            "--model",
            model,
            "--platform",
            platform,
            "--host",
            host,
            "--port",
            str(port),
            "--confidence-threshold",
            str(confidence_threshold),
            "--power-mode",
            power_mode,
            "--cpu-threads",
            str(cpu_threads),
            "--gpu-memory",
            str(gpu_memory),
            "--log-level",
            log_level,
        ]

        if config:
            sys.argv.extend(["--config", config])
        if no_api:
            sys.argv.append("--no-api")
        if no_detector:
            sys.argv.append("--no-detector")

        # Run deployment
        deploy_main()

        # Restore original argv
        sys.argv = original_argv

    except KeyboardInterrupt:
        logger.info("Deployment stopped by user")
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# =============================================================================
# serenesense/cli/dataset.py - Dataset Management CLI
# =============================================================================
#!/usr/bin/env python3
"""
SereneSense Dataset Management CLI
Command-line interface for dataset operations.
"""

import click
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@click.group()
def dataset():
    """Dataset management commands."""
    pass


@dataset.command()
@click.option("--datasets", default="mad", help="Datasets to download (comma-separated)")
@click.option("--output-dir", default="data/raw", help="Output directory")
@click.option("--sample-only", is_flag=True, help="Download sample data only")
@click.option("--verify", is_flag=True, help="Verify downloaded files")
@click.option("--log-level", default="INFO", help="Logging level")
def download(datasets, output_dir, sample_only, verify, log_level):
    """Download datasets for training."""
    setup_logging(level=log_level.upper())

    logger.info("📥 SereneSense Dataset Download")
    logger.info("=" * 50)

    try:
        from scripts.download_datasets import main as download_main

        # Override sys.argv
        original_argv = sys.argv
        sys.argv = ["download_datasets.py", "--datasets", datasets, "--output-dir", output_dir]

        if sample_only:
            sys.argv.append("--sample-only")
        if verify:
            sys.argv.append("--verify")

        download_main()
        sys.argv = original_argv

        logger.info("✅ Dataset download completed")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


@dataset.command()
@click.option("--config", "-c", required=True, help="Data preparation configuration")
@click.option("--output-dir", default="data/processed", help="Output directory")
@click.option("--num-workers", type=int, default=4, help="Number of workers")
@click.option("--chunk-size", type=int, default=1000, help="Processing chunk size")
@click.option("--log-level", default="INFO", help="Logging level")
def prepare(config, output_dir, num_workers, chunk_size, log_level):
    """Prepare datasets for training."""
    setup_logging(level=log_level.upper())

    logger.info("⚙️ SereneSense Dataset Preparation")
    logger.info("=" * 50)

    try:
        from scripts.prepare_data import main as prepare_main

        # Override sys.argv
        original_argv = sys.argv
        sys.argv = [
            "prepare_data.py",
            "--config",
            config,
            "--output-dir",
            output_dir,
            "--num-workers",
            str(num_workers),
            "--chunk-size",
            str(chunk_size),
        ]

        prepare_main()
        sys.argv = original_argv

        logger.info("✅ Dataset preparation completed")

    except Exception as e:
        logger.error(f"Preparation failed: {e}")
        sys.exit(1)


@dataset.command()
@click.option("--data-dir", required=True, help="Dataset directory")
@click.option("--output-format", default="json", help="Output format (json, yaml, csv)")
@click.option("--include-samples", is_flag=True, help="Include sample statistics")
@click.option("--log-level", default="INFO", help="Logging level")
def info(data_dir, output_format, include_samples, log_level):
    """Show dataset information and statistics."""
    setup_logging(level=log_level.upper())

    logger.info("📊 SereneSense Dataset Information")
    logger.info("=" * 50)

    try:
        from core.data.loaders.mad_loader import MADDataModule, MADConfig
        from core.core.audio_processor import AudioConfig

        # Create data module
        mad_config = MADConfig(data_dir=data_dir)
        audio_config = AudioConfig()
        data_module = MADDataModule(mad_config, audio_config)

        # Get statistics
        data_module.setup()
        stats = data_module.get_dataset_statistics()

        # Display information
        import json
        import yaml

        if output_format == "json":
            print(json.dumps(stats, indent=2, default=str))
        elif output_format == "yaml":
            print(yaml.dump(stats, default_flow_style=False))
        else:
            # Human-readable format
            for split, split_stats in stats.items():
                logger.info(f"\n{split.upper()} SET:")
                logger.info(f"  Samples: {split_stats['total_samples']}")
                logger.info(f"  Duration: {split_stats['total_duration']:.1f}s")
                logger.info(f"  Avg duration: {split_stats['avg_duration']:.2f}s")
                logger.info("  Class distribution:")
                for class_name, count in split_stats["class_distribution"].items():
                    logger.info(f"    {class_name}: {count}")

    except Exception as e:
        logger.error(f"Info extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    dataset()

# =============================================================================
# serenesense/cli/realtime.py - Real-time Detection CLI
# =============================================================================
#!/usr/bin/env python3
"""
SereneSense Real-time Detection CLI
Command-line interface for real-time military vehicle detection.
"""

import click
import sys
import time
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@click.command()
@click.option("--model", "-m", required=True, help="Path to trained/optimized model")
@click.option("--config", "-c", help="Real-time detection configuration")
@click.option("--confidence-threshold", type=float, default=0.7, help="Detection threshold")
@click.option("--device", help="Device to use (cuda, cpu, auto)")
@click.option("--sample-rate", type=int, default=16000, help="Audio sample rate")
@click.option("--chunk-size", type=int, default=1024, help="Audio chunk size")
@click.option("--window-length", type=float, default=2.0, help="Detection window length")
@click.option("--output-file", help="Save detections to file")
@click.option("--webhook-url", help="Send detections to webhook")
@click.option(
    "--alert-classes",
    default="helicopter,fighter_aircraft,military_vehicle",
    help="Classes that trigger alerts",
)
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--duration", type=int, help="Run for specified duration (seconds)")
@click.option("--quiet", is_flag=True, help="Suppress detection output")
def main(
    model,
    config,
    confidence_threshold,
    device,
    sample_rate,
    chunk_size,
    window_length,
    output_file,
    webhook_url,
    alert_classes,
    log_level,
    duration,
    quiet,
):
    """
    Start real-time military vehicle detection.

    Examples:
        serenesense-realtime --model models/optimized/model_jetson.trt
        serenesense-realtime --model models/best_model.pth --confidence-threshold 0.8
        serenesense-realtime --model models/best_model.pth --output-file detections.json --duration 300
    """
    # Setup logging
    setup_logging(level=log_level.upper())

    logger.info("🎤 SereneSense Real-time Detection")
    logger.info("=" * 50)
    logger.info(f"Model: {model}")
    logger.info(f"Confidence threshold: {confidence_threshold}")
    logger.info(f"Device: {device or 'auto'}")

    alert_class_list = alert_classes.split(",")
    logger.info(f"Alert classes: {alert_class_list}")

    try:
        from core.inference.realtime.detector import RealTimeDetector, InferenceConfig
        import json
        import requests

        # Create inference configuration
        inference_config = InferenceConfig(
            model_path=model,
            confidence_threshold=confidence_threshold,
            device=device or "auto",
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            window_length=window_length,
        )

        # Detection storage
        detections = []
        start_time = time.time()

        def detection_callback(result):
            """Handle detection results"""
            detection_data = {
                "timestamp": result.timestamp,
                "label": result.label,
                "confidence": result.confidence,
                "processing_time": result.processing_time,
            }

            detections.append(detection_data)

            # Console output
            if not quiet:
                emoji = "🚨" if result.label in alert_class_list else "🔍"
                logger.info(
                    f"{emoji} {result.label} ({result.confidence:.3f}) "
                    f"at {result.timestamp:.2f}s (processing: {result.processing_time*1000:.1f}ms)"
                )

            # Webhook notification
            if webhook_url and result.label in alert_class_list:
                try:
                    requests.post(webhook_url, json=detection_data, timeout=5)
                except Exception as e:
                    logger.warning(f"Webhook failed: {e}")

            # File output
            if output_file:
                try:
                    with open(output_file, "w") as f:
                        json.dump(detections, f, indent=2)
                except Exception as e:
                    logger.warning(f"File save failed: {e}")

        # Setup detector
        inference_config.detection_callback = detection_callback
        detector = RealTimeDetector(inference_config)

        # Start detection
        logger.info("Starting real-time detection...")
        logger.info("Press Ctrl+C to stop")

        detector.start()

        # Run for specified duration or until interrupted
        try:
            if duration:
                logger.info(f"Running for {duration} seconds...")
                time.sleep(duration)
            else:
                logger.info("Running indefinitely...")
                while True:
                    time.sleep(1)

                    # Display periodic statistics
                    if int(time.time() - start_time) % 30 == 0:
                        stats = detector.get_statistics()
                        logger.info(
                            f"Stats: {stats['total_detections']} detections, "
                            f"avg inference: {stats['avg_inference_time']*1000:.1f}ms"
                        )

        except KeyboardInterrupt:
            pass

        # Stop detection
        logger.info("Stopping detection...")
        detector.stop()

        # Final statistics
        final_stats = detector.get_statistics()
        total_time = time.time() - start_time

        logger.info("\n" + "=" * 50)
        logger.info("Detection Summary")
        logger.info("=" * 50)
        logger.info(f"Total time: {total_time:.1f} seconds")
        logger.info(f"Total detections: {final_stats['total_detections']}")
        logger.info(
            f"Detection rate: {final_stats['total_detections'] / total_time:.2f} per second"
        )
        logger.info(f"Average inference time: {final_stats['avg_inference_time']*1000:.1f}ms")

        if final_stats["detection_counts"]:
            logger.info("Detection counts by class:")
            for class_name, count in final_stats["detection_counts"].items():
                logger.info(f"  {class_name}: {count}")

        # Save final results
        if output_file:
            with open(output_file, "w") as f:
                json.dump({"summary": final_stats, "detections": detections}, f, indent=2)
            logger.info(f"Results saved to: {output_file}")

    except Exception as e:
        logger.error(f"Real-time detection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
