import hydra
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table
import os
from loguru import logger
from pathlib import Path
import sys

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

# Custom Module Imports
from src.segmentation.train import train_model
from src.segmentation.visualize import (
    visualize_training_metrics,
    visualize_predictions,
    visualize_feature_maps,
)
from src.segmentation.data import data_loaders
from src.segmentation.model import MiniUNet

console = Console()


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Simple training and plotting pipeline"""

    console.print("[bold blue]Starting Food Segmentation Training")

    # Show config
    hp = cfg.model.hyperparameters
    console.print(
        f"Epochs: {hp.epochs}, Batch Size: {hp.batch_size}, Classes: {hp.n_classes}"
    )

    # NEW:
    os.makedirs(os.path.dirname(cfg.paths.model_path), exist_ok=True)
    os.makedirs(cfg.paths.plots_path, exist_ok=True)
    os.makedirs(cfg.paths.predictions_path, exist_ok=True)
    os.makedirs(cfg.paths.feature_path, exist_ok=True)

    # Load data
    console.print("[yellow]Loading data...")
    train_loader, test_loader = data_loaders(
        base_dir=cfg.paths.base_dir, batch_size=hp.batch_size, num_workers=4
    )

    # Train model
    console.print("[green]Training model...")
    model = MiniUNet()
    model_path = cfg.paths.model_path

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=hp.epochs,
        model_path=model_path,
    )

    # Plot training metrics
    console.print("[cyan]Generating training plots...")
    try:
        visualize_training_metrics(model_path, loss_path=cfg.paths.plots_path)
        console.print("[green]✓ Training plots saved to reports/training_plots/")
    except Exception as e:
        console.print(f"[red]Training plot error: {e}")

    # Visualize predictions
    console.print("[cyan]Generating prediction visualizations...")
    try:
        visualize_predictions(
            model=trained_model,
            test_loader=test_loader,
            num_images=5,
            segments_path=cfg.paths.predictions_path,
        )
        console.print(
            f"[green]✓ Prediction visualizations saved to {cfg.paths.predictions_path}"
        )
    except Exception as e:
        console.print(f"[red]Prediction visualization error: {e}")

    # Visualize feature maps and filters
    console.print("[cyan]Generating feature map visualizations...")
    try:
        # Get a sample image for feature map visualization
        sample_images, _ = next(iter(test_loader))
        sample_image = sample_images[0]  # Use first image from batch

        # Visualize feature maps for different layers
        for layer_name in ["encoder1", "encoder2", "encoder3"]:
            console.print(f"  Generating {layer_name} feature maps...")
            visualize_feature_maps(
                model=trained_model,
                test_loader=test_loader,
                input_image=sample_image,
                layer_name=layer_name,
                num_maps=16,
                filter_path=cfg.paths.feature_path,
            )

        console.print(
            f"[green]✓ Feature map visualizations saved to {cfg.paths.feature_path}"
        )
    except Exception as e:
        console.print(f"[red]Feature map visualization error: {e}")

    console.print("[bold green]Done!")


if __name__ == "__main__":
    main()
