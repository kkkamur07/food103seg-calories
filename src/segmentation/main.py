import hydra
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import os
import sys
from pathlib import Path

# Get absolute path to configs
project_root = Path(__file__).parent.parent.parent
config_dir = str(project_root / "configs")
sys.path.insert(0, str(project_root))

# Custom Module Imports
from src.segmentation.train import Trainer
from src.segmentation.data import data_loaders

console = Console()
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_QUIET"] = "true"
os.environ["WANDB_CONSOLE"] = "off"


def print_hyperparameters_table(cfg: DictConfig):
    """
    Display hyperparameters in a clean table format using Rich.

    Creates a formatted table showing key training configuration parameters
    including epochs, batch size, learning rate, and base directory path.

    Args:
        cfg (DictConfig): Hydra configuration object containing model hyperparameters
            and path settings. Expected to have structure:
            - cfg.model.hyperparameters.epochs
            - cfg.model.hyperparameters.batch_size
            - cfg.model.hyperparameters.lr
            - cfg.paths.base_dir

    Example:
        >>> print_hyperparameters_table(cfg)
        # Displays a formatted table with training configuration
    """
    table = Table(
        title="🔧 Training Configuration", show_header=True, header_style="bold blue"
    )
    table.add_column("Parameter", style="cyan", width=20)
    table.add_column("Value", style="white", width=15)

    table.add_row("Epochs", str(cfg.model.hyperparameters.epochs))
    table.add_row("Batch Size", str(cfg.model.hyperparameters.batch_size))
    table.add_row("Learning Rate", str(cfg.model.hyperparameters.lr))
    table.add_row("Base Directory", str(cfg.paths.base_dir))

    console.print(table)
    console.print()


@hydra.main(version_base=None, config_path=config_dir, config_name="config")
def main(cfg: DictConfig):
    """
    Main training pipeline for food segmentation model.

    Orchestrates the complete training workflow including data loading,
    model training, and visualization generation. Uses Hydra for configuration
    management and Rich for enhanced console output.

    Args:
        cfg (DictConfig): Hydra configuration object containing all training
            parameters, paths, and settings. Expected structure:
            - cfg.model.hyperparameters: Training hyperparameters
            - cfg.paths.base_dir: Base directory for data and outputs
            - cfg.profiling.enabled: Whether to enable performance profiling

    Pipeline Steps:
        1. Load and display configuration parameters
        2. Initialize data loaders for training and testing
        3. Create and configure trainer instance
        4. Execute training loop
        5. Generate training metrics visualizations
        6. Generate prediction visualizations

    Example:
        Run with default config:
        >>> python main.py

        Run with custom parameters:
        >>> python main.py model.hyperparameters.epochs=50 model.hyperparameters.lr=0.001
    """
    epochs = cfg.model.hyperparameters.epochs
    batch_size = cfg.model.hyperparameters.batch_size
    learning_rate = cfg.model.hyperparameters.lr
    base_dir = cfg.paths.base_dir
    profiling_enabled = cfg.profiling.enabled
    print_hyperparameters_table(cfg)

    console.print(
        Panel.fit("🍕 Food Segmentation Training Pipeline", style="bold green")
    )
    console.print()

    console.print("[yellow]📊 Loading data...")
    train_loader, test_loader = data_loaders(
        base_dir=base_dir, batch_size=batch_size, num_workers=4
    )
    console.print(
        f"[green]✓ Train: {len(train_loader.dataset)} samples | Test: {len(test_loader.dataset)} samples[/green]"
    )
    console.print()

    console.print("[blue]🚀 Starting training...[/blue]")
    trainer = Trainer(
        epochs=epochs,
        base_dir=base_dir,
        batch_size=batch_size,
        lr=learning_rate,
        enable_profiler=profiling_enabled,
        init_wandb=True,
    )

    trainer.train()

    console.print("[magenta]📈 Generating visualizations...[/magenta]")
    trainer.visualize_training_metrics()
    trainer.visualize_predictions()

    console.print(
        Panel.fit("✅ Training Pipeline Completed Successfully!", style="bold green")
    )


if __name__ == "__main__":
    main()
