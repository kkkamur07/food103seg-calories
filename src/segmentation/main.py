import hydra
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import os
from pathlib import Path
import sys

# Custom Module Imports
from src.segmentation.train import Trainer
from src.segmentation.data import data_loaders

console = Console()
os.environ["WANDB_SILENT"] = "true"  # Suppress WandB output
os.environ["WANDB_QUIET"] = "true"


def print_hyperparameters_table(cfg: DictConfig):
    """Display hyperparameters in a clean table format"""

    # Create hyperparameters table
    table = Table(
        title="🔧 Training Configuration", show_header=True, header_style="bold blue"
    )
    table.add_column("Parameter", style="cyan", width=20)
    table.add_column("Value", style="white", width=15)

    # Add hyperparameters
    table.add_row("Epochs", str(cfg.model.hyperparameters.epochs))
    table.add_row("Batch Size", str(cfg.model.hyperparameters.batch_size))
    table.add_row("Learning Rate", str(cfg.model.hyperparameters.lr))
    table.add_row("Base Directory", str(cfg.paths.base_dir))

    console.print(table)
    console.print()


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Simple training and visualization pipeline"""

    # Display configuration
    console.print(
        Panel.fit("🍕 Food Segmentation Training Pipeline", style="bold green")
    )
    print_hyperparameters_table(cfg)

    # Extract parameters
    epochs = cfg.model.hyperparameters.epochs
    batch_size = cfg.model.hyperparameters.batch_size
    learning_rate = cfg.model.hyperparameters.lr
    base_dir = cfg.paths.base_dir

    # Load data
    console.print("[yellow]📊 Loading data...")
    train_loader, test_loader = data_loaders(
        base_dir=base_dir, batch_size=batch_size, num_workers=4
    )
    console.print(
        f"[green]✓ Train: {len(train_loader.dataset)} samples | Test: {len(test_loader.dataset)} samples[/green]"
    )
    console.print()

    # Train model
    console.print("[blue]🚀 Starting training...[/blue]")
    trainer = Trainer(
        epochs=epochs,
        base_dir=base_dir,
        batch_size=batch_size,
        lr=learning_rate,
    )

    trainer.train()

    # Visualizations
    console.print("[magenta]📈 Generating visualizations...[/magenta]")
    trainer.visualize_training_metrics()
    trainer.visualize_predictions()

    console.print(
        Panel.fit("✅ Training Pipeline Completed Successfully!", style="bold green")
    )


def setup_project_path():
    """Setup project path for imports"""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    sys.path.insert(0, str(project_root))


if __name__ == "__main__":
    setup_project_path()
    main()
