import hydra
from omegaconf import DictConfig
import typer
from rich.console import Console
from rich.table import Table
import os
from loguru import logger
from pathlib import Path

# Custom Module Imports
from src.train import FoodSegmentation


app = typer.Typer()
console = Console()


@hydra.main(config_path="../configs", config_name="config")
def hydra_main(cfg: DictConfig):
    """
    Main function to initialize and run the food segmentation model.
    Now uses Hydra for configuration.
    """

    console.rule("[bold blue]Starting Food Segmentation Model Training")

    table = Table(title="Training Configuration")
    table.add_column("Parameter", style="bold")
    table.add_column("Value")

    table.add_row("Number of Classes", str(cfg.n_classes))
    table.add_row("Learning Rate", str(cfg.lr))
    table.add_row("Base Directory", cfg.base_dir)
    table.add_row("Epochs", str(cfg.epochs))
    table.add_row("Batch Size", str(cfg.batch_size))
    table.add_row("Validation Split", str(cfg.validation_split))

    console.print(table)

    # Instantiate the model using Hydra's instantiate utility
    model = FoodSegmentation(
        n_classes=cfg.n_classes,
        lr=cfg.lr,
        base_dir=cfg.base_dir,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_split=cfg.validation_split,
    )

    console.print("[bold green]Model initialized. Starting training...")
    model.train()
    console.print("[bold green]Training complete!")


def main(base_dir: str = "/home/krrish/home/desktop/sensor-behaviour/"):

    # Change the working directory to base directory
    base_dir = Path(base_dir).resolve()

    os.chdir(base_dir)
    print(f"Changed working directory to {os.getcwd()}")

    logger.info(f"Changed working directory to {base_dir}")

    hydra_main()


if __name__ == "__main__":
    main()
