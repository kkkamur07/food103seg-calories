import hydra
from omegaconf import DictConfig
import typer
from rich.console import Console
from rich.table import Table
import os
from loguru import logger
from dotenv import load_dotenv
import wandb

# Custom Module Imports
from src.train import FoodSegmentation


app = typer.Typer()
console = Console()


@hydra.main(config_path="../conf", config_name="config")
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


@app.command()
def main(
    base_dir: str = typer.Option(
        "/home/krrish/home/desktop/sensor-behaviour/data",
        "--base-dir",
        "-b",
        help="Base directory for the dataset.",
    )
):

    # Change the working directory to base directory
    os.environ["BASE_DIR"] = base_dir
    os.chdir(base_dir)

    load_dotenv()  # loads .env
    wandb_key = os.getenv("WANDB_API_KEY")

    if wandb_key:
        wandb.login(key=wandb_key)
    else:
        logger.warning("WANDB_API_KEY not found in .env file. WandB will not be used.")

    logger.info(f"Changed working directory to {base_dir}")


if __name__ == "__main__":
    main()
