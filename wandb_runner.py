# wandb_runner.py
import sys
import subprocess
import argparse
from pathlib import Path


def main():
    """Wrapper script to convert WandB arguments to Hydra overrides"""

    parser = argparse.ArgumentParser(
        description="WandB Runner for Hydra-based training"
    )
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, help="Number of epochs")

    # Parse known args to handle any additional WandB arguments
    args, unknown = parser.parse_known_args()

    # Build the Hydra command
    cmd = ["python", "src/segmentation/main.py"]

    # Convert WandB arguments to Hydra overrides
    if args.lr is not None:
        cmd.append(f"model.hyperparameters.lr={args.lr}")
    if args.batch_size is not None:
        cmd.append(f"model.hyperparameters.batch_size={args.batch_size}")
    if args.epochs is not None:
        cmd.append(f"model.hyperparameters.epochs={args.epochs}")

    # Add any unknown arguments (in case WandB passes additional params)
    cmd.extend(unknown)

    print(f"🚀 Running command: {' '.join(cmd)}")

    # Execute the actual training script
    result = subprocess.run(cmd, cwd=Path.cwd())

    # Exit with the same code as the subprocess
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
