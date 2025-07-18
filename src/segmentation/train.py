import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import os
import numpy as np
from tqdm import tqdm
from loguru import logger
import wandb
from src.segmentation.model import MiniUNet
from src.segmentation.data import data_loaders
from matplotlib import pyplot as plt
from torch.profiler import profile, ProfilerActivity, schedule

# Logging into the files
# logger.remove()
logger.add(
    "saved/logs/model_training.log",
    rotation="1 day",
    level="INFO",
    format="{time} {level} {message}",
)
logger.add(
    "saved/logs/model_training_error.log",
    rotation="1 day",
    level="WARNING",
    format="{time} {level} {message}",
)
logger.add(
    lambda msg: print(msg, end=""), level="WARNING", format="⚠️  {level}: {message}"
)


class Trainer:
    def __init__(
        self,
        lr=None,
        epochs=None,
        batch_size=None,
        base_dir=None,
        enable_profiler=None,
        init_wandb=True,
        prune_amount=0.2,
    ):
        """
        Initialize the Trainer and do experimental logging with Weights and Biases.

        Args:
            lr (float): Learning rate for the optimizer.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            base_dir (str): Project's root directory.
            enable_profiler (bool): Whether to enable PyTorch profiler.
            init_wandb (bool): Whether to initialize Weights and Biases.
        """

        super().__init__()

        # model
        self.model = MiniUNet().to("cuda" if torch.cuda.is_available() else "cpu")

        # Check if pruning is enabled
        if prune_amount > 0.0:
            logger.info(f"Applying pruning with amount: {prune_amount}")

            # Apply global unstructured pruning to Conv2d and Linear layers
            parameters_to_prune = []
            for module in self.model.modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    parameters_to_prune.append((module, "weight"))

            if parameters_to_prune:
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=prune_amount,
                )
                logger.info("Pruning applied to the model.")
            else:
                logger.warning("No Conv2d or Linear layers found for pruning.")

        # Model Parameters
        self.parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        # Training Parameters
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        # Directories
        self.base_dir = base_dir
        self.saved_dir = os.path.join(self.base_dir, "saved")
        self.model_path = os.path.join(self.saved_dir, "models", "model.pth")
        self.plots_path = os.path.join(
            self.saved_dir, "reports", "training_metrics.png"
        )
        self.predictions = os.path.join(
            self.saved_dir, "predictions", "predictions.png"
        )
        # Data Loaders
        self.train_loader, self.test_loader = data_loaders(
            base_dir=self.base_dir,
            batch_size=self.batch_size,
        )

        # Loss function
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Training History
        self.train_losses = []
        self.test_losses = []
        self.train_accs = []
        self.test_accs = []
        self.train_ious = []
        self.test_ious = []
        self.best_test_loss = float("inf")

        logger.info(f"Model initialized with {self.parameters} trainable parameters.")

        self.init_wandb = init_wandb
        self.enable_profiler = enable_profiler
        if not self.enable_profiler:
            logger.info("Profiler is disabled. Training will run without profiling.")

        # Initialize Weights and Biases
        if self.init_wandb and wandb.run is None:
            wandb.init(
                project="Food-Segmentation",
                config={
                    "epochs": self.epochs,
                    "learning_rate": self.lr,
                    "batch_size": self.batch_size,
                    "base_dir": self.base_dir,
                    "trainable_params": self.parameters,
                    "model": "MiniUNet",
                    "optimizer": "Adam",
                    "loss_function": "CrossEntropyLoss",
                },
            )
            wandb.watch(self.model, log="all")
            logger.info("Weights and Biases initialized for tracking.")

    def remove_pruning(self):
        """
        Remove pruning from the model.

        This method iterates through all modules in the model and removes
        pruning parameters if they exist. It is useful for restoring the
        original model state after pruning has been applied.
        """

        for module in self.model.modules():
            if hasattr(module, "weight_orig"):
                prune.remove(module, "weight")
            if hasattr(module, "bias_orig"):
                prune.remove(module, "bias")
                logger.info(f"Removed pruning from bias of {module}")

        logger.info("Pruning removed from the model.")

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def train(self):
        """

        Execute the training loop with optional profiling and Weights and Biases logging.

        Pipeline Steps :
        1. Initialize the device for training (GPU or CPU).
        2. Check if the model path is set.
        3. Create profiler if enabled.
        4. Loop through the number of epochs:
        5. Train the model on the training dataset.
        6. Calculate and log training metrics (loss, accuracy, IoU).
        7. Validate the model on the test dataset.
        8. Save the best model based on test loss.
        9. Log metrics to Weights and Biases.
        10. Finish Weights and Biases run.

        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.model_path:
            logger.warning("Model path is not set. Cannot save the model.")
            return

        os.makedirs("./profiler_logs", exist_ok=True)

        prof = None

        if self.enable_profiler:
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    "./profiler_logs"
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            prof.start()

        for epoch in range(self.epochs):
            # Training
            self.model.train()

            running_loss = 0.0
            running_accuracy = 0.0
            running_iou = 0.0

            train_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{self.epochs} [Train]",
                leave=False,
            )

            for batch_idx, (images, masks) in enumerate(train_bar):
                images = images.to(device)
                masks = masks.to(device).long()

                self.optimizer.zero_grad()
                outputs = self.forward(images)
                loss = self.loss(outputs, masks)
                loss.backward()
                self.optimizer.step()

                # Calculate accuracy for current batch
                pred_classes = torch.argmax(outputs, dim=1)
                train_accuracy = torch.mean((pred_classes == masks).float())
                train_iou = self.calculate_iou(pred_classes, masks)

                # Accumulate loss and accuracy
                running_accuracy += train_accuracy.item() * images.size(0)
                running_loss += loss.item() * images.size(0)
                running_iou += train_iou * images.size(0)

                # Update progress bar
                train_bar.set_postfix(loss=loss.item())

                # Step profiler and limit batches for profiling
                if self.enable_profiler and prof is not None:
                    prof.step()
                    logger.info(f"Profiler step {batch_idx + 1}")
                    if batch_idx >= 20:
                        logger.info("🔥 Profiling completed")
                        prof.stop()
                        break

            # Training metrics
            train_loss = running_loss / len(self.train_loader.dataset)
            train_acc = running_accuracy / len(self.train_loader.dataset)
            train_iou = running_iou / len(self.train_loader.dataset)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.train_ious.append(train_iou)

            # Your existing validation code...
            self.model.eval()
            running_loss = 0.0
            running_accuracy = 0.0
            running_iou = 0.0

            test_bar = tqdm(
                self.test_loader,
                desc=f"Epoch {epoch+1}/{self.epochs} [Test]",
                leave=False,
            )

            with torch.no_grad():
                for images, masks in test_bar:
                    images = images.to(device)
                    masks = masks.to(device).long()

                    outputs = self.forward(images)
                    loss = self.loss(outputs, masks)

                    pred_classes = torch.argmax(outputs, dim=1)
                    test_accuracy = torch.mean((pred_classes == masks).float())
                    test_iou = self.calculate_iou(pred_classes, masks)

                    running_accuracy += test_accuracy.item() * images.size(0)
                    running_loss += loss.item() * images.size(0)
                    running_iou += test_iou * images.size(0)

                    test_bar.set_postfix(test_loss=loss.item())

            test_loss = running_loss / len(self.test_loader.dataset)
            test_acc = running_accuracy / len(self.test_loader.dataset)
            test_iou = running_iou / len(self.test_loader.dataset)

            self.test_losses.append(test_loss)
            self.test_accs.append(test_acc)
            self.test_ious.append(test_iou)

            print(f"Epoch [{epoch+1}/{self.epochs}]:")
            print(
                f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train IoU: {train_iou:.4f}"
            )
            print(
                f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test IoU: {test_iou:.4f}"
            )

            wandb.log(
                {
                    "Train Loss": train_loss,
                    "Train Accuracy": train_acc,
                    "Test Loss": test_loss,
                    "Test Accuracy": test_acc,
                    "epoch": epoch + 1,
                    "Train IoU": train_iou,
                    "Test IoU": test_iou,
                }
            )

            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                self.remove_pruning()
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch + 1,
                        "best_test_loss": self.best_test_loss,
                        "final_train_loss": train_loss,
                        "final_test_loss": test_loss,
                        "final_train_acc": train_acc,
                        "final_test_acc": test_acc,
                    },
                    self.model_path,
                )
                logger.info(f"Best model saved with test_loss: {test_loss:.4f}")

                artifact = wandb.Artifact(
                    "model",
                    type="model",
                    description="Best model based on test loss",
                )
                artifact.add_file(self.model_path)
                wandb.log_artifact(artifact)

            logger.info(f"Training complete. Model saved at {self.model_path}")
            print("Training complete.")

        wandb.finish()

    def visualize_training_metrics(self):
        """
        Visualize training and testing loss & accuracy from saved model checkpoint
        Creates two side-by-side graphs: Loss comparison and Accuracy comparison

        Args:
            base_dir: Base directory for saving plots
            model_path: Path to the saved model checkpoint (.pth file)
            plots_path: Directory to save the plots
        """

        # Extract metrics
        train_losses = self.train_losses
        test_losses = self.test_losses
        train_accs = self.train_accs
        test_accs = self.test_accs
        train_ious = self.train_ious
        test_ious = self.test_ious

        # Check if data exists
        if not train_losses or not test_losses:
            logger.warning("No loss data found in checkpoint!")
            return

        if not train_accs or not test_accs:
            logger.warning("No accuracy data found in checkpoint!")
            return

        if not self.plots_path:
            logger.warning("No plots path provided. Plot not saved.")
            return

        # Create figure with 3 subplots side by side
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        epochs = range(1, len(train_losses) + 1)

        # Graph 1: Training vs Testing Loss
        ax1.plot(
            epochs,
            train_losses,
            label="Training Loss",
            color="blue",
            marker="o",
            linewidth=2,
        )
        ax1.plot(
            epochs,
            test_losses,
            label="Testing Loss",
            color="red",
            marker="s",
            linewidth=2,
        )
        ax1.set_title("Loss Comparison")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # Graph 2: Training vs Testing Accuracy
        ax2.plot(
            epochs,
            train_accs,
            label="Training Accuracy",
            color="green",
            marker="o",
            linewidth=2,
        )
        ax2.plot(
            epochs,
            test_accs,
            label="Testing Accuracy",
            color="orange",
            marker="s",
            linewidth=2,
        )
        ax2.set_title("Accuracy Comparison")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)

        # Graph 3: Training vs Testing IoU
        ax3.plot(
            epochs,
            train_ious,
            label="Training IoU",
            color="purple",
            marker="o",
            linewidth=2,
        )
        ax3.plot(
            epochs,
            test_ious,
            label="Testing IoU",
            color="brown",
            marker="s",
            linewidth=2,
        )
        ax3.set_title("IoU Comparison")
        ax3.set_xlabel("Epochs")
        ax3.set_ylabel("IoU")
        ax3.legend()
        ax3.grid(True)

        # Adjust layout and save the plot
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Adjust top to make space for the title
        plt.suptitle("Training and Testing Metrics", fontsize=20)
        plt.savefig(self.plots_path, dpi=300)  # Save the plot with high resolution
        logger.info(f"Training metrics plot saved: {self.plots_path}")
        plt.tight_layout()
        plt.show()  # Show the plot in interactive mode
        plt.close()  # Close the plot to free memory

    def visualize_predictions(self, num_images=5):
        device = torch.device("cpu")  # Quantized models run on CPU

        # Load saved model checkpoint
        if not os.path.exists(self.model_path):
            logger.warning(f"Model checkpoint not found at {self.model_path}")
            return

        checkpoint = torch.load(self.model_path, map_location=device)

        # Create fresh model and load weights
        quantized_model = MiniUNet()
        quantized_model.load_state_dict(checkpoint["model_state_dict"])
        quantized_model.eval()

        # Apply dynamic quantization on Conv layers (for example)
        quantized_model = torch.quantization.quantize_dynamic(
            quantized_model, {nn.Conv2d, nn.ConvTranspose2d}, dtype=torch.qint8
        )

        images_to_plot = []
        masks_to_plot = []
        preds_to_plot = []

        with torch.no_grad():
            for i, (images, masks) in enumerate(self.test_loader):
                if i >= num_images:
                    break

                images = images.to(device)
                masks = masks.to(device).long()

                outputs = quantized_model(images)
                pred_classes = torch.argmax(outputs, dim=1)

                image = images[0].cpu()
                if image.max() > 1.0:
                    image = image / 255.0

                images_to_plot.append(image.permute(1, 2, 0).numpy())
                masks_to_plot.append(masks[0].cpu().numpy())
                preds_to_plot.append(pred_classes[0].cpu().numpy())

        _, axes = plt.subplots(3, num_images, figsize=(5 * num_images, 15))

        for i in range(num_images):
            axes[0, i].imshow(np.clip(images_to_plot[i], 0, 1))
            axes[0, i].set_title(f"Input Image {i+1}")
            axes[0, i].axis("off")

            axes[1, i].imshow(masks_to_plot[i], cmap="viridis", vmin=0, vmax=103)
            axes[1, i].set_title(f"Ground Truth {i+1}")
            axes[1, i].axis("off")

            axes[2, i].imshow(preds_to_plot[i], cmap="viridis", vmin=0, vmax=103)
            axes[2, i].set_title(f"Prediction {i+1}")
            axes[2, i].axis("off")

        plt.tight_layout()
        plt.savefig(self.predictions, dpi=300, bbox_inches="tight")
        logger.info(f"Prediction grid saved: {self.predictions}")
        plt.show()
        plt.close()

    def calculate_iou(self, pred_mask, true_mask, num_classes=104):
        """Implements the Intersection over Union (IoU) metric for segmentation tasks."""
        ious = []
        pred_mask = pred_mask.view(-1)
        true_mask = true_mask.view(-1)

        for cls in range(num_classes):
            pred_inds = pred_mask == cls
            target_inds = true_mask == cls

            intersection = (pred_inds & target_inds).long().sum().item()
            union = (pred_inds | target_inds).long().sum().item()

            if union == 0:
                continue  # Skip if no pixels for this class

            iou = intersection / union
            ious.append(iou)

        return np.mean(ious) if ious else 0.0
