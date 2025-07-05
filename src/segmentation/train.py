import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
from loguru import logger
import wandb
from src.segmentation.model import MiniUNet
import src.segmentation.loss as loss_module
from src.segmentation.data import data_loaders

# Logging into the files
logger.add("saved/logs/model_training.log", rotation="1 day")


class FoodSegmentation(nn.Module):
    def __init__(
        self,
        n_classes=104,
        lr=1e-4,
        base_dir="/home/krrish/home/desktop/sensor-behaviour/",
        epochs=10,
        batch_size=16,
        validation_split=0.1,
    ):

        super().__init__()

        self.n_classes = n_classes
        self.model = MiniUNet(n_classes=n_classes).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        # Training Parameters
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.validation_split = validation_split

        # Base Directory
        self.base_dir = os.path.join(base_dir, "data")

        # Data Loaders
        self.train_loader, self.val_loader, self.test_loader = data_loaders(
            base_dir=self.base_dir,
            validation_split=self.validation_split,
            batch_size=self.batch_size,
        )

        self.loss = loss_module.CombinedLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4
        )

        # Training History
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.val_ious = []
        self.best_val_loss = float("inf")
        self.best_iou = 0.0

        logger.info(f"Model initialized with {self.parameters} trainable parameters.")

        # Initialize Weights and Biases
        wandb.init(
            project="food-segmentation",
            config={
                "epochs": self.epochs,
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "validation_split": self.validation_split,
                "model": "MiniUNet",
                "n_classes": self.n_classes,
                "base_dir": self.base_dir,
                "trainable_params": self.parameters,
            },
        )

        wandb.watch(self.model, log="all")

    def forward(self, x):
        return self.model(x)

    def train(self, model_path="saved/model/best_model.pth"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            train_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{self.epochs} [Train]",
                leave=False,
            )

            for images, masks in train_bar:
                images = images.to(device)
                masks = masks.to(device).long()

                self.optimizer.zero_grad()
                outputs = self.forward(images)
                loss = self.loss(outputs, masks)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)

                # Calculate accuracy
                pred = torch.argmax(outputs, dim=1)
                correct += (pred == masks).sum().item()
                total += masks.numel()

                train_bar.set_postfix(loss=loss.item())

            # Training metrics
            train_loss = running_loss / len(self.train_loader.dataset)
            train_acc = correct / total
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_ious = []

            val_bar = tqdm(
                self.val_loader,
                desc=f"Epoch {epoch+1}/{self.epochs} [Val]",
                leave=False,
            )

            with torch.no_grad():
                for images, masks in val_bar:
                    images = images.to(device)
                    masks = masks.to(device).long()

                    outputs = self.forward(images)
                    loss = self.loss(outputs, masks)
                    val_loss += loss.item() * images.size(0)

                    # Calculate accuracy
                    pred = torch.argmax(outputs, dim=1)
                    val_correct += (pred == masks).sum().item()
                    val_total += masks.numel()

                    # Calculate IoU
                    iou = loss_module.calculate_iou(
                        pred, masks, num_classes=self.n_classes
                    )
                    all_ious.extend(iou[~np.isnan(iou)])

                    val_bar.set_postfix(val_loss=loss.item())

            # Validation metrics
            val_loss = val_loss / len(self.val_loader.dataset)
            val_acc = val_correct / val_total
            val_iou = np.mean(all_ious) if all_ious else 0.0

            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            self.val_ious.append(val_iou)

            print(f"Epoch [{epoch+1}/{self.epochs}]:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(
                f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val IoU: {val_iou:.4f}"
            )

            # Log to wandb
            wandb.log(
                {
                    "Train Loss": train_loss,
                    "Train Accuracy": train_acc,
                    "Val Loss": val_loss,
                    "Val Accuracy": val_acc,
                    "Val IoU": val_iou,
                    "epoch": epoch + 1,
                }
            )

            # Save best model with all metrics
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_iou = val_iou

                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(model_path), exist_ok=True)

                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch + 1,
                        "train_losses": self.train_losses,
                        "val_losses": self.val_losses,
                        "train_accuracies": self.train_accs,
                        "val_accuracies": self.val_accs,
                        "val_ious": self.val_ious,
                        "best_val_loss": self.best_val_loss,
                        "best_iou": self.best_iou,
                        "final_train_loss": train_loss,
                        "final_val_loss": val_loss,
                        "final_train_acc": train_acc,
                        "final_val_acc": val_acc,
                        "final_val_iou": val_iou,
                        "n_classes": self.n_classes,
                    },
                    model_path,
                )
                logger.info(
                    f"Best model saved with val_loss: {val_loss:.4f}, val_iou: {val_iou:.4f}"
                )
                print(
                    f"  Best model saved with val_loss: {val_loss:.4f}, val_iou: {val_iou:.4f}"
                )

        wandb.finish()
        logger.info("Training complete.")
        print("Training complete.")
        return self.model
