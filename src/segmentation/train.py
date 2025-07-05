import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
from loguru import logger
import wandb
from src.segmentation.model import MiniUNet
import src.segmentation.loss as iou
from src.segmentation.data import data_loaders

# Logging into the files
logger.add("saved/logs/model_training.log", rotation="1 day")


# class FoodSegmentation(nn.Module):
#     def __init__(
#         self,
#         n_classes=104,
#         lr=1e-4,
#         base_dir="/home/krrish/home/desktop/sensor-behaviour/",
#         epochs=10,
#         batch_size=16,
#         validation_split=0.1,
#     ):

#         super().__init__()

#         self.n_classes = n_classes
#         self.model = MiniUnet(n_classes=n_classes).to(
#             "cuda" if torch.cuda.is_available() else "cpu"
#         )

#         # self.model = MobileNetV3DeepLabV3Plus().to(
#         #     "cuda" if torch.cuda.is_available() else "cpu"
#         # )

#         self.parameters = sum(
#             p.numel() for p in self.model.parameters() if p.requires_grad
#         )

#         # Training Parameters
#         self.epochs = epochs
#         self.lr = lr
#         self.batch_size = batch_size
#         self.validation_split = validation_split

#         # Base Directory
#         self.base_dir = os.path.join(base_dir, "data")

#         # Data Loaders
#         self.train_loader, self.val_loader, self.test_loader = data_loaders(
#             base_dir=self.base_dir,
#             validation_split=self.validation_split,
#             batch_size=self.batch_size,
#         )

#         self.loss = loss_module.CombinedLoss()
#         self.optimizer = optim.Adam(
#             self.model.parameters(), lr=self.lr, weight_decay=1e-4
#         )

#         # Training History
#         self.train_losses = []
#         self.val_losses = []
#         self.val_ious = []

#         self.best_iou = 0.0

#         logger.info(f"Model initialized with {self.parameters} trainable parameters.")

#         # Initialize Weights and Biases
#         wandb.init(
#             project="food-segmentation",
#             config={
#                 "epochs": self.epochs,
#                 "learning_rate": self.lr,
#                 "batch_size": self.batch_size,
#                 "validation_split": self.validation_split,
#                 "model": "UnetPlus",
#                 "n_classes": self.n_classes,
#                 "base_dir": self.base_dir,
#                 "trainable_params": self.parameters,
#             },
#         )

#         wandb.watch(self.model, log="all")

#     def forward(self, x):
#         return self.model(x)

#     def train_step(self, image, mask):
#         self.model.train()
#         self.optimizer.zero_grad()

#         # Move data to device
#         image = image.to("cuda" if torch.cuda.is_available() else "cpu")
#         mask = mask.to("cuda" if torch.cuda.is_available() else "cpu")

#         outputs = self.forward(image)
#         loss = self.loss(outputs, mask)

#         loss.backward()
#         self.optimizer.step()

#         return loss.item()

#     def eval_step(self, image, mask):
#         self.model.eval()
#         with torch.no_grad():

#             # Move data to device
#             image = image.to("cuda" if torch.cuda.is_available() else "cpu")
#             mask = mask.to("cuda" if torch.cuda.is_available() else "cpu")

#             outputs = self.forward(image)
#             loss = self.loss(outputs, mask)

#             # Calculate IoU
#             outputs = torch.argmax(outputs, dim=1)
#             iou = loss_module.calculate_iou(outputs, mask, num_classes=self.n_classes)

#         return loss.item(), iou

#     def train(self):
#         for epoch in range(self.epochs):

#             running_loss = 0.0
#             for images, masks in self.train_loader:
#                 loss = self.train_step(images, masks)
#                 running_loss += loss

#             running_loss /= len(self.train_loader)
#             logger.info(
#                 f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {running_loss:.4f}"
#             )
#             wandb.log({"Train Loss": running_loss, "epoch": epoch + 1})
#             self.train_losses.append(running_loss)

#             val_loss = 0.0
#             all_ious = []

#             for images, masks in self.val_loader:
#                 loss, iou = self.eval_step(images, masks)
#                 val_loss += loss

#                 all_ious.extend(iou[~np.isnan(iou)])

#             val_loss /= len(self.val_loader)
#             val_iou = np.mean(all_ious) if all_ious else 0.0

#             logger.info(
#                 f"Epoch [{epoch+1}/{self.epochs}], Validation Loss: {val_loss:.4f}, Validation IoU: {val_iou:.4f}"
#             )
#             wandb.log({"Train Loss": running_loss, "epoch": epoch + 1})

#         # Save the model
#         if val_iou > self.best_iou:
#             self.best_iou = val_iou
#             torch.save(self.model.state_dict(), "best_model.pth")
#             logger.info(f"Model saved with IoU: {self.best_iou:.4f}")

#         wandb.finish()
#         logger.info("Training complete.")


def train_model(
    model, train_loader, test_loader, num_epochs=1, device=None, model_path=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Track metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False
        )
        for images, masks in train_bar:
            images = images.to(device)
            masks = masks.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Calculate accuracy
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == masks).sum().item()
            total += masks.numel()

            train_bar.set_postfix(loss=loss.item())

        # Training metrics
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_bar = tqdm(
            test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False
        )
        with torch.no_grad():
            for images, masks in val_bar:
                images = images.to(device)
                masks = masks.to(device).long()

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                # Calculate accuracy
                pred = torch.argmax(outputs, dim=1)
                val_correct += (pred == masks).sum().item()
                val_total += masks.numel()

                val_bar.set_postfix(val_loss=loss.item())

        # Validation metrics
        val_loss = val_loss / len(test_loader.dataset)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}]:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model with metrics
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_accuracies": train_accs,
                    "val_accuracies": val_accs,
                    "best_val_loss": best_val_loss,
                    "final_train_loss": train_loss,
                    "final_val_loss": val_loss,
                    "final_train_acc": train_acc,
                    "final_val_acc": val_acc,
                },
                model_path,
            )
            print(f"  Best model saved with val_loss: {val_loss:.4f}")

    print("Training complete.")
    return model
