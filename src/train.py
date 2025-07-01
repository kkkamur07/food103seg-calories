import torch
import torch.nn as nn
import torch.optim as optim
from src.data import data_loaders
from loguru import logger
import wandb
from src.model import UnetPlus

# Logging into the files
logger.add("saved/logs/model_training.log", rotation="1 day")


class FoodSegmentation(nn.Module):
    def __init__(
        self,
        n_classes=104,
        lr=1e-4,
        base_dir="/home/krrish/home/desktop/sensor-behaviour/data",
        epochs=10,
        batch_size=16,
        validation_split=0.1,
    ):

        super().__init__()

        self.n_classes = n_classes
        self.model = UnetPlus(n_classes=n_classes).to(
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
        self.base_dir = base_dir

        # Data Loaders
        self.train_loader, self.val_loader, self.test_loader = data_loaders(
            base_dir=self.base_dir,
            validation_split=self.validation_split,
            batch_size=self.batch_size,
        )

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        logger.info(f"Model initialized with {self.parameters} trainable parameters.")

        # Initialize Weights and Biases
        wandb.init(
            project="food-segmentation",
            config={
                "epochs": self.epochs,
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "validation_split": self.validation_split,
                "model": "UnetPlus",
                "n_classes": self.n_classes,
                "base_dir": self.base_dir,
                "trainable_params": self.parameters,
            },
        )

        wandb.watch(self.model, log="all")

    def forward(self, x):
        return self.model(x)

    def train_step(self, image, mask):
        self.model.train()
        self.optimizer.zero_grad()

        # Move data to device
        image = image.to("cuda" if torch.cuda.is_available() else "cpu")
        mask = mask.to("cuda" if torch.cuda.is_available() else "cpu")

        outputs = self.forward(image)
        loss = self.loss(outputs, mask)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_step(self, image, mask):
        self.model.eval()
        with torch.no_grad():

            # Move data to device
            image = image.to("cuda" if torch.cuda.is_available() else "cpu")
            mask = mask.to("cuda" if torch.cuda.is_available() else "cpu")

            outputs = self.forward(image)
            loss = self.loss(outputs, mask)

        return loss.item()

    def train(self):
        for epoch in range(self.epochs):

            train_loss = 0.0
            for images, masks in self.train_loader:
                loss = self.train_step(images, masks)
                train_loss += loss

            train_loss /= len(self.train_loader)
            logger.info(
                f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {train_loss:.4f}"
            )
            wandb.log({"Train Loss": train_loss, "epoch": epoch + 1})

            val_loss = 0.0
            for images, masks in self.val_loader:
                loss = self.eval_step(images, masks)
                val_loss += loss

            val_loss /= len(self.val_loader)
            logger.info(
                f"Epoch [{epoch+1}/{self.epochs}], Validation Loss: {val_loss:.4f}"
            )
            wandb.log({"Train Loss": train_loss, "epoch": epoch + 1})

        wandb.finish()
        logger.info("Training complete.")
