import torch
import torchvision.transforms as transforms
import PIL
import os
import torch
from loguru import logger

# Logging into the files
logger.add("saved/logs/model_training.log", rotation="1 day")


class FoodSegDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, mode="train", transforms=None):
        self.base_dir = base_dir
        self.mode = mode  # 'train' or 'test'

        self.img_dir = os.path.join(base_dir, "Images", "img_dir", mode)
        self.ann_dir = os.path.join(base_dir, "Images", "ann_dir", mode)

        self.transforms = transforms

        self.images = os.listdir(self.img_dir)
        logger.info(f"Found {len(self.images)} images in {self.mode} mode.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.images[idx])
        ann_name = os.path.join(
            self.ann_dir, self.images[idx].replace(".jpg", ".png")
        )  # ensure extension matches

        image = PIL.Image.open(img_name).convert("RGB")
        annotation = PIL.Image.open(ann_name).convert("L")

        if self.transforms:
            image = self.transforms(image)
            annotation = self.transforms(annotation)
            annotation = annotation.squeeze(0).long()

        return image, annotation


def data_loaders(base_dir, batch_size=32, num_workers=4, validation_split=0.2):

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Create the dataset
    train_dataset = FoodSegDataset(base_dir, mode="train", transforms=transform)
    test_dataset = FoodSegDataset(base_dir, mode="test", transforms=transform)

    # Train Validation Split
    train_size = int((1 - validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
