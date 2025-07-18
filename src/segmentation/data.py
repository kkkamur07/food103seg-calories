import torch
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset


class FoodSegDataset(Dataset):
    """
    PyTorch Dataset for food image segmentation.

    Loads images and corresponding annotation (mask) files for segmentation tasks.

    Attributes:
        base_dir (str): Root project directory.
        mode (str): "train" or "test". Used to pick subfolders.
        transforms (callable): Image transformation pipeline.
        ann_transform (callable): Annotation/mask transformation pipeline.
        images (list): List of image filenames.
        annotations (list): List of annotation filenames.
    """

    def __init__(self, base_dir, mode="train", transforms=None, ann_transform=None):
        """
        Initialize the FoodSegDataset.

        Args:
            base_dir (str): Project's root directory.
            mode (str): Either "train" or "test".
            transforms (callable, optional): Transformations for input images.
            ann_transform (callable, optional): Transformations for annotation masks.
        """
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "data")
        self.mode = mode  # 'train' or 'test'

        self.img_dir = os.path.join(self.data_dir, "Images", "img_dir", mode)
        self.ann_dir = os.path.join(self.data_dir, "Images", "ann_dir", mode)

        self.transforms = transforms
        self.ann_transform = ann_transform
        self.images = os.listdir(self.img_dir)
        self.annotations = os.listdir(self.ann_dir)

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of images (samples).
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Fetch image and annotation mask by index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: (image, mask) - Transformed image and segmentation mask tensors.
        """
        img_name = os.path.join(self.img_dir, self.images[idx])
        ann_name = os.path.join(self.ann_dir, self.images[idx].replace(".jpg", ".png"))

        # Load image
        image = Image.open(img_name).convert("RGB")
        if self.transforms:
            image = self.transforms(image)

        # Load mask
        mask = Image.open(ann_name).convert("L")
        if self.ann_transform:
            mask = self.ann_transform(mask)

        return image, mask


def data_loaders(base_dir, batch_size, num_workers=4):
    """
    Prepare PyTorch DataLoaders for training and testing.

    Creates datasets with appropriate transformations and returns DataLoaders
    for both training and testing phases of the segmentation model.

    Args:
        base_dir (str): Root project directory.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple: (train_loader, test_loader) - DataLoaders for training and testing.
    """

    ann_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.squeeze().long()),
            transforms.Lambda(
                lambda x: torch.where(x >= 104, torch.tensor(0, dtype=torch.long), x)
            ),
        ]
    )

    img_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # converts PIL to float tensor and scales to [0, 1]
        ]
    )

    # Create the dataset
    train_dataset = FoodSegDataset(
        base_dir, mode="train", transforms=img_transform, ann_transform=ann_transform
    )
    test_dataset = FoodSegDataset(
        base_dir, mode="test", transforms=img_transform, ann_transform=ann_transform
    )

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_loader, test_loader
