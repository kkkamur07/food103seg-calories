# src/models/model.py
import pytest
import torch
import torchvision.transforms as transforms
import os
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from PIL import Image

from src.segmentation.data import FoodSegDataset, data_loaders


@pytest.fixture()
# Dummy data
def dummy_data(tmp_path):
    base_dir = tmp_path / "dummy_data"
    img_root_dir = base_dir / "data" / "Images" / "img_dir"
    ann_root_dir = base_dir / "data" / "Images" / "ann_dir"
    for split in ["train", "test"]:
        (img_root_dir / split).mkdir(parents=True)
        (ann_root_dir / split).mkdir(parents=True)

        for i in range(2):
            rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            mask = np.random.randint(0, 104, (256, 256), dtype=np.uint8)

            Image.fromarray(rgb).save(img_root_dir / split / f"{i}.jpg")
            Image.fromarray(mask).save(ann_root_dir / split / f"{i}.png")

    return str(base_dir)


def test_len_matches_filecount(dummy_data):
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
            transforms.ToTensor(),
        ]
    )

    # Dataset Instance
    train_dataset = FoodSegDataset(
        base_dir=dummy_data,
        mode="train",
        transforms=img_transform,
        ann_transform=ann_transform,
    )

    assert len(train_dataset) == 2
    image, mask = train_dataset[0]

    assert isinstance(image, torch.Tensor)

    assert image.shape == (3, 224, 224)
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (224, 224)
    assert mask.dtype == torch.long

    assert torch.all(mask >= 0) and torch.all(mask <= 103)


def test_data_loaders_batch(dummy_data):
    train_loader, test_loader = data_loaders(dummy_data, batch_size=2, num_workers=0)

    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)

    assert isinstance(train_loader.dataset, FoodSegDataset)
    assert isinstance(test_loader.dataset, FoodSegDataset)

    assert train_loader.batch_size == 2
    assert train_loader.num_workers == 0

    assert test_loader.batch_size == 2
    assert test_loader.num_workers == 0

    assert len(train_loader.dataset) == 2
    assert len(test_loader.dataset) == 2

    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))

    for x, y in [(x_train, y_train), (x_test, y_test)]:
        assert x.shape == (2, 3, 224, 224)
        assert y.shape == (2, 224, 224)
        assert x.dtype == torch.float32
        assert y.dtype == torch.long
