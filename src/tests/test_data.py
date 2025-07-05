import shutil
import torch
from PIL import Image
from pathlib import Path
import pytest
from torchvision import transforms

from src.data import FoodSegDataset, data_loaders


def create_dummy_data(base_dir, mode="train", num=5):
    img_dir = Path(base_dir) / "Images" / "img_dir" / mode
    ann_dir = Path(base_dir) / "Images" / "ann_dir" / mode
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num):
        img = Image.new("RGB", (256, 256), color=(i * 20, i * 20, i * 20))
        ann = Image.new("L", (256, 256), color=i)

        img.save(img_dir / f"{i}.jpg")
        ann.save(ann_dir / f"{i}.png")


@pytest.fixture
def dummy_data_dir(tmp_path):
    create_dummy_data(tmp_path, mode="train", num=5)
    create_dummy_data(tmp_path, mode="test", num=2)

    yield tmp_path

    # Teardown after tests: remove dummy data folder
    shutil.rmtree(tmp_path)


def test_dataset_length(dummy_data_dir):
    dataset = FoodSegDataset(str(dummy_data_dir), mode="train")
    assert len(dataset) == 5


def test_dataset_output_shape(dummy_data_dir):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataset = FoodSegDataset(str(dummy_data_dir), mode="train", transforms=transform)
    image, mask = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert mask.shape == (224, 224)


def test_data_loader_splits(dummy_data_dir):
    train_loader, val_loader, test_loader = data_loaders(
        str(dummy_data_dir), batch_size=2, validation_split=0.2
    )

    # train size = 5 * 0.8 = 4, val size = 1, test size = 2
    assert len(train_loader.dataset) == 4
    assert len(val_loader.dataset) == 1
    assert len(test_loader.dataset) == 2


def test_empty_dataset(tmp_path):
    (tmp_path / "Images" / "img_dir" / "train").mkdir(parents=True)
    (tmp_path / "Images" / "ann_dir" / "train").mkdir(parents=True)

    dataset = FoodSegDataset(str(tmp_path), mode="train")
    assert len(dataset) == 0


def test_transforms_applied(dummy_data_dir):
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )
    dataset = FoodSegDataset(str(dummy_data_dir), mode="train", transforms=transform)

    img, ann = dataset[0]
    assert img.shape == (3, 128, 128)
    assert ann.shape == (128, 128)
