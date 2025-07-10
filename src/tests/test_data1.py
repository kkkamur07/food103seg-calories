import pytest
import torch
import torchvision.transforms as transforms
import os
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from PIL import Image

from src.segmentation.data import FoodSegDataset,data_loaders

@pytest.fixture()
#Set up temporary directory with dummy image and annotation files.
def dummy_data(tmp_path):
    base_dir=tmp_path/"dummy_data"
    img_root_dir=base_dir/"data"/"Images"/"img_dir"
    ann_root_dir=base_dir/"data"/"Images"/"ann_dir"
    for split in ["train","test"]:
        (img_root_dir/split).mkdir(parents=True)
        (ann_root_dir/split).mkdir(parents=True)

        for i in range(2):
            rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            mask = np.random.randint(0, 104, (256, 256), dtype=np.uint8)

            Image.fromarray(rgb).save(img_root_dir / split / f"{i}.jpg")
            Image.fromarray(mask).save(ann_root_dir / split / f"{i}.png")

    return str(base_dir)

# To check if FoodSegDataset reflects correct number of dummy files created and output types 
def test_len_matches_filecount(dummy_data):
    ann_transform=transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.squeeze().long()),
            transforms.Lambda(
                lambda x: torch.where(x>=104,torch.tensor(0,dtype=torch.long),x)
            ),
        ]
    )
    img_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # converts PIL to float tensor and scales to [0, 1]
        ]
    )

#Create a dataset instance to list the actual dummy files
    train_dataset=FoodSegDataset(
        base_dir=dummy_data,
        mode="train",
        transforms=img_transform,
        ann_transform=ann_transform
    )

    #test if __len__ reflects number of dummy files=2
    assert len(train_dataset)==2
    
    #test if __getitem__ 
    image,mask=train_dataset[0]

    #assert output types and shapes after actual transforms
    assert isinstance(image,torch.Tensor)
    assert image.shape==(3,224,224)
    assert isinstance(mask,torch.Tensor)
    assert mask.shape==(224,224)
    assert mask.dtype==torch.long

    #Assert labels are within expected range
    assert torch.all(mask>=0) and torch.all(mask<=103)


def test_data_loaders_batch(dummy_data):
    train_loader, test_loader = data_loaders(dummy_data, batch_size=2, num_workers=0)
   
   
    #Checks setup

    # check if returned objects are instances of the actual classes
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)

    # Assert that the datasets within the loaders are instances of FoodSegDataset
    assert isinstance(train_loader.dataset, FoodSegDataset)
    assert isinstance(test_loader.dataset, FoodSegDataset)

    # Assert  shuffle settings
    assert train_loader.batch_size == 2
    assert train_loader.num_workers == 0

    assert test_loader.batch_size == 2
    assert test_loader.num_workers == 0

    #Checks data flow

    # Assert dataset sizes are correct based on the dummy data
    assert len(train_loader.dataset) == 2
    assert len(test_loader.dataset) == 2 
    # Check if loader yield one batch of two samples
    x_train, y_train = next(iter(train_loader))
    x_test,  y_test  = next(iter(test_loader))

    for x, y in [(x_train, y_train), (x_test, y_test)]:
        assert x.shape == (2, 3, 224, 224)
        assert y.shape == (2, 224, 224)
        assert x.dtype == torch.float32
        assert y.dtype == torch.long

