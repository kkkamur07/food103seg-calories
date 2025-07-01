# import torch
# import pytest
# from unittest.mock import MagicMock, patch
# from src.train import FoodSegmentation

# class DummyModel(torch.nn.Module):
#     def __init__(self, n_classes=None):
#         super().__init__()
#         self.linear = torch.nn.Linear(10, n_classes if n_classes is not None else 104)
#     def forward(self, x):
#         return self.linear(x)

# @pytest.fixture
# def mock_data_loaders():
#     return MagicMock()

# @pytest.fixture
# def fake_dataloader():
#     images = torch.randn(2, 3, 224, 224)
#     masks = torch.randint(0, 104, (2, 224, 224))
#     return [(images, masks)]

# @patch("src.train.UnetPlus", new=DummyModel)
# @patch("src.train.wandb")
# def test_food_segmentation_train(mock_wandb, mock_data_loaders, fake_dataloader):

#     model = FoodSegmentation(epochs=1, batch_size=2, validation_split=0.1, n_classes=104)

#     # Override data loaders with fake ones
#     model.train_loader = fake_dataloader
#     model.val_loader = fake_dataloader
#     model.test_loader = fake_dataloader

#     model.train()

#     # Check optimizer learning rate
#     assert model.optimizer.param_groups[0]['lr'] == model.lr

#     # Check wandb.log called at least twice (train & val)
#     assert mock_wandb.log.call_count >= 2

#     # Forward pass shape
#     x = torch.randn(1, 3, 224, 224)
#     y = model.forward(x)
#     assert y.shape[1] == model.n_classes
