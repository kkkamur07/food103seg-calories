import pytest
import torch
import torch.nn as nn
from src.segmentation.model import MiniUNet
from unittest.mock import patch


@pytest.fixture
def mini_unet_instance():
    return MiniUNet()


def test_mini_unet_initialization(mini_unet_instance):
    assert isinstance(mini_unet_instance.encoder1, nn.Sequential)
    assert isinstance(mini_unet_instance.bottleneck, nn.Sequential)
    assert isinstance(mini_unet_instance.final, nn.Sequential)
    assert isinstance(mini_unet_instance.pool, nn.MaxPool2d)
    assert isinstance(mini_unet_instance.upconv1, nn.ConvTranspose2d)


def test_conv_block_structure(mini_unet_instance):
    block = mini_unet_instance.conv_block(3, 64)
    assert len(block) == 4  # as there are two Conv2d and 2 ReLu
    assert isinstance(block[0], nn.Conv2d)
    assert isinstance(block[1], nn.ReLU)


@patch("torch.nn.init.kaiming_normal_")
@patch("torch.nn.init.constant_")
def test_initialize_weights_calls_init_functions(mock_constant_, mock_kaiming_normal_):
    class DummyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 1)
            self.conv_t = nn.ConvTranspose2d(1, 1, 1)
            MiniUNet._initialize_weights(self)

    DummyModule()

    assert mock_kaiming_normal_.called
    assert mock_constant_.called


def test_mini_unet_forward_pass_output_shape(mini_unet_instance):
    input_tensor = torch.randn(1, 3, 224, 224)  # Batch,Channels,Height,Width
    output = mini_unet_instance.forward(input_tensor)
    assert output.shape == (1, 104, 224, 224)  # (batch_size,num_classes,height,width)
