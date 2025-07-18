from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

from src.segmentation.model import MiniUNet
from src.segmentation.data import FoodSegDataset, data_loaders


@pytest.fixture()
def dummy_data(tmp_path: Path):
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


class MockMiniUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return (
            torch.randn(x.shape[0], 104, 224, 224, device=x.device)
            + self.dummy_param.to(x.device) * 0.0
        )  # Output shape should match CrossEntropyLoss.


# Mock external dependencies


@pytest.fixture
def mock_interfaces():
    with (
        patch("wandb.init") as mock_wandb_init,
        patch("wandb.watch") as mock_wandb_watch,
        patch("wandb.log") as mock_wandb_log,
        patch("wandb.Artifact") as MockWandbArtifact,
        patch("wandb.log_artifact") as mock_wandb_log_artifact,
        patch("wandb.finish") as mock_wandb_finish,
        patch("matplotlib.pyplot.subplots") as mock_subplots,
        patch("matplotlib.pyplot.savefig") as mock_savefig,
        patch("matplotlib.pyplot.show") as mock_show,
        patch("matplotlib.pyplot.close") as mock_close,
        patch("tqdm.tqdm") as mock_tqdm_class,
        patch("torch.save") as mock_torch_save,
        patch("loguru.logger.info") as mock_logger_info,
        patch("loguru.logger.warning") as mock_logger_warning,
        patch("builtins.print") as mock_builtins_print,
        patch("src.segmentation.data.data_loaders") as mock_data_loaders,
    ):

        # Configure mock_tqdm_class
        mock_tqdm_instance = MagicMock()
        mock_tqdm_instance.set_postfix = MagicMock()
        mock_tqdm_instance.__iter__.return_value = iter([])
        mock_tqdm_class.side_effect = (
            lambda iterable, *args, **kwargs: mock_tqdm_instance
        )

        # Configure mock_data_loaders to simulate two samples
        mock_train_dataset = MagicMock(spec=FoodSegDataset)
        mock_train_dataset.__len__.return_value = 2
        mock_test_dataset = MagicMock(spec=FoodSegDataset)
        mock_test_dataset.__len__.return_value = 2

        mock_train_loader = MagicMock(spec=torch.utils.data.DataLoader)
        mock_train_loader.dataset = mock_train_dataset
        mock_train_loader.batch_size = 2

        mock_test_loader = MagicMock(spec=torch.utils.data.DataLoader)
        mock_test_loader.dataset = mock_test_dataset
        mock_test_loader.batch_size = 2

        mock_image_batch = torch.randn(2, 3, 224, 224)  # batch_size, channels, H, W
        mock_mask_batch = torch.randint(
            0, 104, (2, 224, 224), dtype=torch.long
        )  # batch_size, H, W

        mock_train_loader.__iter__.return_value = iter(
            [(mock_image_batch, mock_mask_batch)]
        )
        mock_test_loader.__iter__.return_value = iter(
            [(mock_image_batch, mock_mask_batch)]
        )

        mock_data_loaders.return_value = (mock_train_loader, mock_test_loader)

        yield {
            "mock_wandb_init": mock_wandb_init,
            "mock_wandb_watch": mock_wandb_watch,
            "mock_wandb_log": mock_wandb_log,
            "MockWandbArtifact": MockWandbArtifact,
            "mock_wandb_log_artifact": mock_wandb_log_artifact,
            "mock_wandb_finish": mock_wandb_finish,
            "mock_subplots": mock_subplots,
            "mock_savefig": mock_savefig,
            "mock_show": mock_show,
            "mock_close": mock_close,
            "mock_tqdm_instance": mock_tqdm_instance,  # Use a mock instance for tqdm
            "mock_tqdm_class": mock_tqdm_class,
            "mock_torch_save": mock_torch_save,
            "mock_logger_info": mock_logger_info,
            "mock_logger_warning": mock_logger_warning,
            "mock_builtins_print": mock_builtins_print,
            "mock_data_loaders": mock_data_loaders,
            "mock_train_loader_instance": mock_train_loader,
            "mock_test_loader_instance": mock_test_loader,
        }


@pytest.fixture
def trainer_instance(dummy_data, mock_interfaces):
    with patch("src.segmentation.model.MiniUNet", new=MockMiniUNet):
        from src.segmentation.train import Trainer

        base_dir = dummy_data
        os.makedirs(os.path.join(base_dir, "saved", "models"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "saved", "reports"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "saved", "predictions"), exist_ok=True)
        trainer = Trainer(lr=0.001, epochs=2, batch_size=2, base_dir=base_dir)
        return trainer


def test_trainer_intialization(trainer_instance, mock_interfaces):
    trainer = trainer_instance
    deps = mock_interfaces

    assert trainer.epochs == 2
    assert trainer.lr == 0.001
    assert trainer.batch_size == 2
    assert isinstance(trainer.model, MockMiniUNet)
    assert isinstance(trainer.loss, nn.CrossEntropyLoss)
    assert isinstance(trainer.optimizer, optim.Adam)

    deps["mock_data_loaders"].assert_called_once_with(
        base_dir=trainer.base_dir, batch_size=trainer.batch_size
    )
    assert trainer.train_loader is deps["mock_train_loader_instance"]
    assert trainer.test_loader is deps["mock_test_loader_instance"]

    deps["mock_wandb_init"].assert_called_once()
    deps["mock_wandb_watch"].assert_called_once_with(trainer.model, log="all")


# test if forward method is called
def test_trainer_forward_method(trainer_instance):
    input_tensor = torch.randn(1, 3, 224, 224)
    output = trainer_instance.forward(input_tensor)
    assert output.shape == (1, 104, 224, 224)


@pytest.mark.parametrize("epochs_to_test", [1, 2, 3])
def test_trainer_train_loop_execution(
    trainer_instance, mock_interfaces, epochs_to_test
):
    trainer = trainer_instance
    deps = mock_interfaces

    trainer.epochs = epochs_to_test
    trainer.train()
    assert len(trainer.train_losses) == epochs_to_test
    assert len(trainer.test_losses) == epochs_to_test
    assert len(trainer.train_accs) == epochs_to_test
    assert len(trainer.test_accs) == epochs_to_test

    deps["mock_torch_save"].assert_called_once()
    assert deps["mock_wandb_log"].call_count == trainer.epochs
    deps["mock_wandb_finish"].assert_called_once()
    deps["mock_wandb_log_artifact"].assert_called()
    deps["MockWandbArtifact"].assert_called()
    deps["mock_logger_info"].assert_called_with(
        f"Training complete. Model saved at {trainer.model_path}"
    )
    deps["mock_builtins_print"].assert_called_with("Training complete.")


# train exits if model path is not set
def test_train_model_path(trainer_instance, mock_interfaces):
    trainer = trainer_instance
    dependencies = mock_interfaces
    trainer.model_path = None
    trainer.train()
    dependencies["mock_logger_warning"].assert_called_once_with(
        "Model path is not set. Cannot save the model."
    )
    dependencies["mock_wandb_finish"].assert_not_called()


# Test visualization training metrics
def test_visualize_training_metrics(trainer_instance, mock_interfaces):
    trainer = trainer_instance
    dependencies = mock_interfaces
    # dummy data for plotting
    trainer.train_losses = [0.1, 0.05]
    trainer.test_losses = [0.2, 0.1]
    trainer.train_accs = [0.8, 0.9]
    trainer.test_accs = [0.7, 0.85]

    mock_fig = MagicMock()
    mock_ax1 = MagicMock()
    mock_ax2 = MagicMock()
    mock_ax3 = MagicMock()
    dependencies["mock_subplots"].return_value = (
        mock_fig,
        (mock_ax1, mock_ax2, mock_ax3),
    )

    trainer.visualize_training_metrics()

    dependencies["mock_subplots"].assert_called_once()
    dependencies["mock_savefig"].assert_called_once_with(trainer.plots_path, dpi=300)
    dependencies["mock_show"].assert_called_once()
    dependencies["mock_close"].assert_called_once()
    dependencies["mock_logger_info"].assert_called_with(
        f"Training metrics plot saved: {trainer.plots_path}"
    )


# Test visualization predictions
def test_visualize_predictions(trainer_instance, mock_interfaces):
    trainer = trainer_instance
    dependencies = mock_interfaces

    # Mocking 3 rows, num_images columns
    num_images = 1  # Or set to 5 if you're using default
    mock_ax = MagicMock()
    mock_axes_array = np.empty((3, num_images), dtype=object)
    for i in range(3):
        for j in range(num_images):
            mock_axes_array[i, j] = MagicMock()

    dependencies["mock_subplots"].return_value = (mock_ax, mock_axes_array)

    with patch.object(
        trainer.model, "forward", wraps=trainer.model.forward
    ) as mock_model_forward:
        trainer.visualize_predictions(num_images=1)
        assert mock_model_forward.called

    dependencies["mock_subplots"].assert_called_once()
    dependencies["mock_savefig"].assert_called_once_with(
        trainer.predictions, dpi=300, bbox_inches="tight"
    )
    dependencies["mock_show"].assert_called_once()
    dependencies["mock_close"].assert_called_once()
    dependencies["mock_logger_info"].assert_called_with(
        f"Prediction grid saved: {trainer.predictions}"
    )

    for i in range(3):
        for j in range(num_images):
            ax = mock_axes_array[i, j]
            ax.imshow.assert_called()


# Visualization metric warns if no training data is present
def test_visualization_warns_no_data(trainer_instance, mock_interfaces):
    trainer = trainer_instance
    dependencies = mock_interfaces

    trainer.train_losses = []
    trainer.test_losses = []
    trainer.train_accs = []
    trainer.test_accs = []

    trainer.visualize_training_metrics()

    dependencies["mock_logger_warning"].assert_any_call(
        "No loss data found in checkpoint!"
    )
