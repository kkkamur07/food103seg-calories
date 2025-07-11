import bentoml
import torch
import numpy as np
from PIL import Image as PILImage
from torchvision import transforms
import matplotlib.pyplot as plt


@bentoml.service(
    name="food_segmentation_model:latest",
    resources={"gpu": 1} if torch.cuda.is_available() else {"cpu": 1},
    traffic={"timeout": 60},
)

# Defining our BentoML Services
class FoodSegmentationService:
    def __init__(self):
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize the service with the model
        self.model = bentoml.models.BentoModel("food_segmentation_model:latest")

    # Creating a service to handle the segmentation of food items
    @bentoml.api()
    async def segment(self, image: PILImage.Image) -> PILImage.Image:
        """Segment food items and return visualization"""
        # Preprocess
        img_tensor = await self.preprocess_image(image)
        img_tensor = img_tensor.to(self.device)

        return self.postprocess_output(img_tensor)

    async def preprocess_image(self, image: PILImage.Image) -> torch.Tensor:
        """Convert PIL Image to normalized tensor"""
        image = image.convert("RGB")
        img_resize = image.resize(
            (256, 256)
        )  # Changing the size according to the model's input size
        img_tensor = transforms.ToTensor()(img_resize)
        return img_tensor.unsqueeze(0)

    def postprocess_output(self, output: torch.Tensor) -> PILImage.Image:
        """Convert model output to PIL Image"""
        output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        colored_mask = plt.cm.tab20(output % 20)[
            ..., :3
        ]  # Drop alpha channel. As we only want to extract the RGB channels
        colored_mask = (colored_mask * 255).astype(np.uint8)
        return PILImage.fromarray(colored_mask)

