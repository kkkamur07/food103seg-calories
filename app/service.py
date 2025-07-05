import bentoml
from bentoml import Service
from bentoml.io import Image
import torch
import numpy as np
from PIL import Image as PILImage

runner = bentoml.models.get("food_segmentation_model:latest").to_runner()

svc = Service(name= "Food_Segmentation", runners = [runner])

@svc.api(input=Image(), output=Image())

async def segmenting_food(image: PILImage.Image) -> PILImage.Image:
    # Preprocessing the image
    img_tensor = preprocessing_image(image)

       # Run inference
    with torch.no_grad():
        output = await runner.async_run([img_tensor])

       # Postprocess the output
    segmented_image = postprocessing_output(output)

    return segmented_image

def preprocessing_image(image: PILImage.Image) -> torch.Tensor:
        image = image.convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        return img_tensor.unsqueeze(0)

def postprocessing_output(output: torch.Tensor) -> PILImage.Image:
    output_image = output.squeeze(0).permute(1, 2, 0).numpy()
    output_image = (output_image * 255).astype(np.uint8)
    return PILImage.fromarray(output_image)

