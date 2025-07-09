import sys
import os
import io
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../saved/models/best_model.pth")
from fastapi import FastAPI, File, UploadFile
from src.segmentation.model import MiniUNet
from PIL import Image 
import torch

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniUNet()
#Loading the model weights
dict=torch.load(model_path, map_location=device)
model.load_state_dict(dict["model_state_dict"])
model.to(device)
model.eval()

def preprocess_image(image: Image.Image):
    """Convert PIL Image to normalized tensor"""
    image = image.convert("RGB")
    img_resize = image.resize((256, 256))  # Changing the size according to the model's input size
    img_tensor = transforms.ToTensor()(img_resize)
    return img_tensor.unsqueeze(0) 

def postprocess_output(output: torch.Tensor):
        """Convert model output to PIL Image"""
        output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        colored_mask = plt.cm.tab20(output % 20)[..., :3]  # Drop alpha channel. As we only want to extract the RGB channels
        colored_mask = (colored_mask * 255).astype(np.uint8) 
        return Image.fromarray(colored_mask)    
    
@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    img_tensor = preprocess_image(image)
    img_tensor = img_tensor.to(device)

    result = postprocess_output(img_tensor)
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/png")



