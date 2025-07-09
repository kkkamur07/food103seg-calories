import io
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
from src.segmentation.model import MiniUNet
from PIL import Image
import torch
from contextlib import asynccontextmanager
from loguru import logger
import os

# Global variables for model
model = None
device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniUNet()

    # Load the model weights
    checkpoint = torch.load("saved/models/model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Model loaded on {device}")
    yield

    # Shutdown: Clean up if needed
    logger.info("Shutting down...")


app = FastAPI(
    lifespan=lifespan,
    title="Food Segmentation Service",
    description="A FastAPI service for segmenting food items in images using a MiniUNet model.",
    version="1.0.0",
)


def preprocess_image(image: Image.Image):
    """Convert PIL Image to normalized tensor"""
    image = image.convert("RGB")
    img_resize = image.resize((256, 256))
    img_tensor = transforms.ToTensor()(img_resize)
    return img_tensor.unsqueeze(0)


def postprocess_output(output: torch.Tensor):
    """Convert model output to PIL Image"""
    output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    colored_mask = plt.cm.tab20(output % 20)[..., :3]
    colored_mask = (colored_mask * 255).astype(np.uint8)
    return Image.fromarray(colored_mask)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to the Food Segmentation API!"}


@app.get("/healthz")
async def health_check():
    """Health check endpoint for Streamlit frontend"""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    """Segment food items in uploaded image"""
    try:
        # Read and process the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Preprocess the image
        img_tensor = preprocess_image(image)
        img_tensor = img_tensor.to(device)
        
        # Postprocess the model output
        result = postprocess_output(img_tensor)
        # Convert result to bytes for response
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return {"error": f"Processing failed: {str(e)}"}