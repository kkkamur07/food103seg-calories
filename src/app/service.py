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
import time
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Prometheus monitoring imports
from prometheus_client import (
    Counter,
    Histogram,
    Summary,
    make_asgi_app,
    CollectorRegistry,
)

# Global variables for model
model = None
device = None

# Custom Prometheus registry
MY_REGISTRY = CollectorRegistry()

# Prometheus metrics with custom registry
error_counter = Counter(
    "api_errors_total",
    "Total number of errors encountered by the API",
    registry=MY_REGISTRY,
)
request_counter = Counter(
    "api_requests_total",
    "Total number of requests received by the API",
    registry=MY_REGISTRY,
)
classification_time_histogram = Histogram(
    "classification_duration_seconds",
    "Time taken to classify a review",
    registry=MY_REGISTRY,
)
review_size_summary = Summary(
    "review_size_bytes", "Size of the reviews classified in bytes", registry=MY_REGISTRY
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI app
    This function initializes the model and device on startup and cleans up on shutdown.

    """
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

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")


# Serve favicon
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")


# Mount Prometheus metrics endpoint with custom registry
metrics_app = make_asgi_app(registry=MY_REGISTRY)
app.mount("/metrics", metrics_app)


def preprocess_image(image: Image.Image):
    """Convert PIL Image to normalized tensor
    Args:
        image (PIL.Image): Input image to preprocess.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = image.convert("RGB")
    img_resize = image.resize((256, 256))
    img_tensor = transforms.ToTensor()(img_resize)
    return img_tensor.unsqueeze(0)


def postprocess_output(output: torch.Tensor):
    """Convert model output to PIL Image
    Args:
        output (torch.Tensor): Model output tensor.
    Returns:
        PIL.Image: Postprocessed image.
    """
    output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    colored_mask = plt.cm.tab20(output % 20)[..., :3]
    colored_mask = (colored_mask * 255).astype(np.uint8)
    return Image.fromarray(colored_mask)


@app.get("/")
async def root():
    """Root endpoint
    Returns:
        dict: A simple message indicating the API is running.
    """
    return {"message": "Food Segmentation API is running"}


@app.get("/status")
async def health_check():
    """Health check endpoint for Streamlit frontend
    Returns:
        dict: Health status of the API and model.
    """
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    """Segment food items in uploaded image
    Args:
        file (UploadFile): The uploaded image file.
    Returns:
        StreamingResponse: The segmented image.
    """
    # Increment request counter
    request_counter.inc()

    # Start timing the classification
    start_time = time.time()

    try:
        # Read and process the uploaded image
        contents = await file.read()

        # Measure file size for summary metric
        file_size = len(contents)
        review_size_summary.observe(file_size)

        # Add validation to trigger error for testing
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            raise ValueError("File size too large - maximum 100MB allowed")

        image = Image.open(io.BytesIO(contents))

        # Preprocess the image
        img_tensor = preprocess_image(image)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            model_output = model(img_tensor)

        # Postprocess the model output
        result = postprocess_output(model_output)

        # Convert result to bytes for response
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)

        # Record classification time
        classification_duration = time.time() - start_time
        classification_time_histogram.observe(classification_duration)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        # Increment error counter when an error occurs
        error_counter.inc()

        # Still record the classification time even if it failed
        classification_duration = time.time() - start_time
        classification_time_histogram.observe(classification_duration)

        logger.error(f"Processing failed: {str(e)}")
        return {"error": f"Processing failed: {str(e)}"}
