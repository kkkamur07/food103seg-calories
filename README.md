# Food103Seg Calories

Welcome to **Food103Seg** – a deep learning project for food image segmentation.

## 🚀 Live Demo

Try out the app here: [Live Demo](https://segmentation-frontend-289925381630.us-central1.run.app/)

## Overview

**Food103Seg** utilizes computer vision to identify different food items in images via semantic segmentation. The project is built with PyTorch, leveraging a simplified MiniUNet architecture, and can segment 104 food categories with moderate accuracy. Training was performed on an RTX 4090 for approximately 20 minutes, with optimal hyperparameters determined using Weights & Biases sweeps.

## Features

- **🍕 Food Segmentation**: Detects and segments multiple food items within an image.
- **🧠 MiniUNet Model**: Lightweight custom U-Net variant designed for efficient food segmentation.
- **🌐 Web Interface**: Simple and interactive Streamlit application.
- **⚙️ Flexible Configuration**: Managed with Hydra for consistency and reproducibility.
- **📈 Experiment Tracking**: Seamless integration with Weights & Biases for run comparison and tracking.
- **🦾 MLOps-ready**: Includes ready-to-use MLOps cookiecutter template.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/kkkamur07/food103seg-calories

# Navigate to the project directory
cd food103seg-calories

# Create a virtual environment using UV
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Train the model
python -m src.segmentation.main

# Launch the API Server
uvicorn src.app.api:app --host 0.0.0.0 --port 8000 --reload

# Launch the Web App
streamlit run src/app/frontend.py
```

## MLOps Template

A production-ready MLOps template is included for rapid project structuring:

```bash
# Install cookiecutter
pip install cookiecutter

# Generate a new project structure
cookiecutter https://github.com/kkkamur07/cookie-cutter --directory=mlops
```

Find the full template and installation guide at:
```
https://github.com/kkkamur07/cookie-cutter --directory=mlops
```

## Project Structure

```
├── src/
│   ├── segmentation/         # Core ML modules
│   │   ├── data.py           # Data loading utilities
│   │   ├── model.py          # MiniUNet architecture
│   │   ├── train.py          # Model training pipeline
│   │   ├── main.py           # Main training runner
│   │   ├── bentoml.py        # BentoML deployment
│   │   └── bentoml_setup.py  # BentoML configuration
│   └── app/                  # Web and API interface
│       ├── frontend.py       # Streamlit web UI
│       └── service.py        # API backend logic
├── configs/                  # Model and training configs
├── notebooks/                # Jupyter Notebooks for experimentation
└── docs/                     # Project documentation
```

## Model Performance

- **104 Food Classes:** Extensive coverage of common food items.
- **Moderate Accuracy:** Achieves over 20% mean IoU on the test dataset.
- **Moderate Inference Speed:** ~100ms per image with GPU support.
- **Compact Size:** Weighs around 15MB for deployment ease.

## System Architecture

<!-- Space intentionally left for a system architecture diagram.
   Insert a comprehensive diagram here (e.g. flow of data from frontend upload -> segmentation model -> API -> calorie estimates/output) for better clarity. -->

_Add your system architecture diagram here to illustrate component interaction and flow._

## Getting Started

1. **[Installation](installation.md)** – Step-by-step setup guide.
2. **[Quick Start](quickstart.md)** – Fastest way to run the project.
3. **[API Reference](api/data.md)** – Explore available code and endpoints.
4. **[Training Guide](guide/training.md)** – Tutorial for training your own models.

## Try It Yourself

To get started, check the [installation guide](installation.md) or launch the [live demo](https://segmentation-frontend-289925381630.us-central1.run.app/).
