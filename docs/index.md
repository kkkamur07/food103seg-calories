# Food103Seg Calories

Welcome to **Food103Seg** - a deep learning project for food image segmentation.

## 🚀 Live Demo

**Try the application now!** Our Streamlit app is live at: [**the link**](https://segmentation-frontend-289925381630.us-central1.run.app/)

## Overview

This project uses computer vision to identify different food items in images by segmenting them. We have Built this with PyTorch and by using a simplified MiniUNet architecture, it can segment 104 different food categories with moderate accuracy. We trained the MiniUNet architecture on *RTX* 4090 for about 20 Minutes and the hyperparameter configuration were determined by the hyperparameter sweep of wandb.

## Features

- **🍕 Food Segmentation**: Identifies and segments different food items in images
- **🧠 MiniUNet Model**: Lightweight U-Net architecture optimized for food segmentation
- **🌐 Web Interface**: User-friendly Streamlit application
- **⚙️ Easy Configuration**: Hydra-based configuration management
- **📈 Experiment Tracking**: Integration with Weights & Biases

## Quick Start

```bash
# Clone the repository
git clone https://github.com/kkkamur07/food103seg-calories

# Navigate to the project directory
cd food103seg-calories

# Create virtual environment using UV
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the dependencies
uv pip install -r requirements.txt

# Train the model
python -m src.segmentation.main

# Launch API Server
uvicorn src.app.service:app --host 0.0.0.0 --port 8000 --reload

# Launch Web App
streamlit run src/app/frontend.py
```

## MLOps Template

We provide a ready-to-use MLOps template for this project structure:

```bash
# Install cookiecutter
pip install cookiecutter

# Generate project using our template
cookiecutter https://github.com/kkkamur07/cookie-cutter --directory=mlops
```

Find the complete template and installation guide at:
```
https://github.com/kkkamur07/cookie-cutter --directory=mlops
```

Sources


## Project Structure

```
├── src/
│   ├── segmentation/         # Core ML modules
│   │   ├── data.py           # Data loading
│   │   ├── model.py          # MiniUNet architecture
│   │   ├── train.py          # Training pipeline
│   │   └── main.py           # Main training script
│   └── app/                  # Web application
│       ├── frontend.py       # Streamlit interface
│       └── service.py        # API service
├── configs/                  # Configuration files and management
├── notebooks/                # Jupyter notebooks for experiments
└── docs/                     # Documentation
```

We have also used `bentoML` for optimized API for ML models specially, this can be found in.
```
├── src/
    ├── segmentation/
        ├── bentoml.py        # For BentoML
        ├── bentoml_setup.py  # For setting up BentoML

```


**Serve the API locally using BentoML**

```bash
bentoml serve src/segmentation/bentoml:latest
```
   

## Getting Started

1. **[Installation](installation.md)** - Set up the project
2. **[API Reference](source/data.md)** - Explore the code
3. **[Training Guide](source/training.md)** - Train your own models

## Model Performance

- **104 Food Classes**: Comprehensive food category coverage
- **Moderate Accuracy**: >20% mean IoU on test set
- **Moderate Inference Speed**: ~100ms per image on GPU
- **Lightweight**: ~15MB model size

Ready to start? Check out our [installation guide](installation.md) or try the [live demo]((https://segmentation-frontend-289925381630.us-central1.run.app/))!

Sources
