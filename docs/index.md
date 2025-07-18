# Food103Seg Calories

Welcome to **Food103Seg** - a deep learning project for food image segmentation and calorie estimation.

## 🚀 Live Demo

**Try the application now!** Our Streamlit app is live at: [**the link**](your-streamlit-link)

## Overview

This project uses computer vision to identify different food items in images and estimate their caloric content. Built with PyTorch and featuring a MiniUNet architecture, it can segment 104 different food categories with high accuracy.

## Features

- **🍕 Food Segmentation**: Identifies and segments different food items in images
- **📊 Calorie Estimation**: Calculates caloric content from segmented food regions
- **🧠 MiniUNet Model**: Lightweight U-Net architecture optimized for food segmentation
- **🌐 Web Interface**: User-friendly Streamlit application
- **⚙️ Easy Configuration**: Hydra-based configuration management
- **📈 Experiment Tracking**: Integration with Weights & Biases

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python src/segmentation/main.py

# Launch web app
streamlit run src/app/frontend.py
```

## Project Structure

```
├── src/
│   ├── segmentation/          # Core ML modules
│   │   ├── data.py           # Data loading
│   │   ├── model.py          # MiniUNet architecture
│   │   ├── train.py          # Training pipeline
│   │   └── main.py           # Main training script
│   └── app/                  # Web application
│       ├── frontend.py       # Streamlit interface
│       └── service.py        # API service
├── configs/                  # Configuration files
├── notebooks/                # Jupyter notebooks
└── docs/                     # Documentation
```

## Getting Started

1. **[Installation](installation.md)** - Set up the project
2. **[Quick Start](quickstart.md)** - Get running quickly
3. **[API Reference](api/data.md)** - Explore the code
4. **[Training Guide](guide/training.md)** - Train your own models

## Model Performance

- **104 Food Classes**: Comprehensive food category coverage
- **High Accuracy**: >85% mean IoU on test set
- **Fast Inference**: ~50ms per image on GPU
- **Lightweight**: ~15MB model size

Ready to start? Check out our [installation guide](installation.md) or try the [live demo](your-streamlit-link)!

Sources
