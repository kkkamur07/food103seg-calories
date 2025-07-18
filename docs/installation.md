# Installation Guide

## 🚀 Live Demo

**Try the application now!** Our Streamlit app is live at: **the link**

## Prerequisites

Before installing the Food103Seg Calories project, ensure you have the following prerequisites:

- **Python 3.8+** (Python 3.9 or 3.10 recommended)
- **CUDA-compatible GPU** (recommended for training, optional for inference)
- **Git** for cloning the repository
- **pip** package manager

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.9 or 3.10 |
| RAM | 8GB | 16GB+ |
| GPU Memory | 4GB | 8GB+ |
| Storage | 10GB | 20GB+ |

## Installation Steps

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd kkkamur07-food103seg-calories
```

### 2. Create Virtual Environment (Recommended)

Using **venv**:
```bash
python -m venv food-seg-env
source food-seg-env/bin/activate  # On Windows: food-seg-env\Scripts\activate
```

Using **conda**:
```bash
conda create -n food-seg-env python=3.9
conda activate food-seg-env
```

### 3. Install Dependencies

#### Production Installation
```bash
pip install -r requirements.txt
```

#### Development Installation
```bash
pip install -r requirements_dev.txt
```

#### Install Project in Development Mode
```bash
pip install -e .
```

### 4. GPU Setup (Optional but Recommended)

For CUDA support, install PyTorch with CUDA:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU installation:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Data Setup

### 1. Download Dataset

Download the Food103 segmentation dataset and place it in the `data/` directory:

```bash
mkdir -p data
# Download your dataset here
```

### 2. Expected Directory Structure

Ensure your data follows this structure:

```
data/
├── Images/
│   ├── img_dir/
│   │   ├── train/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── ...
│   │   └── test/
│   │       ├── image1.jpg
│   │       ├── image2.jpg
│   │       └── ...
│   └── ann_dir/
│       ├── train/
│       │   ├── image1.png
│       │   ├── image2.png
│       │   └── ...
│       └── test/
│           ├── image1.png
│           ├── image2.png
│           └── ...
```

## Configuration Setup

### 1. Create Required Directories

```bash
mkdir -p saved/{logs,models,reports,predictions}
mkdir -p configs/outputs
```

### 2. Environment Variables (Optional)

Create a `.env` file in the project root:

```bash
# .env
WANDB_PROJECT=Food-Segmentation
CUDA_VISIBLE_DEVICES=0
```

## Verification

### 1. Test Installation

Run the following commands to verify your installation:

```bash
# Test imports
python -c "import torch; import torchvision; print('PyTorch installed successfully')"

# Test project modules
python -c "from src.segmentation.data import data_loaders; print('Project modules working')"

# Test data loading
python -c "from src.segmentation.data import data_loaders; print('Data loading test passed')"
```

### 2. Quick Training Test

Run a quick training test with minimal epochs:

```bash
python src/segmentation/main.py model.hyperparameters.epochs=1
```

## Running the Application

### 1. **Streamlit Web App**
```bash
streamlit run src/app/frontend.py
```

### 2. **Training Pipeline**
```bash
python src/segmentation/main.py
```

### 3. **Custom Training**
```bash
python src/segmentation/main.py model.hyperparameters.epochs=50 model.hyperparameters.lr=0.001
```

## Troubleshooting

### Common Issues

#### **CUDA Out of Memory**
- Reduce batch size in config: `model.hyperparameters.batch_size=16`
- Use CPU training: `CUDA_VISIBLE_DEVICES="" python src/segmentation/main.py`

#### **Missing Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

#### **Data Loading Errors**
- Verify directory structure matches expected format
- Check file permissions: `chmod -R 755 data/`
- Ensure image and annotation files have correct extensions

#### **Import Errors**
```bash
# Reinstall in development mode
pip install -e . --force-reinstall
```

### Getting Help

If you encounter issues:

1. **Check the logs** in `saved/logs/`
2. **Verify GPU setup** with `nvidia-smi`
3. **Review configuration** in `configs/config.yaml`
4. **Check Python version** compatibility

## Optional Components

### Docker Setup

If you prefer Docker:

```bash
# Build backend
docker build -f Dockerfile.backend -t food-seg-backend .

# Build frontend
docker build -f Dockerfile.frontend -t food-seg-frontend .

# Run with docker-compose
docker-compose up
```

### Development Tools

Install additional development tools:

```bash
# Pre-commit hooks
pre-commit install

# Jupyter for notebooks
pip install jupyter
jupyter notebook notebooks/
```

## Performance Optimization

### For Training
- Use **mixed precision**: Add `--mixed-precision` flag
- **Increase batch size** if you have enough GPU memory
- **Use multiple GPUs** with `CUDA_VISIBLE_DEVICES=0,1`

### For Inference
- Use **TorchScript** for faster inference
- **Batch processing** for multiple images
- **CPU inference** for deployment scenarios

## Next Steps

After successful installation:

1. **Review the configuration** in `configs/config.yaml`
2. **Run the training pipeline** with your data
3. **Explore the Streamlit app** at the live link above
4. **Check the documentation** for advanced usage
5. **Set up monitoring** with Weights & Biases

You're now ready to start training your food segmentation model and estimating calories! 🍕📊

Sources
