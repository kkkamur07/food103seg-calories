# Installation Guide


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

## Setup Instructions Using uv

### 1. Clone the Repository

```bash
git clone https://github.com/kkkamur07/food103seg-calories
cd food103seg-calories
```

### 2. Create Virtual Environment and Install Dependencies

Using **uv** (recommended for fastest setup):

```bash
# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install production dependencies
uv pip install -r requirements.txt

# Install development dependencies (optional)
uv pip install -r requirements_dev.txt

# Install project in development mode
uv pip install -e .
```

**Alternative one-liner approach:**
```bash
# Create environment and install dependencies in one step
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt
```

### 3. GPU Setup (Optional but Recommended)

For CUDA support, install PyTorch with CUDA using uv:

```bash
# For CUDA 11.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU installation:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### About the Requirements Files

This project provides two dependency files for different use cases[1][2]:

- **`requirements.txt`** - Contains production dependencies needed to run the application
- **`requirements_dev.txt`** - Contains additional development dependencies for testing, linting, and development tools[3]

The uv package manager provides significant speed improvements over traditional pip installations[4][2], making it ideal for projects with multiple dependencies. You can install from either file using `uv pip install -r <filename>`[5].


## Data Setup

## 1. Download Dataset

### Data Storage and Versioning with DVC

This project uses **DVC (Data Version Control)** with Google Cloud Storage for data versioning and management. The data and models are stored in two separate GCS buckets:

- **Data storage**: `gs://dvc-storage-sensor/`
- **Model storage**: `gs://food-segmentation-models/`

### Setting Up DVC with Google Cloud Storage

```bash
# Create data directory
mkdir -p data

# Install required tools
pip install dvc-gs

# List available GCS buckets
gsutil ls

# Add remote storage (replace <output-from-gsutils> with actual bucket path)
dvc remote add -d remote_storage gs://dvc-storage-sensor/

# Configure version-aware storage
dvc remote modify remote_storage version_aware true

# List configured remotes
dvc remote list

# Pull data from remote storage
dvc pull
```

### DVC Management Commands

```bash
# Remove a remote if needed
dvc remote remove gcp_storage

# Set default remote
dvc remote default remote_storage

# Push data (note: --no-cache may have issues)
dvc push --no-cache
```

### Known Issues with DVC Setup

During development, several challenges were encountered with the DVC workflow, particularly with the `dvc push --no-cache` command. While DVC provides excellent data versioning capabilities, the setup proved complex for this project's requirements.

### Alternative: Direct Dataset Download

If you prefer to bypass the DVC setup or encounter issues, you can download the **Food103 segmentation dataset** directly from:

**Dataset source**: https://paperswithcode.com/dataset/foodseg103

```bash
# Create data directory
mkdir -p data

# Download dataset manually from Papers with Code
# Extract and place in data/ directory
```

### Recommended Approach

For this project, you can choose either approach:

1. **DVC approach** - Use the GCS buckets with DVC for version control
2. **Direct download** - Download the Food103 dataset directly from Papers with Code

The DVC setup provides better data versioning and collaboration features, while the direct download approach is simpler for getting started quickly.


### 2. Expected Data Directory Structure

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

## Copy the Template
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

### 2. **API Server (FastAPI with Uvicorn)**
```bash
uvicorn src.app.api:app --host 0.0.0.0 --port 8000 --reload
```

### 3. **Training Pipeline**
```bash
python src/segmentation/main.py
```

### 4. **Custom Training**
```bash
python src/segmentation/main.py model.hyperparameters.epochs=50 model.hyperparameters.lr=0.001
```

### Access Points
- **Web App**: `http://localhost:8501`
- **API**: `http://localhost:8000`
- **API Docs**: `http://localhost:8000/docs`


## Troubleshooting

### Common Issues

#### **CUDA Out of Memory**
- Reduce batch size in config: `model.hyperparameters.batch_size=16`

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

## Next Steps

After successful installation:

1. **Review the configuration** in `configs/config.yaml`
2. **Run the training pipeline** with your data
3. **Explore the Streamlit app** at the live link above
4. **Check the documentation** for advanced usage
5. **Set up monitoring** with Weights & Biases

You're now ready to start training your food segmentation model and estimating calories! 🍕📊
