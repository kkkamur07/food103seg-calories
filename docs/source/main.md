# Main Pipeline

Main training pipeline with Hydra configuration management for food segmentation.

## CLI and Hydra Configuration Management

This module demonstrates advanced **CLI configuration management** using **Hydra** and **OmegaConf**, providing flexible parameter override capabilities and structured configuration handling.

### Key Configuration Features

**Hydra Integration:**
```python
@hydra.main(version_base=None, config_path=config_dir, config_name="config")
def main(cfg: DictConfig):
```

**Dynamic Configuration Loading:**
- Automatically resolves config directory from project root
- Loads configuration from `configs/config.yaml`
- Supports hierarchical configuration structures

### CLI Usage Examples

**Default Configuration:**
```bash
python src/segmentation/main.py
```

**Parameter Override:**
```bash
python src/segmentation/main.py model.hyperparameters.epochs=50 model.hyperparameters.lr=0.001
```

**Multiple Parameters:**
```bash
python src/segmentation/main.py model.hyperparameters.epochs=100 model.hyperparameters.batch_size=32 paths.base_dir=/custom/path
```

### Configuration Structure

The pipeline expects configuration with the following structure:

```yaml
model:
  hyperparameters:
    epochs: 20
    batch_size: 16
    lr: 0.0001

paths:
  base_dir: "data/"

profiling:
  enabled: false
```

### Enhanced User Experience

**Rich Console Integration:**
- Displays formatted hyperparameter tables
- Provides colorized progress indicators
- Shows training status with visual panels

**Automatic Environment Setup:**
- Configures Weights & Biases (wandb) in silent mode
- Sets up proper Python path resolution
- Handles project root discovery

### Pipeline Workflow

1. **Configuration Loading** - Hydra loads and validates config
2. **Parameter Display** - Rich table shows current hyperparameters
3. **Data Loading** - Initializes train/test data loaders
4. **Training Execution** - Runs complete training pipeline
5. **Visualization Generation** - Creates metrics and prediction plots
6. **Completion** - Displays success confirmation

This approach provides **reproducible experiments** with easy parameter tuning through CLI overrides, making it ideal for hyperparameter sweeps and experiment tracking.

```python
:::src.segmentation.main
```

Sources
