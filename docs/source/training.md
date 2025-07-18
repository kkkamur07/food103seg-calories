# Training Module

Comprehensive training pipeline with metrics tracking and visualization for food segmentation.

## What We Are Tracking

### Core Metrics
- **Loss Functions** - Training and validation loss per epoch
- **Segmentation Accuracy** - Pixel-wise accuracy and mean IoU
- **Learning Progress** - Learning rate schedules and training time
- **Model Performance** - Validation metrics and best model checkpointing

### Experiment Tracking
- **Weights & Biases Integration** - Hyperparameters, model architecture, and system metrics
- **Visualization Outputs** - Training curves, loss plots, and prediction visualizations

## Tracking Architecture

**Hybrid approach** combining:
- Local logging with Rich console output
- Weights & Biases for cloud-based experiment tracking
- File-based visualization saves
- Model checkpoint management

This focuses on **essential segmentation metrics** while maintaining training pipeline simplicity.

```python
:::src.segmentation.train
```
