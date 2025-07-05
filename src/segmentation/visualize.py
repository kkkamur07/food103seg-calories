import matplotlib.pyplot as plt
import torch
import os
import numpy as np


def visualize_predictions(model, test_loader, num_images, predictions_path=None):
    """
    Simple function to visualize original images, ground truth masks, and predictions

    Args:
        model: Trained model
        data_loader: DataLoader (train_loader or test_loader)
        num_images: Number of images to visualize
        segments_path: Directory to save segmented images (optional)
    """

    # Get device and set model to eval
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Get batch of images and masks
    images, masks = next(iter(test_loader))

    # Generate predictions
    with torch.no_grad():
        predictions = model(images.to(device))

    # Create save directory if specified
    if predictions_path:
        os.makedirs(predictions_path, exist_ok=True)

    # Visualize each image
    for i in range(min(num_images, len(images))):
        _, axis = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axis[0].imshow(images[i].permute(1, 2, 0).numpy())
        axis[0].set_title(f"Image {i}")
        axis[0].axis("off")

        # Ground truth mask
        axis[1].imshow(masks[i].numpy())
        axis[1].set_title(f"Ground Truth {i}")
        axis[1].axis("off")

        # Predicted mask
        predicted_mask = predictions[i].argmax(dim=0).cpu().numpy()
        axis[2].imshow(predicted_mask)
        axis[2].set_title(f"Predicted Mask {i}")
        axis[2].axis("off")

        plt.tight_layout()

        # Save or show
        if predictions_path:
            plt.savefig(
                os.path.join(predictions_path, f"prediction_{i}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()


def visualize_feature_maps(
    model, test_loader, input_image, layer_name, num_maps=16, feature_path=None
):
    """Simple function to visualize feature maps from a specific layer"""

    model.eval()
    device = next(model.parameters()).device

    input_image, _ = next(iter(test_loader))

    # Hook to capture outputs
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # Register hook based on layer name
    if layer_name == "encoder1":
        handle = model.encoder1.register_forward_hook(get_activation("encoder1"))
    elif layer_name == "encoder2":
        handle = model.encoder2.register_forward_hook(get_activation("encoder2"))
    elif layer_name == "encoder3":
        handle = model.encoder3.register_forward_hook(get_activation("encoder3"))
    else:
        print(f"Layer {layer_name} not found")
        return

    # Forward pass
    with torch.no_grad():
        _ = model(input_image.unsqueeze(0).to(device))

    # Get feature maps
    feature_maps = activation[layer_name][0].cpu().numpy()
    handle.remove()

    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(min(num_maps, feature_maps.shape[0]))))

    # Plot feature maps
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(min(num_maps, feature_maps.shape[0])):
        axes[i].imshow(feature_maps[i], cmap="viridis")
        axes[i].set_title(f"Map {i}", fontsize=8)
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(min(num_maps, feature_maps.shape[0]), len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"{layer_name} Feature Maps")
    plt.tight_layout()

    if feature_path:
        plt.savefig(
            os.path.join(feature_path, f"{layer_name}_feature_maps.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


def visualize_training_metrics(model_path, plots_path=None):
    """
    Visualize training and testing loss & accuracy from saved model checkpoint
    Creates two side-by-side graphs: Loss comparison and Accuracy comparison

    Args:
        model_path: Path to the saved model checkpoint (.pth file)
        save_dir: Directory to save the plots
    """

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")

    # Extract metrics from checkpoint
    train_losses = checkpoint.get("train_losses", [])
    val_losses = checkpoint.get("val_losses", [])
    train_accs = checkpoint.get("train_accuracies", [])
    val_accs = checkpoint.get("val_accuracies", [])

    # Check if data exists
    if not train_losses or not val_losses:
        print("No loss data found in checkpoint!")
        return

    if not train_accs or not val_accs:
        print("No accuracy data found in checkpoint!")
        return

    # Create save directory
    os.makedirs(plots_path, exist_ok=True)

    # Create figure with 2 subplots side by side
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    epochs = range(1, len(train_losses) + 1)

    # Graph 1: Training vs Testing Loss
    ax1.plot(
        epochs,
        train_losses,
        label="Training Loss",
        color="blue",
        marker="o",
        linewidth=2,
    )
    ax1.plot(
        epochs, val_losses, label="Testing Loss", color="red", marker="s", linewidth=2
    )
    ax1.set_title("Training vs Testing Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Graph 2: Training vs Testing Accuracy
    ax2.plot(
        epochs,
        train_accs,
        label="Training Accuracy",
        color="green",
        marker="o",
        linewidth=2,
    )
    ax2.plot(
        epochs,
        val_accs,
        label="Testing Accuracy",
        color="orange",
        marker="s",
        linewidth=2,
    )
    ax2.set_title("Training vs Testing Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epochs", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(plots_path, "training_vs_testing_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Print summary
    print(f"Training metrics visualization saved to: {plot_path}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    print(f"Final training loss: {checkpoint.get('final_train_loss', 'N/A'):.4f}")
    print(f"Final testing loss: {checkpoint.get('final_val_loss', 'N/A'):.4f}")
    print(f"Final training accuracy: {checkpoint.get('final_train_acc', 'N/A'):.4f}")
    print(f"Final testing accuracy: {checkpoint.get('final_val_acc', 'N/A'):.4f}")
    print(f"Training completed at epoch: {checkpoint.get('epoch', 'N/A')}")
