import torch


def iou(pred, target):
    """
    Calculate Intersection over Union (IoU) for semantic segmentation.

    Memory efficient version optimized for very large images/batches.
    Only processes classes that are present in the current batch to save computation.

    Args:
        pred (torch.Tensor): Predicted segmentation masks. Can be either:
            - 4D tensor (B, C, H, W) with class probabilities/logits
            - 3D tensor (B, H, W) with class indices
        target (torch.Tensor): Ground truth segmentation masks with class indices.
            Shape: (B, H, W) where each value is a class index.

    Returns:
        numpy.ndarray: IoU scores for each class (shape: [104,]).
            Classes not present in the batch will have NaN values.
            Only classes present in pred or target will have computed IoU scores.

    Note:
        - Assumes 104 total classes (food segmentation classes)
        - Automatically handles conversion from logits to class predictions
        - Memory efficient by only computing IoU for classes present in batch
        - Returns results on CPU as numpy array for compatibility

    Example:
        >>> pred = torch.rand(2, 104, 224, 224)  # Batch of 2, 104 classes
        >>> target = torch.randint(0, 104, (2, 224, 224))  # Ground truth
        >>> iou_scores = iou(pred, target)
        >>> print(f"Mean IoU: {np.nanmean(iou_scores):.3f}")
    """
    num_classes = 104

    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)  # max class value along each pixel

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Find unique classes present in this batch
    unique_classes = torch.unique(torch.cat([pred_flat, target_flat]))
    unique_classes = unique_classes[unique_classes < num_classes]

    # Initialize full IoU array with NaN
    iou_results = torch.full((num_classes,), float("nan"), device=pred.device)

    # Only calculate IoU for classes present in batch
    for cls in unique_classes:
        pred_cls = pred_flat == cls
        target_cls = target_flat == cls

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union > 0:
            iou_results[cls] = intersection / union

    return iou_results.cpu().numpy()
