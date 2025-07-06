import numpy
import torch


def iou(pred, target):
    """
    Memory efficient version for very large images/batches
    Only processes classes that are present in the batch
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
