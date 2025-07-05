import numpy as np


def iou(pred, target, num_classes=103):
    """Calculate mean IoU"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()
        union = (
            pred_inds.long().sum().data.cpu().item()
            + target_inds.long().sum().data.cpu().item()
            - intersection
        )
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(float(intersection) / float(max(union, 1)))

    return np.array(ious)
