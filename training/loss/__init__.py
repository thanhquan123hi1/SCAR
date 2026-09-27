"""Loss functions for segmentation training."""
from training.loss.losses import DiceLoss, SegmentationLoss


def build_loss(loss_name="ce_dice", **kwargs):
    """Build the compound loss (M2/M3), returning loss and logging components."""
    if loss_name.lower().replace("-", "_") not in {"ce_dice", "segmentation", "segmentation_loss"}:
        raise ValueError(f"Unknown loss {loss_name!r}; expected 'ce_dice'.")
    if "num_classes" in kwargs:
        classes = kwargs.pop("num_classes")
        if "n_classes" in kwargs and kwargs["n_classes"] != classes:
            raise ValueError("Conflicting num_classes and n_classes")
        kwargs["n_classes"] = classes
    return SegmentationLoss(**kwargs)


__all__ = [
    "DiceLoss",
    "SegmentationLoss",
    "build_loss",
]
