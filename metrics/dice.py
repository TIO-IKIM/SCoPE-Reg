"""Dice similarity coefficient.

Overlap between a predicted binary mask and the GT segmentation:
    Dice = 2|P n T| / (|P| + |T|),  range [0, 1], higher is better.

In the registration setting the "predicted" mask is usually the GT slice
segmentation warped by the residual pose error (see evaluate.py); for models
that output their own mask, that mask can be passed directly.
"""
import torch


def dice(pred, target, threshold=0.5, eps=1e-6):
    """
    pred, target : mask tensors (B, ...) — pred is thresholded, target assumed
    binary (any >threshold counts as foreground). Returns the batch-mean Dice.
    """
    p = (pred > threshold).float().reshape(pred.shape[0], -1)
    t = (target > threshold).float().reshape(target.shape[0], -1)
    inter = (p * t).sum(dim=1)
    per_sample = (2 * inter + eps) / (p.sum(dim=1) + t.sum(dim=1) + eps)
    return per_sample.mean()
