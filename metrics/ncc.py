"""I-NCC — image normalized cross-correlation.

Global (Pearson) normalized cross-correlation between two images — typically
the input frame and the slice resampled from the volume at the predicted pose.
Range [-1, 1], higher is better. Matches the NCC value train.py reports
(ncc = 1 - ncc_loss).
"""
import torch


def ncc(a, b, eps=1e-8):
    """
    a, b : tensors with matching shape (B, ...). Correlation is computed per
    sample over all non-batch elements, then averaged over the batch.
    """
    a = a.reshape(a.shape[0], -1).float()
    b = b.reshape(b.shape[0], -1).float()
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = a.norm(dim=1) * b.norm(dim=1) + eps
    return (num / den).mean()
