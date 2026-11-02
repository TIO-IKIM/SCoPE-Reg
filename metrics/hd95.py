"""HD95 — 95th-percentile symmetric Hausdorff surface distance (millimetres).

Robust boundary-distance metric: the 95th percentile of the two-way distances
between the pred and target mask contours, times the pixel spacing. Less
sensitive to single outlier points than the full Hausdorff distance.

Uses scipy's Euclidean distance transform. Samples with an empty pred or target
mask return NaN and are ignored in the batch mean.
"""
import numpy as np

try:
    from scipy.ndimage import distance_transform_edt, binary_erosion
    _HAS_SCIPY = True
except ImportError:                                                # pragma: no cover
    _HAS_SCIPY = False


def _surface(mask):
    """1-pixel-thick boundary of a binary 2D mask."""
    return mask ^ binary_erosion(mask)


def _hd95_single(p, t, spacing):
    if not p.any() or not t.any():
        return np.nan
    sp, st = _surface(p), _surface(t)
    d_pred_to_t = distance_transform_edt(~st) * spacing            # dist to target surface
    d_t_to_pred = distance_transform_edt(~sp) * spacing
    dists = np.concatenate([d_pred_to_t[sp], d_t_to_pred[st]])
    return float(np.percentile(dists, 95))


def hd95(pred, target, spacing=0.62, threshold=0.5):
    """
    pred, target : mask tensors (B, 1, H, W) or (B, H, W). Returns the batch-mean
    HD95 in millimetres (NaN samples — empty masks — are skipped).
    """
    if not _HAS_SCIPY:
        raise ImportError("hd95 needs scipy (scipy.ndimage).")
    p = (pred > threshold).detach().cpu().numpy().astype(bool)
    t = (target > threshold).detach().cpu().numpy().astype(bool)
    p = p.reshape(p.shape[0], *p.shape[-2:])
    t = t.reshape(t.shape[0], *t.shape[-2:])
    vals = [_hd95_single(p[i], t[i], spacing) for i in range(p.shape[0])]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float('nan')
