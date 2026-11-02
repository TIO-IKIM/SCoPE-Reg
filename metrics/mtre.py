"""mTRE — mean target registration error, in millimetres (Equal to DistErr in EUReg, DreamReg).

The mean L2 distance between five landmarks (the four frame corners and the
centre) transformed by the predicted pose and by the ground-truth pose, times
the pixel spacing. This is the primary metric the paper reports.
"""
from ._geometry import (slice_landmarks, transform_points, to_matrix,
                        corner_half_size, CORNER_HALF_SIZE, CAMUS_SPACING_MM)


def mtre(pred, gt, spacing=CAMUS_SPACING_MM, half_size=None,
             reduction='mean', hw=None):
    """
    pred, gt     : (B,9) poses or (B,4,4) transforms (canonical z,y,x frame).
    spacing      : mm per voxel (0.62 for CAMUS).
    hw           : (H, W) of the frame. Given, the landmarks are placed at
                   the 114/128 fraction OF THAT FRAME -- exact agreement on
                   128px frames, correct on every other size. Without it the
                   legacy 57-voxel constant is used, which sits outside any
                   frame smaller than 114px.
    Returns scalar mean (default) or per-sample (B,) tensor in millimetres.
    """
    if half_size is None:
        half_size = corner_half_size(hw) if hw is not None else CORNER_HALF_SIZE
    pred_T, gt_T = to_matrix(pred), to_matrix(gt)
    pts = slice_landmarks(half_size, pred_T.device, pred_T.dtype)
    p = transform_points(pred_T, pts)
    g = transform_points(gt_T, pts)
    per_sample = (p - g).norm(dim=-1).mean(dim=1) * spacing        # (B,)
    return per_sample.mean() if reduction == 'mean' else per_sample
