"""TransErr — translation error in millimetres.

Per-sample L2 distance between the predicted and GT translation, averaged over
the batch. This is the reference definition (plain L2, NOT SmoothL1), so the
number is directly comparable to the published tables.

The translation is read in the (x,y,z) reference frame via `to_dof6`, which
inverts the dataset's dof2mat exactly — so for a GT built by the preprocessing
this recovers the sampled translation up to float error.
"""
from ._geometry import to_dof6, CAMUS_SPACING_MM


def trans_err(pred, gt, spacing=CAMUS_SPACING_MM, reduction='mean'):
    """
    pred, gt : (B,9) poses or (B,4,4) transforms, canonical (z,y,x) frame.
    spacing  : mm per voxel. Pass the pool variant's own spacing.
    Returns scalar mean (default) or per-sample (B,) tensor, in millimetres.
    """
    p, g = to_dof6(pred), to_dof6(gt)
    per_sample = (p[:, :3] - g[:, :3]).norm(dim=1) * spacing
    return per_sample.mean() if reduction == 'mean' else per_sample
