"""RotErr — rotation error in degrees.

Two definitions, both kept because they answer different questions:

  mode='euler' (default)
      Per-sample L2 distance between the Euler angle triples, averaged over the
      batch — the reference definition (plain L2, NOT SmoothL1), so the number
      is directly comparable to the published tables. Euler-convention specific.

  mode='geodesic'
      Angle of the relative rotation,
          theta = arccos((trace(R_pred^T R_gt) - 1) / 2),
      convention-free and therefore the sounder measure when comparing models
      that parameterise rotation differently.

Never scaled by spacing — this is degrees, not millimetres.
"""
import torch

from ._geometry import to_matrix, to_dof6


def rot_err(pred, gt, mode='euler', reduction='mean'):
    """
    pred, gt : (B,9) poses or (B,4,4) transforms, canonical (z,y,x) frame.
    Returns scalar mean (default) or per-sample (B,) tensor, in degrees.
    """
    if mode == 'euler':
        p, g = to_dof6(pred), to_dof6(gt)
        per_sample = (p[:, 3:] - g[:, 3:]).norm(dim=1)
    elif mode == 'geodesic':
        Rp = to_matrix(pred)[:, :3, :3]
        Rg = to_matrix(gt)[:, :3, :3]
        Rrel = torch.bmm(Rp.transpose(1, 2), Rg)
        trace = Rrel.diagonal(dim1=1, dim2=2).sum(-1)
        cos = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
        per_sample = torch.rad2deg(torch.arccos(cos))
    else:
        raise ValueError(f"mode must be 'euler' or 'geodesic', got '{mode}'")
    return per_sample.mean() if reduction == 'mean' else per_sample
