"""Dense supervision of the coordinate field.

Follows the coarse-flow loss of EUReg (Hui et al., MICCAI 2025), Eqs. (4)-(5):
phi is supervised against phi*, built from the ground-truth rigid transform on
the slice identity grid, with a gradient-magnitude term alongside the direct
SmoothL1.

phi and phi* must share a frame and units: the fusion block emits phi in
centered grid units at the feature resolution, channel order (z, y, x), so the
GT translation is divided by the encoder total stride (TRANS_SCALE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder total stride: phi lives on the feature grid, the GT translation
# in volume voxels, so phi* divides by this. Derived from the fixed
# encoder, not a free parameter.
TRANS_SCALE = 8.0


def _slice_identity_grid(Hs, Ws, device, dtype=torch.float32):
    """(3, Hs, Ws) centered identity grid, order (z=0, y=h, x=w)."""
    ys = torch.arange(Hs, device=device, dtype=dtype) - (Hs - 1) / 2
    xs = torch.arange(Ws, device=device, dtype=dtype) - (Ws - 1) / 2
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    gz = torch.zeros_like(gy)
    return torch.stack([gz, gy, gx], 0)


def build_phi_star(transform, Hs, Ws, trans_scale=TRANS_SCALE, device=None):
    """
    Build the GT coarse flow phi*.

    transform : (B, 4, 4) GT homogeneous rigid transform (R*, T*), in the SAME
                frame/order (z,y,x) as the fusion block's coordinate grid.
    Hs, Ws    : slice feature-map size (the size of phi).
    trans_scale: divide the translation part by this to bring it to the feature
                grid resolution (e.g. output_stride if T* is in volume voxels
                and phi is at the down-sampled feature grid). Default 1.0.

    Returns phi_star : (B, 3, Hs, Ws).
    """
    device = device or transform.device
    B = transform.shape[0]

    gs = _slice_identity_grid(Hs, Ws, device)                 # (3, Hs, Ws)
    gs_flat = gs.reshape(3, -1)                                # (3, N)
    ones = torch.ones(1, gs_flat.shape[1], device=device)
    gs_h = torch.cat([gs_flat, ones], 0).unsqueeze(0).expand(B, -1, -1)  # (B,4,N)

    A = transform.clone()
    A[:, :3, 3] = A[:, :3, 3] / trans_scale                   # T*_down

    I = torch.eye(4, device=device).unsqueeze(0)
    phi = torch.bmm(A - I, gs_h)[:, :3]                        # (B, 3, N)
    return phi.reshape(B, 3, Hs, Ws)


def _grad_mag(x):
    """Per-sample spatial gradient magnitude, (B,3,H,W) -> scalar-ish per (B,3)."""
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=(2, 3))   # (B,3)
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=(2, 3))   # (B,3)
    return dh + dw


class FlowLoss(nn.Module):
    """
    SmoothL1(phi - phi*) + w_grad * SmoothL1(|grad phi| - |grad phi*|).
    """

    def __init__(self, w_grad=1.0):
        super().__init__()
        self.w_grad = w_grad

    def forward(self, phi, phi_star):
        l_val = F.smooth_l1_loss(phi, phi_star)
        l_grad = F.smooth_l1_loss(_grad_mag(phi), _grad_mag(phi_star))
        return l_val + self.w_grad * l_grad


def flow_loss(phi, phi_star, w_grad=1.0):
    """Functional form of FlowLoss."""
    l_val = F.smooth_l1_loss(phi, phi_star)
    l_grad = F.smooth_l1_loss(_grad_mag(phi), _grad_mag(phi_star))
    return l_val + w_grad * l_grad


# ---------------------------------------------------------------------
# Helper: frame conversion between (x,y,z) and canonical (z,y,x)
# ---------------------------------------------------------------------
def transform_xyz_to_zyx(T):
    """
    Conjugate a homogeneous 4x4 transform acting on (x,y,z,1)-ordered vectors
    (the dof2mat / FrameRigidTransformer convention) into one acting on
    (z,y,x,1)-ordered vectors — the canonical frame of every grid in this
    repo (coord fusion, build_phi_star, procrustes):  M_zyx = P M_xyz P.

    Required before feeding a dof2mat-derived transform into build_phi_star:
    without it, a pure in-plane rotation produces phi* with its largest
    component in the through-plane channel (see verify_phi_convention.py).
    The same conjugation maps back (P is its own inverse).
    """
    perm = torch.tensor([2, 1, 0, 3], device=T.device)
    return T[:, perm][:, :, perm]


# ---------------------------------------------------------------------
# Helper: 9D pose -> 4x4 transform, for GT in 9D [t3, 6d-rot]
# ---------------------------------------------------------------------
def pose9_to_transform4x4(pose9):
    """
    (B,9) [tx,ty,tz, 6d-rot] -> (B,4,4). Uses Gram-Schmidt on the 6D rotation.
    The resulting frame/order must match the coordinate grid (z,y,x); a 6D
    rotation in (x,y,z) column order needs permuting first.
    """
    t = pose9[:, :3]
    x_raw, y_raw = pose9[:, 3:6], pose9[:, 6:9]
    x = F.normalize(x_raw, dim=1, eps=1e-6)
    y = y_raw - (x * y_raw).sum(1, keepdim=True) * x
    y = F.normalize(y, dim=1, eps=1e-6)
    z = torch.linalg.cross(x, y, dim=1)
    R = torch.stack([x, y, z], dim=-1)                         # (B,3,3)
    B = pose9.shape[0]
    M = torch.eye(4, device=pose9.device).unsqueeze(0).repeat(B, 1, 1)
    M[:, :3, :3] = R
    M[:, :3, 3] = t
    return M