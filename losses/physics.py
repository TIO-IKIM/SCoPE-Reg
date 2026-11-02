"""Navier-Cauchy regularizer on the predicted coordinate field.

Penalises the PDE residual mu*Lap(phi) + (mu+lam)*grad(div phi). mu is fixed at
1 -- the overall scale is train.lambda_physics -- and the Poisson ratio is the
soft-tissue value. Every nu shares the same affine null space and the target
residual is zero, so nu is a constant here, not something to fit.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

POISSON = 0.45
MU = 1.0
LAM = 2.0 * MU * POISSON / (1.0 - 2.0 * POISSON)


def _pad1(x, dim):
    first = x.narrow(dim, 0, 1)
    last = x.narrow(dim, x.size(dim) - 1, 1)
    return torch.cat([first, x, last], dim)


def _d1(x, dim):
    """Central first difference along spatial dim (replicate boundary)."""
    xp = _pad1(x, dim)
    n = x.size(dim)
    return (xp.narrow(dim, 2, n) - xp.narrow(dim, 0, n)) * 0.5


def _d2(x, dim):
    """Second difference along spatial dim (replicate boundary)."""
    xp = _pad1(x, dim)
    n = x.size(dim)
    return xp.narrow(dim, 2, n) + xp.narrow(dim, 0, n) - 2.0 * x


def _crop(x, n_axes, m=1):
    """
    Drop the m-pixel boundary ring along the last n_axes dims before reducing:
    replicate-padded central differences are only exact in the interior, and
    keeping the ring breaks the exact affine null space of the second-order
    operators (a rigid field would pick up spurious boundary residual).
    """
    if m <= 0:
        return x
    sl = [slice(None)] * x.dim()
    for d in range(x.dim() - n_axes, x.dim()):
        if x.size(d) > 2 * m:
            sl[d] = slice(m, -m)
    return x[tuple(sl)]


def _axes_channels(u):
    """
    Map spatial axes to the channels displacing along them.

    Returns (spatial_dims, axis_channels):
      spatial_dims  : tensor dims of the domain axes, order (z,)y,x
      axis_channels : channel index whose displacement is along that axis
    Out-of-plane channels (C=3 on a 2D domain: channel 0 = z) are handled by
    the callers explicitly.
    """
    if u.dim() == 4:                     # (B, C, H, W): dims (y, x)
        dims = (2, 3)
        if u.shape[1] == 3:
            chans = (1, 2)               # (z, y, x) -> y, x in-plane
        elif u.shape[1] == 2:
            chans = (0, 1)               # (y, x)
        else:
            raise ValueError(f"2D-domain field needs C in (2,3), got {u.shape[1]}")
        return dims, chans
    if u.dim() == 5:                     # (B, 3, D, H, W): dims (z, y, x)
        if u.shape[1] != 3:
            raise ValueError(f"3D-domain field needs C=3, got {u.shape[1]}")
        return (2, 3, 4), (0, 1, 2)
    raise ValueError(f"expected 4D or 5D field, got {u.dim()}D")


def divergence(u):
    """div u = sum_i d u_i / d x_i over domain axes (out-of-plane excluded)."""
    dims, chans = _axes_channels(u)
    return sum(_d1(u[:, c:c + 1], d) for d, c in zip(dims, chans)).squeeze(1)


def laplacian(u):
    """Per-channel Laplacian over domain axes."""
    dims, _ = _axes_channels(u)
    return sum(_d2(u, d) for d in dims)


def navier_cauchy_residual(u, mu=1.0, lam=1.0, reduce=True):
    """
    Residual of the Navier-Cauchy equilibrium equation,

        r = mu * Lap(u) + (mu + lam) * grad(div u),

    grad(div) acting along domain axes (zero for the out-of-plane channel,
    which only feels the Laplacian). Returns mean(r^2) (or the residual field
    with reduce=False). Vanishes exactly on affine fields -> compatible with
    the rigid Procrustes readout by construction.
    """
    dims, chans = _axes_channels(u)
    lap = laplacian(u)
    div = divergence(u).unsqueeze(1)
    r = mu * lap
    gd = torch.zeros_like(u)
    for d, c in zip(dims, chans):
        gd[:, c:c + 1] = _d1(div, d)
    r = r + (mu + lam) * gd
    if reduce:
        # grad(div) composes two first differences -> 2-ring stencil validity
        return _crop(r ** 2, len(dims), 2).mean()
    return r


class PhysicsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, phi):
        loss = navier_cauchy_residual(phi, mu=MU, lam=LAM)
        return loss, {"phys": loss.item()}
