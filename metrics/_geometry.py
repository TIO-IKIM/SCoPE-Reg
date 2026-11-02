import torch

# CAMUS frames are 114x114 regardless of pixel resolution — the corner size the
# reference mTRE is defined on.
CORNER_FRAC = 114.0 / 128.0        # 114-voxel span on a 128px frame
CORNER_HALF_SIZE = 114.0 / 2.0     # 57 voxels -- legacy scalar, 128px frames only
# Voxel->mm default for reporting mTRE/TransErr in mm. The pool loader
# supplies the variant's real spacing; this is only the reference default.
CAMUS_SPACING_MM = 0.62

# Axis-reversal between the repo's canonical (z,y,x) frame and the (x,y,z)
# frame the reference implementation's Euler decomposition assumes. P = P^-1.
_P_ZYX_XYZ = ((0., 0., 1.), (0., 1., 0.), (1., 0., 0.))


def to_matrix(pose):
    """Accept a 9D pose (B,9) or a 4x4 transform (B,4,4); return (B,4,4).

    Lets every metric take either representation: the training loop works in
    9D, evaluate.py in 4x4.
    """
    if pose.dim() == 3 and pose.shape[-2:] == (4, 4):
        return pose
    if pose.dim() != 2 or pose.shape[1] != 9:
        raise ValueError(f"expected (B,9) pose or (B,4,4) transform, got {tuple(pose.shape)}")
    t = pose[:, :3]
    R = _rot6d_to_matrix(pose[:, 3:9])
    T = torch.eye(4, device=pose.device, dtype=pose.dtype).repeat(pose.shape[0], 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T


def _rot6d_to_matrix(r6):
    """(B,6) continuous 6D rotation -> (B,3,3) via Gram-Schmidt."""
    a1, a2 = r6[:, 0:3], r6[:, 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(1, keepdim=True) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=-1)


def to_xyz(T):
    """(B,4,4) canonical (z,y,x) -> R (B,3,3), t (B,3) in the (x,y,z) frame."""
    P = torch.tensor(_P_ZYX_XYZ, device=T.device, dtype=T.dtype)
    return P @ T[:, :3, :3] @ P, T[:, :3, 3].flip(1)


def matrix_to_euler_deg(R):
    """(B,3,3) -> (B,3) intrinsic XYZ Euler angles in degrees."""
    sy = -R[:, 2, 0].clamp(-1.0, 1.0)
    ry = torch.asin(sy)
    cy = torch.cos(ry)
    safe = cy.abs() > 1e-6
    rx = torch.where(safe, torch.atan2(R[:, 2, 1], R[:, 2, 2]),
                     torch.atan2(-R[:, 1, 2], R[:, 1, 1]))
    rz = torch.where(safe, torch.atan2(R[:, 1, 0], R[:, 0, 0]), torch.zeros_like(ry))
    return torch.stack([rx, ry, rz], dim=-1) * (180.0 / torch.pi)


def to_dof6(pose):
    """Pose (9D or 4x4) -> [tx,ty,tz, rx,ry,rz(deg)] in the reference frame.

    Inverts the reference dof2mat exactly, so for a GT built from dof2mat(param)
    this recovers `param` up to float error — required wherever numeric parity
    with the published numbers is claimed."""
    R, t = to_xyz(to_matrix(pose))
    return torch.cat([t, matrix_to_euler_deg(R)], dim=1)


def sample_slice(volume, pose, out_hw, mode='bilinear'):
    """Resample `volume` at `pose` exactly like the FrameRigidTransformer that
    generated the dataset frames (centred voxel grid, align_corners=True).

    volume (B,1,D,H,W), pose (B,9) or (B,4,4), out_hw (Hs,Ws) -> (B,1,Hs,Ws)
    """
    B, _, D, H, W = volume.shape
    Hs, Ws = out_hw
    dev, dt = volume.device, volume.dtype
    R, t = to_xyz(to_matrix(pose).to(dt))

    ys = torch.linspace(-0.5 * (Hs - 1), 0.5 * (Hs - 1), Hs, device=dev, dtype=dt)
    xs = torch.linspace(-0.5 * (Ws - 1), 0.5 * (Ws - 1), Ws, device=dev, dtype=dt)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    g = torch.stack([gx, gy, torch.zeros_like(gx)], 0).reshape(3, -1)   # (x,y,z=0)
    new = R @ g.unsqueeze(0).expand(B, -1, -1) + t.unsqueeze(-1)

    # Same normalisation as the FRT: 2*((v + 0.5*(S-1))/(S-1) - 0.5) == v/((S-1)/2)
    grid = torch.stack([new[:, 0] / ((W - 1) / 2.0),
                        new[:, 1] / ((H - 1) / 2.0),
                        new[:, 2] / ((D - 1) / 2.0)], -1).view(B, 1, Hs, Ws, 3)
    return torch.nn.functional.grid_sample(
        volume, grid, mode=mode, align_corners=True).squeeze(2)


def as_4d(x):
    """(B,1,1,H,W) -> (B,1,H,W); leaves 4D untouched."""
    return x.squeeze(2) if x.dim() == 5 else x


def corner_half_size(hw):
    """Frame-relative mTRE landmark offsets -> (half_y, half_x) in voxels.


        CAMUS 128x128 -> (57.0, 57.0)   identical to the hardcoded value, so
                                        every CAMUS number ever computed here,
                                        and comparability with the published
                                        published mTRE, is unchanged.
        ProReg 80x96  -> (35.6, 42.8)   the constant put the landmarks at 57,
                                        i.e. OUTSIDE an 80x96 frame (corners at
                                        40/48), inflating the rotation lever by
                                        1.29x for every method.
    """
    h, w = hw
    return (CORNER_FRAC * h / 2.0, CORNER_FRAC * w / 2.0)


def slice_landmarks(half_size=CORNER_HALF_SIZE, device=None, dtype=torch.float32):
    """The 5 mTRE landmarks (4 frame corners + centre), stacked in (z,y,x).

    half_size: a scalar (square frame, legacy) or a (half_y, half_x) pair, as
    returned by corner_half_size() for a non-square frame.
    """
    try:
        hy, hx = half_size
    except TypeError:
        hy = hx = half_size
    return torch.tensor([
        [0., -hy, -hx],
        [0., -hy,  hx],
        [0.,  hy, -hx],
        [0.,  hy,  hx],
        [0.,  0.,  0.],
    ], device=device, dtype=dtype)


def slice_grid(hs, ws, device=None, dtype=torch.float32):
    """(N, 3) centred slice-plane grid points (z=0, y, x) — the TRE targets."""
    ys = torch.arange(hs, device=device, dtype=dtype) - (hs - 1) / 2
    xs = torch.arange(ws, device=device, dtype=dtype) - (ws - 1) / 2
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    gz = torch.zeros_like(gy)
    return torch.stack([gz, gy, gx], -1).reshape(-1, 3)


def transform_points(T, pts):
    """Apply (B,4,4) transforms to (N,3) points -> (B,N,3): R @ p + t."""
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    return torch.einsum('nc,bdc->bnd', pts, R) + t.unsqueeze(1)
