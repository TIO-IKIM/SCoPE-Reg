"""Pose conversions shared by the losses and the training loop."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


_REF_POINTS = torch.tensor([
	[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
	[1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
	[0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0],
], dtype=torch.float32)


def compute_rotation_matrix_from_ortho6d(ortho6d):
	"""(B,6) -> (B,3,3) via Gram-Schmidt on the first two columns."""
	x_raw, y_raw = ortho6d[:, 0:3], ortho6d[:, 3:6]
	x = F.normalize(x_raw, dim=1, eps=1e-6)
	y = y_raw - (x * y_raw).sum(dim=1, keepdim=True) * x
	y = F.normalize(y, dim=1, eps=1e-6)
	z = torch.linalg.cross(x, y, dim=1)
	return torch.stack((x, y, z), dim=-1)


def pose_9d_to_transform(pose_9d):
	"""(B,9) -> R (B,3,3), t (B,3)."""
	t = pose_9d[:, :3]
	R = compute_rotation_matrix_from_ortho6d(pose_9d[:, 3:9])
	return R, t


def geodesic_distance_so3(R_pred, R_gt, eps=1e-6):
	"""Per-sample geodesic distance on SO(3), in radians."""
	R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
	trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
	cos_theta = (trace - 1.0) * 0.5
	rx = R_rel[:, 2, 1] - R_rel[:, 1, 2]
	ry = R_rel[:, 0, 2] - R_rel[:, 2, 0]
	rz = R_rel[:, 1, 0] - R_rel[:, 0, 1]
	sin_theta = 0.5 * torch.sqrt(rx * rx + ry * ry + rz * rz + eps)
	return torch.atan2(sin_theta, cos_theta).abs()


def _metrics(pred9, gt9, total, scale_factor=1.0, **extra):
	R_pred, t_pred = pose_9d_to_transform(pred9)
	R_gt, t_gt = pose_9d_to_transform(gt9)
	pts = (_REF_POINTS.to(pred9.device)) * scale_factor
	pred_pts = torch.einsum('nc,bdc->bnd', pts, R_pred) + t_pred.unsqueeze(1)
	gt_pts = torch.einsum('nc,bdc->bnd', pts, R_gt) + t_gt.unsqueeze(1)
	dist = (pred_pts - gt_pts).norm(dim=-1).mean().item()
	rot_deg = torch.rad2deg(geodesic_distance_so3(R_pred, R_gt)).mean().item()
	trans_l2 = (t_pred - t_gt).norm(dim=1).mean().item()
	det = {'total': float(total), 'dist': dist,
		   'rot_deg': rot_deg, 'trans_l2': trans_l2}
	det.update(extra)
	return det


def euler_xyz_to_matrix(e):
	"""(B,3) intrinsic XYZ Euler in RAD -> (B,3,3)."""
	cx, cy, cz = torch.cos(e[:, 0]), torch.cos(e[:, 1]), torch.cos(e[:, 2])
	sx, sy, sz = torch.sin(e[:, 0]), torch.sin(e[:, 1]), torch.sin(e[:, 2])
	R = torch.zeros(e.shape[0], 3, 3, device=e.device, dtype=e.dtype)
	R[:, 0, 0] = cy * cz
	R[:, 0, 1] = cz * sx * sy - cx * sz
	R[:, 0, 2] = cx * cz * sy + sx * sz
	R[:, 1, 0] = cy * sz
	R[:, 1, 1] = cx * cz + sx * sy * sz
	R[:, 1, 2] = -cz * sx + cx * sy * sz
	R[:, 2, 0] = -sy
	R[:, 2, 1] = cy * sx
	R[:, 2, 2] = cx * cy
	return R


def matrix_to_9d(M):
	"""(B,4,4)/(B,3,4) rigid -> 9D [tx,ty,tz, R_col0, R_col1]."""
	R, t = M[:, :3, :3], M[:, :3, 3]
	return torch.cat([t, R[:, :, 0], R[:, :, 1]], dim=1)


def euler6_to_9d(p, degrees=False):
	"""[tx,ty,tz, rx,ry,rz] -> 9D [tx,ty,tz, 6d-rot]."""
	t = p[:, :3]
	rot = p[:, 3:6]
	if degrees:
		rot = rot * (torch.pi / 180.0)
	R = euler_xyz_to_matrix(rot)
	return torch.cat([t, R[:, :, 0], R[:, :, 1]], dim=1)
