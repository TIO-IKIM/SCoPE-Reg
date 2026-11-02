import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import _REF_POINTS, _metrics, pose_9d_to_transform


# Half-extent of the reference-point cloud, in encoded voxels. It fixes the
# units the pose loss is measured in; the tunable quantity is the weight of
# each loss, not the scale inside one.
SCALE = 43.0


class PoseLoss(nn.Module):
    """MSE between the predicted and true pose, measured on a fixed point cloud.

    The cloud is 14 points (8 cube corners + 6 face centres) at +-SCALE encoded
    voxels, so the loss penalises rotation and translation in one term and in
    the same units. This is the training objective; the reported mTRE
    metric uses the 5 frame landmarks in metrics/_geometry.py instead.
    """

    def __init__(self):
        super().__init__()
        self.scale_factor = SCALE

    def forward(self, pred9, gt9, volume=None, query_slice=None):
        R_pred, t_pred = pose_9d_to_transform(pred9)
        R_gt, t_gt = pose_9d_to_transform(gt9)
        pts = _REF_POINTS.to(pred9.device) * self.scale_factor
        pred_pts = torch.einsum('nc,bdc->bnd', pts, R_pred) + t_pred.unsqueeze(1)
        gt_pts = torch.einsum('nc,bdc->bnd', pts, R_gt) + t_gt.unsqueeze(1)
        loss = F.mse_loss(pred_pts, gt_pts)
        return loss, _metrics(pred9, gt9, loss.item(), scale_factor=self.scale_factor)
