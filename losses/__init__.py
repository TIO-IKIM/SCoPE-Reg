"""Training objectives.

    pose     landmark distance between the predicted and true pose (always on)
    coord    optional dense supervision on the coordinate field
    physics  optional Navier-Cauchy residual on the coordinate field
"""
from .pose import PoseLoss
from .flow import (FlowLoss, build_phi_star, flow_loss,
                    pose9_to_transform4x4, transform_xyz_to_zyx)
from .physics import PhysicsLoss, navier_cauchy_residual
from .utils import euler6_to_9d, matrix_to_9d, pose_9d_to_transform

LOSS_REGISTRY = {"pose": PoseLoss}


def build_loss(name="pose", **kwargs):
    if name not in LOSS_REGISTRY:
        raise ValueError(f"unknown loss '{name}'; available: {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](**kwargs)


__all__ = [
    "PoseLoss", "FlowLoss", "PhysicsLoss",
    "build_loss", "LOSS_REGISTRY",
    "build_phi_star", "flow_loss", "navier_cauchy_residual",
    "pose9_to_transform4x4", "transform_xyz_to_zyx", "pose_9d_to_transform",
    "euler6_to_9d", "matrix_to_9d",
]
