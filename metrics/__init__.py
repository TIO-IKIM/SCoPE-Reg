from .dice import dice
from .hd95 import hd95
from .ncc import ncc
from .mtre import mtre
from .rot_err import rot_err
from .trans_err import trans_err
from .ssim import ssim
from .params import count_params
from .flops import gflops
from .fps import fps
from .vram import max_vram_mb
from ._geometry import (
    slice_landmarks, slice_grid, transform_points,
    to_matrix, to_dof6, to_xyz, matrix_to_euler_deg, sample_slice, as_4d,
    CORNER_HALF_SIZE, CORNER_FRAC, corner_half_size, CAMUS_SPACING_MM,
)

# Metric groupings — handy for building report tables.
SEG_METRICS = ('dice', 'hd95')
POSE_METRICS = ('mtre', 'trans_err', 'rot_err')
IMG_METRICS = ('ncc', 'ssim')
COST_METRICS = ('count_params', 'gflops', 'fps', 'max_vram_mb')

__all__ = [
    'dice', 'hd95', 'ncc', 'ssim', 'mtre', 'trans_err', 'rot_err',
    'count_params', 'gflops', 'fps', 'max_vram_mb',
    'slice_landmarks', 'slice_grid', 'transform_points',
    'to_matrix', 'to_dof6', 'to_xyz', 'matrix_to_euler_deg',
    'sample_slice', 'as_4d',
    'CORNER_HALF_SIZE', 'CORNER_FRAC', 'corner_half_size', 'CAMUS_SPACING_MM',
    'SEG_METRICS', 'POSE_METRICS', 'COST_METRICS',
]