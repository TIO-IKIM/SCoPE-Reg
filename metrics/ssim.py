"""SSIM — structural similarity between the frame and the resampled slice.

Image-space counterpart to the pose metrics: resample the volume at the
PREDICTED pose and compare that slice with the input frame. Same comparison
`ncc` makes, a different similarity measure — SSIM is sensitive to local
structure rather than global intensity correlation. Range [-1, 1], higher is
better.

Takes the images, not the pose, so it stays a pure metric: use
`_geometry.sample_slice(volume, pose, frame.shape[-2:])` to produce the
resampled slice.
"""
from ._geometry import as_4d


def ssim(a, b, data_range=1.0, win_size=11):
    """
    a, b : (B,1,H,W) — typically the resampled slice and the input frame.
           5D (B,1,1,H,W) is accepted and squeezed.
    data_range : intensity span; 1.0 because the pools are normalised to [0,1].
    win_size   : SSIM window, shrunk automatically for small frames.
    """
    from pytorch_msssim import ssim as _ssim
    a, b = as_4d(a).float(), as_4d(b).float()
    # The window must be odd and fit inside the image.
    win = min(win_size, (min(a.shape[-2:]) // 2) * 2 - 1)
    return _ssim(a, b, data_range=data_range, size_average=True, win_size=win)
