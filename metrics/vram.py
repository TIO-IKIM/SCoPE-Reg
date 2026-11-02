"""Peak VRAM usage of a single inference forward (megabytes).

Returns None on CPU (no CUDA allocator to query).
"""
import torch


def max_vram_mb(model, inputs, device):
    """
    model  : nn.Module (set to eval()).
    inputs : tuple of example forward arguments (already on `device`).
    Returns peak CUDA memory allocated during one forward, in MiB, or None on CPU.
    """
    if device is None or 'cuda' not in str(device):
        return None
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    with torch.no_grad():
        model(*inputs)
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
