"""Inference throughput (frames per second)."""
import time
import torch


def fps(model, inputs, n_iters=50, warmup=10, device=None):
    """
    model  : nn.Module (set to eval()).
    inputs : tuple of example forward arguments.
    Returns frames/second = (n_iters * batch_size) / elapsed, averaged over
    `n_iters` timed forwards after `warmup` un-timed ones. CUDA is synchronised
    around the timed region so the number reflects real device time.
    """
    is_cuda = device is not None and 'cuda' in str(device)
    model.eval()
    batch = inputs[0].shape[0]
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            model(*inputs)
        if is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return n_iters * batch / elapsed
