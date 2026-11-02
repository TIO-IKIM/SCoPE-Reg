"""GFLOPs of a single forward pass.

Uses fvcore's FlopCountAnalysis (counts multiply-accumulates; the community
convention printed as "FLOPs"). Falls back to thop if fvcore is missing, and
returns None if neither is available.
"""


def gflops(model, inputs):
    """
    model  : nn.Module (put in eval() by the caller).
    inputs : tuple of example forward arguments, e.g. (vol, frame).
    Returns GFLOPs (float) for one forward on this batch, or None.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(model, inputs)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return fca.total() / 1e9
    except Exception:
        pass
    try:
        from thop import profile
        macs, _ = profile(model, inputs=inputs, verbose=False)
        return macs / 1e9
    except Exception:
        return None
