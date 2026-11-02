"""Parameter count of a model."""


def count_params(model):
    """Returns {'total', 'trainable', 'total_millions'} parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable, 'total_millions': total / 1e6}
