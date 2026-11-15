import torch

def chamfer(a, b):
    """a, b : (B, N, 3). Chamfer Distance symétrique (carrée)."""
    d = torch.cdist(a, b)                       # (B, N, M)
    return (d.min(2)[0] ** 2).mean(1) + (d.min(1)[0] ** 2).mean(1)