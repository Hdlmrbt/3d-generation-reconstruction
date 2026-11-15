import torch
from src.losses import chamfer

@torch.no_grad()
def pairwise_cd(A, B, chunk=100):
    out = torch.zeros(len(A), len(B))
    for i in range(len(A)):
        for j in range(0, len(B), chunk):
            b = B[j:j + chunk]
            out[i, j:j + len(b)] = chamfer(A[i:i + 1].expand(len(b), -1, -1), b).cpu()
    return out

def mmd_cov(gen, ref):
    """MMD : plus bas = mieux (qualité). COV : plus haut = mieux (diversité)."""
    D = pairwise_cd(gen, ref)                      # (n_gen, n_ref)
    mmd = D.min(0)[0].mean().item()
    cov = len(set(D.argmin(1).tolist())) / len(ref)
    return mmd, cov