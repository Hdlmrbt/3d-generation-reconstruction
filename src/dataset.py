import numpy as np, torch, trimesh
from pathlib import Path
from torch.utils.data import Dataset

def off_to_points(path, n=1024):
    mesh = trimesh.load(path, force="mesh")
    pts = mesh.sample(n)
    pts = pts - pts.mean(0)
    pts = pts / np.linalg.norm(pts, axis=1).max()   # sphère unité
    return pts.astype(np.float32)

def build_cache(root, classes, split, n=1024, max_per_class=300, out="data/processed"):
    Path(out).mkdir(parents=True, exist_ok=True)
    X, y = [], []
    for ci, c in enumerate(classes):
        files = sorted((Path(root) / c / split).glob("*.off"))[:max_per_class]
        for f in files:
            X.append(off_to_points(f, n)); y.append(ci)
    np.savez(f"{out}/{split}_{n}.npz", X=np.stack(X), y=np.array(y))

class PointCloudDataset(Dataset):
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.X, self.y = torch.from_numpy(d["X"]), torch.from_numpy(d["y"])
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]