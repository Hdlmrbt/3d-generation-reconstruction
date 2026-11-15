import torch
import torch.nn as nn

class PointNetAE(nn.Module):
    def __init__(self, latent=128, n_points=1024):
        super().__init__()
        self.n = n_points
        self.enc = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512), nn.ReLU())
        self.to_z = nn.Linear(512, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, n_points * 3))

    def encode(self, x):                     # x: (B, N, 3)
        h = self.enc(x.transpose(1, 2)).max(dim=2)[0]
        return self.to_z(h)

    def decode(self, z):
        return self.dec(z).view(-1, self.n, 3)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z
    
    @torch.no_grad()
    def sample(self, n, device):
        return self.decode(torch.randn(n, self.latent, device=device))