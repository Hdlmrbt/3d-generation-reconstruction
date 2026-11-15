import torch
import torch.nn as nn

class PointNetVAE(nn.Module):
    def __init__(self, latent=128, n_points=1024):
        super().__init__()
        self.n = n_points
        self.enc = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512), nn.ReLU())
        self.fc_mu = nn.Linear(512, latent)
        self.fc_logvar = nn.Linear(512, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, n_points * 3))
        self.latent = latent

    def encode(self, x):
        h = self.enc(x.transpose(1, 2)).max(dim=2)[0]
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.dec(z).view(-1, self.n, 3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, device):
        return self.decode(torch.randn(n, self.latent, device=device))