import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, latent_dim=100, channels=3):
        super().__init__()

        self.net = nn.Sequential(
            # (N, latent_dim, 1, 1) → (N, 512, 4, 4)
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # (N, 512, 4, 4) → (N, 256, 8, 8)
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # (N, 256, 8, 8) → (N, 128, 16, 16)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # (N, 128, 16, 16) → (N, 64, 32, 32)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # (N, 64, 32, 32) → (N, channels, 64, 64)
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)
