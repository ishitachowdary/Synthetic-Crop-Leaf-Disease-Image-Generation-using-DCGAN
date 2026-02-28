import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, channels=3):
        super().__init__()

        self.net = nn.Sequential(
            # (N, 3, 64, 64) → (N, 64, 32, 32)
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (N, 64, 32, 32) → (N, 128, 16, 16)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # (N, 128, 16, 16) → (N, 256, 8, 8)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # (N, 256, 8, 8) → (N, 512, 4, 4)
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # (N, 512, 4, 4) → (N, 1, 1, 1)
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)
