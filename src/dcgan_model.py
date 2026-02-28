import torch
import torch.nn as nn
import torch.optim as optim


class DCGAN:

    def __init__(self, generator, discriminator, config, device):
        self.device = device

        self.G = generator.to(device)
        self.D = discriminator.to(device)

        self.criterion = nn.BCELoss()

        self.g_optimizer = optim.Adam(
            self.G.parameters(),
            lr=config["optimizer"]["lr"],
            betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"])
        )

        self.d_optimizer = optim.Adam(
            self.D.parameters(),
            lr=config["optimizer"]["lr"],
            betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"])
        )

        self.latent_dim = config["dcgan"]["latent_dim"]

    def save(self, g_path, d_path):
        torch.save(self.G.state_dict(), g_path)
        torch.save(self.D.state_dict(), d_path)
