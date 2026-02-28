import os
import torch
import numpy as np
from tqdm import tqdm

from generator import Generator
from discriminator import Discriminator
from dcgan_model import DCGAN
from data_loader import get_dataloader
from utils.config import Config
from utils.logger import Logger


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


def train():
    config = Config(
        "configs/data_config.yaml",
        "configs/train_config.yaml"
    )
    logger = Logger(log_dir="logs")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataloader = get_dataloader(
        split_dir=config.data_dir / "Train",
        image_size=config.data["image"]["size"],
        batch_size=config.train["dcgan"]["batch_size"]
    )

    G = Generator(
        latent_dim=config.train["dcgan"]["latent_dim"]
    )
    D = Discriminator()

    G.apply(weights_init)
    D.apply(weights_init)

    dcgan = DCGAN(G, D, config.train, device)

    fixed_noise = torch.randn(
        64, dcgan.latent_dim, 1, 1, device=device
    )
    logger.save_metadata({
        "model": "DCGAN",
        "latent_dim": config.train["dcgan"]["latent_dim"],
        "batch_size": config.train["dcgan"]["batch_size"],
        "epochs": config.train["dcgan"]["epochs"],
        "image_size": config.data["image"]["size"],
        "device": str(device)
    })

    for epoch in range(1, config.train["dcgan"]["epochs"] + 1):
        g_losses = []
        d_losses = []

        for real_images in tqdm(dataloader, desc=f"Epoch {epoch}"):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

    
            dcgan.D.zero_grad()

            real_labels = torch.full(
                (batch_size, 1), 0.9, device=device
            )
            fake_labels = torch.zeros(
                (batch_size, 1), device=device
            )

            output_real = dcgan.D(real_images)
            d_loss_real = dcgan.criterion(output_real, real_labels)

            noise = torch.randn(
                batch_size, dcgan.latent_dim, 1, 1, device=device
            )
            fake_images = dcgan.G(noise)
            output_fake = dcgan.D(fake_images.detach())
            d_loss_fake = dcgan.criterion(output_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            dcgan.d_optimizer.step()

  
            dcgan.G.zero_grad()

            gen_labels = torch.ones(
                (batch_size, 1), device=device
            )
            output = dcgan.D(fake_images)
            g_loss = dcgan.criterion(output, gen_labels)
            g_loss.backward()
            dcgan.g_optimizer.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        print(
            f"[Epoch {epoch}/{config.train['dcgan']['epochs']}] "
            f"D Loss: {np.mean(d_losses):.4f} | "
            f"G Loss: {np.mean(g_losses):.4f}"
        )
        logger.log_training(
            epoch=epoch,
            d_loss=np.mean(d_losses),
            g_loss=np.mean(g_losses)
        )

        if epoch % config.train["logging"]["save_checkpoints_every"] == 0:
            dcgan.save(
                config.checkpoints_dir / f"G_epoch_{epoch:03d}.pth",
                config.checkpoints_dir / f"D_epoch_{epoch:03d}.pth"
            )

    print("DCGAN Training Completed")


if __name__ == "__main__":
    train()
