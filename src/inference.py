import os
import torch
from torchvision import transforms
from PIL import Image

from generator import Generator
from utils.config import Config


def run_inference(num_images=50, output_dir="samples/inference"):
    config = Config(
        "configs/data_config.yaml",
        "configs/train_config.yaml"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    generator = Generator(
        latent_dim=config.train["dcgan"]["latent_dim"]
    ).to(device)

    generator.load_state_dict(
        torch.load(config.checkpoints_dir / "G_epoch_150.pth", map_location=device)
    )
    generator.eval()

    to_pil = transforms.ToPILImage()

    with torch.no_grad():
        for i in range(num_images):
            z = torch.randn(
                1, config.train["dcgan"]["latent_dim"], 1, 1, device=device
            )
            img = generator(z)[0]
            img = (img + 1) / 2  # [-1,1] → [0,1]
            img_pil = to_pil(img.cpu())
            img_pil.save(f"{output_dir}/synthetic_{i:03d}.png")

    print(f"{num_images} synthetic images saved to {output_dir}")


if __name__ == "__main__":
    run_inference()
