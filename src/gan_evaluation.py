import torch
import pandas as pd
from pathlib import Path

from generator import Generator
from utils.metrics import inception_score
from utils.config import Config


def evaluate_gan():
    config = Config(
        "configs/data_config.yaml",
        "configs/train_config.yaml"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    generator = Generator(
        latent_dim=config.train["dcgan"]["latent_dim"]
    ).to(device)

    g_path = config.checkpoints_dir / "G_epoch_150.pth"
    generator.load_state_dict(torch.load(g_path, map_location=device))
    generator.eval()

    num_images = 500
    noise = torch.randn(
        num_images,
        config.train["dcgan"]["latent_dim"],
        1,
        1,
        device=device
    )

    with torch.no_grad():
        fake_images = generator(noise)

    is_mean, is_std = inception_score(fake_images, device)

    print(f"Inception Score: {is_mean:.3f} ± {is_std:.3f}")

    results = pd.DataFrame([{
        "InceptionScore_mean": is_mean,
        "InceptionScore_std": is_std,
        "num_images": num_images
    }])

    eval_dir = Path("logs")
    eval_dir.mkdir(exist_ok=True)

    results.to_csv(eval_dir / "gan_inception_score.csv", index=False)
    print("GAN evaluation saved")


if __name__ == "__main__":
    evaluate_gan()
