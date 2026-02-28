import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torchvision.utils import make_grid
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import models
import pandas as pd
import seaborn as sns

from generator import Generator
from utils.config import Config


# -------------------------------------------------
# Utility: Save image grid
# -------------------------------------------------
def save_image_grid(images, path, nrow=8, title=None):

    images = (images + 1) / 2  # [-1,1] → [0,1]
    grid = make_grid(images, nrow=nrow, padding=2)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# -------------------------------------------------
# GAN Sample Grid
# -------------------------------------------------
def visualize_samples(epoch=150, num_images=64):
    config = Config("configs/data_config.yaml", "configs/train_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(latent_dim=config.train["dcgan"]["latent_dim"]).to(device)
    g_path = config.checkpoints_dir / f"G_epoch_{epoch:03d}.pth"
    generator.load_state_dict(torch.load(g_path, map_location=device))
    generator.eval()

    noise = torch.randn(num_images, config.train["dcgan"]["latent_dim"], 1, 1, device=device)
    with torch.no_grad():
        fake_images = generator(noise)

    out_dir = Path("samples")
    out_dir.mkdir(exist_ok=True)

    save_image_grid(
        fake_images,
        out_dir / f"samples_epoch_{epoch:03d}.png",
        nrow=int(np.sqrt(num_images)),
        title="GAN Generated Leaf Samples"
    )

    print("GAN sample grid saved")


# -------------------------------------------------
# Latent Space Interpolation
# -------------------------------------------------
def latent_interpolation(steps=10):
    config = Config("configs/data_config.yaml", "configs/train_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(latent_dim=config.train["dcgan"]["latent_dim"]).to(device)
    generator.load_state_dict(
        torch.load(config.checkpoints_dir / "G_epoch_150.pth", map_location=device)
    )
    generator.eval()

    z1 = torch.randn(1, config.train["dcgan"]["latent_dim"], 1, 1, device=device)
    z2 = torch.randn(1, config.train["dcgan"]["latent_dim"], 1, 1, device=device)

    images = []
    for alpha in np.linspace(0, 1, steps):
        z = (1 - alpha) * z1 + alpha * z2
        with torch.no_grad():
            images.append(generator(z))

    images = torch.cat(images, dim=0)

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    save_image_grid(
        images,
        out_dir / "latent_interpolation.png",
        nrow=steps,
        title="Latent Space Interpolation"
    )

    print("Latent interpolation saved")


# -------------------------------------------------
# GAN + Classifier Interpretation
# -------------------------------------------------
def visualize_gan_with_classifier(num_images=16):
    config = Config("configs/data_config.yaml", "configs/train_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(latent_dim=config.train["dcgan"]["latent_dim"]).to(device)
    generator.load_state_dict(
        torch.load(config.checkpoints_dir / "G_epoch_150.pth", map_location=device)
    )
    generator.eval()

    dataset = ImageFolder(config.data_dir / "Train")
    class_names = dataset.classes

    classifier = models.resnet18(pretrained=False)
    classifier.fc = torch.nn.Linear(classifier.fc.in_features, len(class_names))
    classifier.load_state_dict(
        torch.load(config.checkpoints_dir / "classifier_augmented.pth", map_location=device)
    )
    classifier.to(device)
    classifier.eval()

    transform = transforms.Resize((224, 224))

    noise = torch.randn(num_images, config.train["dcgan"]["latent_dim"], 1, 1, device=device)

    predictions = []

    plt.figure(figsize=(10, 10))

    with torch.no_grad():
        fake_images = generator(noise)

        for i in range(num_images):
            img = fake_images[i]
            img_norm = (img + 1) / 2
            img_cls = transform(img_norm).unsqueeze(0).to(device)

            logits = classifier(img_cls)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)

            label = class_names[idx.item()]
            predictions.append(label)

            plt.subplot(4, 4, i + 1)
            plt.imshow(img_norm.permute(1, 2, 0).cpu())
            plt.title(f"{label}\n{conf.item()*100:.1f}%", fontsize=8)
            plt.axis("off")

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / "gan_classifier_interpretation.png")
    plt.close()

    print("GAN + classifier interpretation saved")

    return predictions


# -------------------------------------------------
# Distribution of Generated Classes
# -------------------------------------------------
def plot_generated_class_distribution(predictions):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    df = pd.DataFrame(predictions, columns=["Class"])
    counts = df["Class"].value_counts().sort_values()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=counts.values,
        y=counts.index,
        orient="h"
    )

    plt.xlabel("Count")
    plt.ylabel("Predicted Class")
    plt.title("Distribution of Classes Generated by GAN (via Classifier)")
    plt.tight_layout()

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / "gan_class_distribution.png")
    plt.close()

    print("GAN class distribution plot saved")

# -------------------------------------------------
# GAN Training Curves (from CSV logs)
# -------------------------------------------------
def plot_gan_training_curves(
    log_path="training_log.csv",
    out_path="figures/gan_training_curves.png"
):


    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Training log not found: {log_path}")

    df = pd.read_csv(log_path)

    if not {"epoch", "d_loss", "g_loss"}.issubset(df.columns):
        raise ValueError(
            "CSV must contain columns: epoch, d_loss, g_loss"
        )

    plt.figure(figsize=(8, 5))

    plt.plot(df["epoch"], df["d_loss"], label="Discriminator Loss")
    plt.plot(df["epoch"], df["g_loss"], label="Generator Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DCGAN Training Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True)

    plt.savefig(out_path)
    plt.close()

    print(f"GAN training curves saved: {out_path}")

if __name__ == "__main__":
    visualize_samples()
    latent_interpolation()

    preds = visualize_gan_with_classifier(num_images=16)
    plot_generated_class_distribution(preds)

    plot_gan_training_curves(
        log_path="logs/training_log.csv",
        out_path="figures/gan_training_curves.png"
    )
