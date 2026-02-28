import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from generator import Generator
from utils.config import Config


def pseudo_label_images(
    generator,
    teacher_model,
    class_names,
    device,
    out_root,
    num_images=500,
    confidence_threshold=0.75
):


    generator.eval()
    teacher_model.eval()

    latent_dim = generator.net[0].in_channels
    to_pil = transforms.ToPILImage()

    os.makedirs(out_root, exist_ok=True)
    for cname in class_names:
        os.makedirs(os.path.join(out_root, cname), exist_ok=True)

    resize_for_classifier = transforms.Resize((224, 224))

    accepted = 0

    with torch.no_grad():
        for i in range(num_images):
            z = torch.randn(1, latent_dim, 1, 1, device=device)
            img = generator(z)[0]
            img = (img + 1) / 2  # [-1,1] → [0,1]

            input_img = resize_for_classifier(img).unsqueeze(0).to(device)
            logits = teacher_model(input_img)
            probs = torch.softmax(logits, dim=1)[0]

            conf, pred = torch.max(probs, dim=0)

            if conf.item() >= confidence_threshold:
                cname = class_names[pred.item()]
                to_pil(img.cpu()).save(
                    os.path.join(out_root, cname, f"syn_{i:05d}.png")
                )
                accepted += 1

    print(f"Pseudo-labeled {accepted}/{num_images} synthetic images")

def remove_empty_class_dirs(root_dir):
    removed = 0
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if len(files) == 0:
            os.rmdir(class_path)
            removed += 1

    print(f"Removed {removed} empty synthetic class folders")

# -------------------------------------------------
# Train classifier (baseline or augmented)
# -------------------------------------------------
def train_classifier(use_synthetic=False):
    config = Config(
        "configs/data_config.yaml",
        "configs/train_config.yaml"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dir = config.data_dir / "Train"

    # -------------------------------------------------
    # BASE DATASET
    # -------------------------------------------------
    train_dataset = ImageFolder(
        root=train_dir,
        transform=transform
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)

    # -------------------------------------------------
    # SYNTHETIC AUGMENTATION
    # -------------------------------------------------
    if use_synthetic:
        print("🔧 Using GAN-based pseudo-labeled augmentation")

        gen = Generator(
            latent_dim=config.train["dcgan"]["latent_dim"]
        ).to(device)

        gen.load_state_dict(
            torch.load(
                config.checkpoints_dir / "G_epoch_150.pth",
                map_location=device
            )
        )

        # ----- Load Baseline Classifier as TEACHER -----
        teacher = models.resnet18(weights=None)
        teacher.fc = nn.Linear(teacher.fc.in_features, num_classes)

        teacher.load_state_dict(
            torch.load(
                config.checkpoints_dir / "classifier_baseline.pth",
                map_location=device
            )
        )

        teacher.to(device)
        teacher.eval()

        synthetic_root = "data/synthetic_pseudo"


        pseudo_label_images(
            generator=gen,
            teacher_model=teacher,
            class_names=class_names,
            device=device,
            out_root=synthetic_root,
            num_images=1000,              
            confidence_threshold=0.75
        )

        remove_empty_class_dirs(synthetic_root)

        # ----- Merge real + synthetic datasets -----
        synthetic_dataset = ImageFolder(
            root=synthetic_root,
            transform=transform
        )

        train_dataset.samples.extend(synthetic_dataset.samples)
        train_dataset.targets.extend(synthetic_dataset.targets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )

    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_dataset)}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"[{'AUGMENTED' if use_synthetic else 'BASELINE'}] "
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {running_loss/len(train_loader):.4f}"
        )

    out_name = (
        "classifier_augmented.pth"
        if use_synthetic else
        "classifier_baseline.pth"
    )

    torch.save(
        model.state_dict(),
        config.checkpoints_dir / out_name
    )

    print(f"Classifier saved: {out_name}")


if __name__ == "__main__":
    config = Config(
        "configs/data_config.yaml",
        "configs/train_config.yaml"
    )

    baseline_ckpt = config.checkpoints_dir / "classifier_baseline.pth"

    if not baseline_ckpt.exists():
        print("Training BASELINE classifier")
        train_classifier(use_synthetic=False)
    else:
        print("Baseline classifier already exists — skipping training")

    print("\nTraining AUGMENTED classifier (pseudo-labeled GAN images)")
    train_classifier(use_synthetic=True)

