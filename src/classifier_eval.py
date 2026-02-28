import torch
import numpy as np
import pandas as pd
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from utils.config import Config


def evaluate(model_path):
    config = Config(
        "configs/data_config.yaml",
        "configs/train_config.yaml"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test_dataset = ImageFolder(
        root=config.data_dir / "Testing",
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    num_classes = len(test_dataset.classes)

    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    return acc, f1, cm, test_dataset.classes


if __name__ == "__main__":
    config = Config(
        "configs/data_config.yaml",
        "configs/train_config.yaml"
    )

    base_path = config.checkpoints_dir / "classifier_baseline.pth"
    aug_path = config.checkpoints_dir / "classifier_augmented.pth"

    acc_b, f1_b, _, classes = evaluate(base_path)
    acc_a, f1_a, _, _ = evaluate(aug_path)

    print("\nCLASSIFIER COMPARISON")
    print("------------------------")
    print(f"Baseline Accuracy : {acc_b:.4f}")
    print(f"Baseline F1-score : {f1_b:.4f}")
    print()
    print(f"Augmented Accuracy: {acc_a:.4f}")
    print(f"Augmented F1-score: {f1_a:.4f}")

    report = pd.DataFrame([
        {"Model": "Baseline", "Accuracy": acc_b, "F1_macro": f1_b},
        {"Model": "Augmented", "Accuracy": acc_a, "F1_macro": f1_a}
    ])

    report.to_csv("logs/classifier_comparison.csv", index=False)
    print("\nResults saved: logs/classifier_comparison.csv")
