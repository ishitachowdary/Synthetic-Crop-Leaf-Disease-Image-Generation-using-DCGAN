import os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class PlantLeafDataset(Dataset):

    def __init__(self, root_dir, image_size=64):
        self.root_dir = Path(root_dir)
        self.image_paths = self._collect_images()

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )  # maps [0,1] → [-1,1]
        ])

    def _collect_images(self):
        valid_ext = (".jpg", ".jpeg", ".png")
        paths = []
        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir():
                for img in class_dir.rglob("*"):
                    if img.suffix.lower() in valid_ext:
                        paths.append(img)
        return paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image)


def get_dataloader(
    split_dir,
    image_size=64,
    batch_size=64,
    shuffle=True,
    num_workers=2
):

    dataset = PlantLeafDataset(
        root_dir=split_dir,
        image_size=image_size
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def count_images(root_dir):

    valid_ext = (".jpg", ".jpeg", ".png")
    count = 0
    for root, _, files in os.walk(root_dir):
        count += sum(f.lower().endswith(valid_ext) for f in files)
    return count
