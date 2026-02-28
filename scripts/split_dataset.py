import os
import random
import shutil

SOURCE_DIR = "limited_dataset"
TARGET_DIR = "data/Real"

SPLITS = {
    "Train": 0.7,
    "Validation": 0.15,
    "Testing": 0.15
}

random.seed(42)

for split in SPLITS:
    os.makedirs(os.path.join(TARGET_DIR, split), exist_ok=True)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * SPLITS["Train"])
    n_val = int(n * SPLITS["Validation"])

    split_map = {
        "Train": images[:n_train],
        "Validation": images[n_train:n_train + n_val],
        "Testing": images[n_train + n_val:]
    }

    for split, imgs in split_map.items():
        out_dir = os.path.join(TARGET_DIR, split, class_name)
        os.makedirs(out_dir, exist_ok=True)

        for img in imgs:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(out_dir, img)
            )

print("Train / Validation / Test split completed")
