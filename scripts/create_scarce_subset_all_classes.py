import os
import random
import shutil

SOURCE_DIR = "plantvillage_dataset/color"
TARGET_DIR = "limited_dataset"
MAX_IMAGES_PER_CLASS = 100  
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

random.seed(42)
os.makedirs(TARGET_DIR, exist_ok=True)

class_dirs = [
    d for d in os.listdir(SOURCE_DIR)
    if os.path.isdir(os.path.join(SOURCE_DIR, d))
]

print(f"📂 Found {len(class_dirs)} classes")

total_copied = 0

for class_name in sorted(class_dirs):
    src_class_dir = os.path.join(SOURCE_DIR, class_name)
    tgt_class_dir = os.path.join(TARGET_DIR, class_name)

    images = [
        f for f in os.listdir(src_class_dir)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]

    if len(images) == 0:
        print(f"⚠️ No images found in {class_name}, skipping")
        continue

    random.shuffle(images)
    selected = images[:MAX_IMAGES_PER_CLASS]

    os.makedirs(tgt_class_dir, exist_ok=True)

    for img in selected:
        shutil.copy(
            os.path.join(src_class_dir, img),
            os.path.join(tgt_class_dir, img)
        )

    total_copied += len(selected)
    print(f"{class_name:45s} → {len(selected)} images")

print("-" * 100)
print(f"🎉 Limited dataset created")
print(f"📊 Total classes   : {len(class_dirs)}")
print(f"📊 Images per class: ≤ {MAX_IMAGES_PER_CLASS}")
print(f"📊 Total images    : {total_copied}")
print(f"📂 Output dir      : {TARGET_DIR}")
