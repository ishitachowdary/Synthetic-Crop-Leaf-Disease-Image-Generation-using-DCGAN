import os
import subprocess
import zipfile

DATASET = "abdallahalidev/plantvillage-dataset"
ZIP_NAME = "plantvillage_dataset.zip"
EXTRACT_DIR = "PlantVillage"

def download_dataset():
    print("Downloading PlantVillage dataset from Kaggle...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET],
        check=True
    )
    print("Download complete")

def unzip_dataset():
    print("Unzipping dataset...")
    with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
        zip_ref.extractall(".")
    print("Unzip complete")

if __name__ == "__main__":
    if not os.path.exists(ZIP_NAME):
        download_dataset()
    else:
        print("Dataset zip already exists, skipping download")

    if not os.path.exists(EXTRACT_DIR):
        unzip_dataset()
    else:
        print("Dataset already extracted")

    print("PlantVillage dataset ready")
