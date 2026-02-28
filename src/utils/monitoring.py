import time
import csv
import os
from datetime import datetime
from pathlib import Path


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

INFERENCE_LOG = LOG_DIR / "inference_latency.csv"
USAGE_LOG = LOG_DIR / "api_usage.csv"


# -------------------------------------------------
# CSV Initializer
# -------------------------------------------------
def _init_csv(path, headers):
    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


_init_csv(
    INFERENCE_LOG,
    ["timestamp", "latency_ms", "num_images"]
)

_init_csv(
    USAGE_LOG,
    ["timestamp", "source", "num_images", "top_class"]
)


# -------------------------------------------------
# Inference Latency Logger
# -------------------------------------------------
def log_inference_latency(start_time, num_images):
    """
    Logs inference latency in milliseconds.

    Args:
        start_time (float): time.time() before inference
        num_images (int): number of images generated
    """
    latency_ms = (time.time() - start_time) * 1000

    with open(INFERENCE_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            round(latency_ms, 2),
            num_images
        ])


# -------------------------------------------------
# Usage Logger (App / API)
# -------------------------------------------------
def log_usage(source, num_images, top_class):
    """
    Logs how the system is being used.

    Args:
        source (str): 'streamlit' or 'api'
        num_images (int): images generated
        top_class (str): dominant predicted class
    """
    with open(USAGE_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            source,
            num_images,
            top_class
        ])
