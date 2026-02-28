import csv
import json
from pathlib import Path
from datetime import datetime


class Logger:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.train_log = self.log_dir / "training_log.csv"
        self.infer_log = self.log_dir / "inference_log.csv"
        self.meta_log = self.log_dir / "metadata.json"

        self._init_csv(self.train_log, ["timestamp", "epoch", "d_loss", "g_loss"])
        self._init_csv(self.infer_log, ["timestamp", "num_images", "top_class"])

    def _init_csv(self, path, headers):
        if not path.exists():
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    # -------------------------------------------------
    # Training logging
    # -------------------------------------------------
    def log_training(self, epoch, d_loss, g_loss):
        with open(self.train_log, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                epoch,
                float(d_loss),
                float(g_loss)
            ])

    # -------------------------------------------------
    # Inference logging
    # -------------------------------------------------
    def log_inference(self, num_images, top_class):
        with open(self.infer_log, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                num_images,
                top_class
            ])

    # -------------------------------------------------
    # Metadata (run info)
    # -------------------------------------------------
    def save_metadata(self, info: dict):
        with open(self.meta_log, "w") as f:
            json.dump(info, f, indent=4)
