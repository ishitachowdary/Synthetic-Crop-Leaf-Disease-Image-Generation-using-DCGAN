import yaml
from pathlib import Path


def load_yaml(path):

    with open(path, "r") as f:
        return yaml.safe_load(f)


class Config:


    def __init__(self, data_config_path, train_config_path):
        self.data = load_yaml(data_config_path)
        self.train = load_yaml(train_config_path)

        self.project_root = Path(__file__).resolve().parents[2]
        self.data_dir = self.project_root / self.data["data_dir"]

        self.checkpoints_dir = self.project_root / "checkpoints"
        self.samples_dir = self.project_root / "samples"
        self.figures_dir = self.project_root / "figures"

        self._create_dirs()

    def _create_dirs(self):
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.samples_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)
