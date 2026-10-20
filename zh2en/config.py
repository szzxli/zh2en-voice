import yaml
from pathlib import Path

def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

ROOT = Path(__file__).resolve().parent.parent
