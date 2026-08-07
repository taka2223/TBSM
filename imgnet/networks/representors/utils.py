import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ZOO = Path(os.environ.get("DM_MODEL_ZOO", PROJECT_ROOT / "model_zoo"))


def pretrained_path(*parts):
    return str(MODEL_ZOO / "pretrained" / Path(*parts))
