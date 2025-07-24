# path_utils.py
from pathlib import Path

# directory where *this file* lives (i.e. your project root)
ROOT = Path(__file__).resolve().parent

DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)  # create once; no error if already there
