import os
from pathlib import Path

DATA_PATH = Path(os.getenv(
    "DATA_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
))