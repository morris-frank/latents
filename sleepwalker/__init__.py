"""isort:skip_file
"""
import sys
from pathlib import Path
from rich import pretty, traceback
pretty.install()
traceback.install()

LIB_DIR = Path(__file__) / "../lib"
MODEL_DIR = Path(__file__) / "../data"

sys.path.insert(0, LIB_DIR / "clip")
sys.path.insert(0, LIB_DIR / "taming-transformers")

from .walker import SleepWalker