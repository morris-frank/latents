"""isort:skip_file
"""
import sys
from subprocess import call
from pathlib import Path
from rich import pretty, traceback
from tqdm import tqdm
import urllib.request
pretty.install()
traceback.install()

LIB_DIR = Path(__file__).parents[1] / "lib"
LIB_DIR.mkdir(exist_ok=True)
libs = {
    "clip": "https://github.com/openai/CLIP.git",
    "taming-transformer": "https://github.com/CompVis/taming-transformers"
}
for lib, url in libs.items():
    if not (LIB_DIR / lib).exists():
        call(f"git clone {url} {LIB_DIR / lib}", shell=True)
    sys.path.insert(0, str(LIB_DIR / lib))

def wget(source: str, target: Path):
    if target.exists():
        return
    with urllib.request.urlopen(source) as source, open(str(target), "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))

MODEL_DIR = Path(__file__).parents[1] / "models" / "vqgan"
MODEL_DIR.mkdir(exist_ok=True, parents=True)
wget('https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1', MODEL_DIR / "last.ckpt")
wget('https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1', MODEL_DIR / "model.yaml") 

from .walker import SleepWalker, print