import sys
import urllib.request
from pathlib import Path
from subprocess import call
from typing import Dict

from rich import get_console, pretty, traceback
from tqdm import tqdm

libs = {
    "clip": "https://github.com/openai/CLIP.git",
    "taming-transformer": "https://github.com/CompVis/taming-transformers",
}

ckpts = {
    "last.ckpt": "https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1",
    "model.yaml": "https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1",
}

console = get_console()
print = console.print
rule = console.rule


def wget(source: str, target: Path) -> None:
    with urllib.request.urlopen(source) as source, open(str(target), "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))


def install_libs(libs: Dict) -> None:
    lib_dir = Path(__file__).parents[1] / "lib"
    lib_dir.mkdir(exist_ok=True)

    for lib, url in libs.items():
        if not (lib_dir / lib).exists():
            call(f"git clone {url} {lib_dir / lib}", shell=True)
        sys.path.insert(0, str(lib_dir / lib))


def download_ckpts(ckpts: Dict) -> Path:
    cache_dir = Path.home() / ".cache/walker"
    cache_dir.mkdir(exist_ok=True, parents=True)

    for filename, url in ckpts.items():
        if not (cache_dir / filename).exists():
            wget(url, cache_dir / filename)
    return cache_dir


def install() -> Path:
    pretty.install()
    traceback.install()
    install_libs(libs)
    return download_ckpts(ckpts)
