#! /usr/bin/env python
import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from walker import Sampler, print


def load_song(name: str):
    return pd.read_csv(
        f"./songs/{name}.csv", header=None, names=["line", "cue"]
    ).sort_values("cue")


def prepare_result_folder(name):
    i = 0
    while (results_path := Path(f"./results/{name}_{i:03d}")).exists():
        i += 1
    results_path.mkdir()
    os.chdir(results_path)


def main(cfg):
    if cfg.cont:
        song = load_song("_".join(cfg.song.split("_")[:-1]))
        os.chdir(f"./results/{cfg.song}")
    else:
        song = load_song(cfg.song)
        prepare_result_folder(cfg.song)

    print(os.getcwd())
    walker = Sampler(
        device=cfg.device,
        width=cfg.width,
        height=cfg.height,
        targets={
            "perception_repeller": "disconnected, confusing, incoherent",
            "generation_attractor": cfg.img
        }
    )
    walker.generate_keyframes_v4(song)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(dest="song", type=str, help="name of song")
    parser.add_argument("-device", type=str, required=True)
    parser.add_argument("-width", type=int, default=512)
    parser.add_argument("-height", type=int, default=512)
    parser.add_argument("-c", action="store_true", dest="cont")
    parser.add_argument("-img", type=str, default=None)
    main(parser.parse_args())
