#! /usr/bin/env python
from argparse import ArgumentParser
from sleepwalker import SleepWalker
from pathlib import Path
import os
import pandas as pd


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
    song = load_song(cfg.song)

    prepare_result_folder(cfg.song)
    walker = SleepWalker(width=cfg.width, height=cfg.height, device=cfg.device)
    walker.generate_keyframes(song, fps=cfg.sample_fps)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(dest="song", type=str, help="name of song")
    parser.add_argument("-sample_fps", type=int, default=15)
    parser.add_argument("-device", type=str, required=True)
    parser.add_argument("-width", type=int, default=512)
    parser.add_argument("-height", type=int, default=512)
    main(parser.parse_args())
