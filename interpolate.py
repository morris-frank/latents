#! /usr/bin/env python
from  argparse import ArgumentParser
from sleepwalker import SleepWalker
import os
from pathlib import Path

def prepare_result_folder(folder: str):
    os.chdir(f"./results/{folder}")
    Path("interpolations").mkdir(exist_ok=True)


def main(cfg):
    prepare_result_folder(cfg.folder)
    walker = SleepWalker(None, None, cfg.device, only_decode=True)
    walker.interpolate_frames(cfg.multiplier)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(dest="folder")
    parser.add_argument("-device", type=str, required=True)
    parser.add_argument("-multiplier", type=int, default=4)
    main(parser.parse_args())