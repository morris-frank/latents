#! /usr/bin/env python
import os
from argparse import ArgumentParser

from sleepwalker import SleepWalker


def main(cfg):
    os.chdir(f"./results/{cfg.folder}")
    walker = SleepWalker(None, None, cfg.device, only_decode=True)
    walker.interpolate_frames(cfg.multiplier)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(dest="folder")
    parser.add_argument("-device", type=str, required=True)
    parser.add_argument("-multiplier", type=int, default=4)
    main(parser.parse_args())
