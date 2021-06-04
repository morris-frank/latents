#! /usr/bin/env python
import os
from argparse import ArgumentParser

from walker.generator import Generator


def main(cfg):
    os.chdir(f"./results/{cfg.folder}")
    walker = Generator(None, None, cfg.device)
    walker.interpolate_frames(cfg.multiplier)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(dest="folder")
    parser.add_argument("-device", type=str, required=True)
    parser.add_argument("-multiplier", type=int, default=4)
    main(parser.parse_args())
