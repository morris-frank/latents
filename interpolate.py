#! /usr/bin/env python
import os
from argparse import ArgumentParser

from walker.generator import Generator
from walker.utils import load_script


def main(cfg):
    script = load_script("_".join(cfg.folder.split("_")[:-1]))
    os.chdir(f"./results/{cfg.folder}")
    walker = Generator(None, None, cfg.device)
    walker.render(script, cfg.fps)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(dest="folder")
    parser.add_argument("-device", type=str, required=True)
    parser.add_argument("-fps", type=int, default=60)
    main(parser.parse_args())
