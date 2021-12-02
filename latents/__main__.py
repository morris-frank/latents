from pathlib import Path
from argparse import ArgumentParser

def main(inputs: list[str], device: str):
    print(Path(".").resolve())
    for input in inputs:
        pass

parser = ArgumentParser()
parser.add_argument("inputs", nargs="+", help="Positive inputs to use (path to image, path to script or prompt)")
parser.add_argument("-device", default="cuda:0")

main(**vars(parser.parse_args()))