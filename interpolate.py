#! /usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
from sleepwalker import SleepWalker, print
import os
from subprocess import call


def main(cfg):
    os.chdir(f"./results/{cfg.folder}")
    walker = SleepWalker(None, None, cfg.device, only_decode=True)
    walker.interpolate_frames(cfg.multiplier)

    fps = int(cfg.sample_fps * cfg.multiplier)

    call(
        f'ffmpeg -framerate {fps} -pattern_type glob -i "interpolations/*.png" {cfg.folder}.mp4',
        shell=True,
    )

    song_path = Path(f"../../songs/{cfg.folder.split('_')[0]}.mp3")
    if song_path.exists():
        call(
            "ffmpeg -i {cfg.folder}.mp4 -i {song_path} -c:a aac {cfg.folder}_with_audio.mp4",
            shell=True,
        )
    else:
        print(
            f"The audio file {song_path} wasn't found. If you move it their, I will make the video also with music."
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(dest="folder")
    parser.add_argument("-device", type=str, required=True)
    parser.add_argument("-multiplier", type=int, default=4)
    parser.add_argument("-sample_fps", type=int, default=15)
    main(parser.parse_args())
