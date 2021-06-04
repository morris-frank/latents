import random
from pathlib import Path

import torch
import torchvision
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
from torch import Module, Tensor
from torch.nn import functional as F
from tqdm import tqdm

from . import CACHE_DIR
from .utils import linspace_gaussian, pairwise


class Generator:
    def __init__(self, width: int, height: int, device: str) -> None:
        self.width, self.height = width, height
        self.device = device
        self.model = self.load()

        self.up_noise = 0.1
        self.augentations = torch.nn.Sequential(
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomAffine(24, (0.1, 0.1), fill=0),
        ).to(self.device)
        self.normalization = torchvision.transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ).to(self.device)

    def load(self) -> Module:
        config = OmegaConf.load(CACHE_DIR / "model.yaml")
        model = VQModel(**config.model.params)
        state_dict = torch.load(CACHE_DIR / "last.ckpt", map_location="cpu")[
            "state_dict"
        ]
        _, _ = model.load_state_dict(state_dict, strict=False)
        return model.to(self.device).eval()

    @torch.no_grad()
    def img2embed(self, path: str) -> Tensor:
        img = torchvision.io.read_image(f"../../anchors/{path}") / 255
        img = self.normalization(img)
        img = torchvision.transforms.Resize((512, 512))(img)
        img = img[None, ...].to(self.device)
        return self.model.encode(img)[0]

    def embed2img(self, embed: Tensor) -> Tensor:
        return self.model.decode(embed)

    def embed2img_augmented(self, embed: Tensor) -> Tensor:
        image = self.embed2img(embed)
        image = self.augment(image)
        return self.normalization(image)

    @torch.no_grad()
    def imsave(self, embed: Tensor, path: str, save_embed: bool = False):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(embed, Tensor):
            embed = Tensor(embed)
        if save_embed:
            torch.save(embed.cpu(), f"{path}.p")
        img = self.embed2img(embed.to(self.device)).cpu()
        torchvision.utils.save_image(img[0, ...], f"{path}.png")

    @torch.no_grad()
    def interpolate_frames(self, multiplier: int):
        i = 0
        for start, end in tqdm(pairwise(sorted(self.path["keyframe"].glob("*.p")))):
            for embed in linspace_gaussian(start, end, multiplier + 1)[:multiplier]:
                self.imsave(embed, self.path["interpolations"] / f"{i:06d}")
                i += 1

    def augment(self, img: Tensor, cutn=32) -> Tensor:
        img = F.pad(
            img,
            (self.width // 2, self.width // 2, self.height // 2, self.height // 2),
            mode="constant",
            value=0,
        )

        img = self.augmentations(img)
        p_s = []
        for ch in range(cutn):
            # size = torch.randint(int(0.5 * self.width), int(0.98 * self.width), ())
            size = int(torch.normal(1.2, 0.3, ()).clip(0.43, 1.9) * self.width)

            if ch > cutn - 4:
                size = int(self.width * 1.4)
            offsetx = torch.randint(0, int(self.width * 2 - size), ())
            offsety = torch.randint(0, int(self.height * 2 - size), ())
            apper = img[:, :, offsetx : offsetx + size, offsety : offsety + size]
            apper = torch.nn.functional.interpolate(
                apper, (224, 224), mode="bilinear", align_corners=True
            )
            p_s.append(apper)
        into = torch.cat(p_s, 0)
        into = into + self.up_noise * random.random() * torch.randn_like(
            into, requires_grad=False
        )
        return (into.clip(-1, 1) + 1) / 2
