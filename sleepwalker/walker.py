import random
from itertools import product
from pathlib import Path

import clip
import pandas as pd
import torch
import torchvision
import numpy as np
from omegaconf import OmegaConf
from rich import get_console
from taming.models.vqgan import VQModel
from torch import Tensor
from torch.nn import functional as F
from scipy import stats
from tqdm import tqdm

from . import MODEL_DIR

console = get_console()
print = console.print


def load_perceptor(device):
    clip.available_models()
    perceptor, _ = clip.load("ViT-B/32", device=device, jit=False)
    perceptor = perceptor.eval()
    return perceptor


def load_vqgan() -> VQModel:
    config = OmegaConf.load(MODEL_DIR / "model.yaml")
    ckpt_path= MODEL_DIR / "last.ckpt"

    model = VQModel(**config.model.params)
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    # NOTE: Whats up with this line:??
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()


class LatentPivot(torch.nn.Module):
    def __init__(self, width, height, batch_size=1):
        super().__init__()
        self.weights = (
            0.5 * torch.randn(batch_size, 256, width // 16, height // 16).cuda()
        )
        self.weights = torch.nn.Parameter(torch.sinh(1.9 * torch.arcsinh(self.weights)))

    def forward(self):
        return self.weights.clip(-6, 6)


class SleepWalker:
    path = {
        "samples": Path("samples"),
        "keyframes": Path("keyframes"),
        "interpolations": Path("interpolations"),
    }

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        device: str = "cuda:0",
        only_decode: bool = False,
    ) -> None:
        self.width, self.height = width, height
        self.device = device
        self.vqgan = load_vqgan().to(device)

        if not only_decode:
            self.perceptor = load_perceptor(self.device)
            self.weight_decay = 0.1
            self.batch_size = 1
            self.up_noise = 0.1
            self.lr = 0.5
            self.augmentations = torch.nn.Sequential(
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomAffine(24, (0.1, 0.1), fill=0),
            ).to(self.device)
            self.normalization = torchvision.transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ).to(self.device)

        for path in self.path.values():
            path.mkdir(exist_ok=True)

    def reset(self):
        self.pivot = LatentPivot(self.width, self.height, self.batch_size).to(
            self.device
        )
        self.optimizer = torch.optim.AdamW(
            [{"params": [self.pivot.weights], "lr": self.lr}],
            weight_decay=self.weight_decay,
        )

    @staticmethod
    def clip_image(tensor: Tensor) -> Tensor:
        return (tensor.clip(-1, 1) + 1) / 2

    @torch.no_grad()
    def img2latent(self, img: str) -> Tensor:
        img = torchvision.io.read_image(f"../../anchors/{img}")/255
        img = self.normalization(img)
        tr = torchvision.transforms.Resize((512, 512))
        img = tr(img)
        img = img[None, ...].to(self.device)
        return self.vqgan.encode(img)[0]

    def latent2img(self, latents: Tensor) -> Tensor:
        return self.vqgan.decode(latents)
        # quantized_embedding = self.vqgan.post_quant_conv(latents)
        # return self.vqgan.decoder(quantized_embedding)

    def text2embedding(self, text: str) -> Tensor:
        text = clip.tokenize(text).to(self.device)
        return self.perceptor.encode_text(text).detach().clone().cpu()

    def image2embedding(self, x: Tensor) -> Tensor:
        return self.perceptor.encode_image(x)

    def augment_image(self, img: Tensor, cutn=32) -> Tensor:
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
        return into

    def pivot_loss(self):
        current_image = self.latent2img(self.pivot())
        current_image = self.clip_image(self.augment_image(current_image))
        current_image = self.normalization(current_image)
        image_semantic_embedding = self.image2embedding(current_image)

        pos_loss = -10 * torch.cosine_similarity(
            self.pos_piv, image_semantic_embedding, -1
        )
        neg_loss = 5 * torch.cosine_similarity(
            self.neg_piv, image_semantic_embedding, -1
        )

        if self.pos_img_piv is not None:
            pos_loss += torch.cosine_similarity(self.pos_img_piv, self.pivot()).mean()

        return pos_loss + neg_loss

    def train_step(self) -> Tensor:
        loss = self.pivot_loss()

        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with_decay = torch.abs(self.pivot()).max() > 5
        for g in self.optimizer.param_groups:
            g["weight_decay"] = self.weight_decay if with_decay else 0

        return loss

    def train(self, basename: str, n_steps: int):
        for i in range(n_steps):
            loss = self.train_step()
            if i % 100 == 0:
                for g in self.optimizer.param_groups:
                    print(
                        f"step={i:05d}, loss={loss.item():.3e}, lr={g['lr']}, decay={g['weight_decay']}"
                    )
                self.checkin_lats(self.path["samples"] / f"{basename}_{i:04d}")
        self.checkin_lats(self.path["keyframes"] / basename, True)

    @torch.no_grad()
    def checkin(self, latents: Tensor, path: str):
        latents = latents.to(self.device)
        for image in self.latent2img(latents).cpu():
            torchvision.utils.save_image(image, f"{path}.png")

    @torch.no_grad()
    def checkin_lats(self, path, save_lats: bool = False):
        latents = self.pivot()
        if save_lats:
            torch.save(latents.cpu(), f"{path}.p")
        else:
            self.checkin(latents, path)

    def generate_keyframes(
        self,
        song: pd.DataFrame,
        fps: int,
        iter_per_frame: int = 100,
        warm_up_factor: float = 1.1,
        neg_text_anchor: str = "disconnected, confusing, incoherent",
        pos_text_anchor: str = None,
        neg_img_anchor: str = None,
        pos_img_anchor: str = None,
    ):
        self.reset()
        if neg_text_anchor is not None:
            neg_anchor = self.text2embedding(neg_text_anchor).to(self.device)
        
        if pos_text_anchor is not None:
            pos_anchor = self.text2embedding(pos_text_anchor).to(self.device)

        self.pos_img_piv = None
        if pos_img_anchor is not None:
            self.pos_img_piv = self.img2latent(pos_img_anchor)

        song["embedding"] = song.line.apply(self.text2embedding)

        assigs = []
        starting = True
        for win in song.rolling(3, center=True):
            cues = win.cue.tolist()
            indx = win.index.tolist()
            _i = len(win) == 2
            win_size = int((cues[2-_i] - cues[1-_i]) * fps)
            if _i:
                if starting:
                    indx.insert(0, None)
                    starting = False
                else:
                    indx.append(None)
            for i in range(win_size):
                assigs.append((indx, i/win_size))

        for frame, ((i_left, i_mid, i_right), progress) in enumerate(assigs):
            name = f"{frame:05d}"
            print(name)

            if (keyframe := self.path["keyframes"] / f"{name}.p").exists():
                latent = torch.load(keyframe)
                self.pivot.weights.data = latent.to(self.device)
                continue

            left, mid, right = None, song.loc[i_mid], None
            if i_left is not None:
                left = song.loc[i_left]
            if i_right is not None:
                right = song.loc[i_right]

            self.pos_piv = mid.embedding.to(self.device)
            if pos_text_anchor is not None:
                self.pos_piv += pos_anchor
            if right is not None and progress >= 0.8:
                self.pos_piv += right.embedding.to(self.device)
            self.pos_piv = self.pos_piv / self.pos_piv.norm(dim=-1, keepdim=True)

            self.neg_piv = neg_anchor
            if left is not None and progress <= 0.2:
                self.neg_piv += left.embedding.to(self.device)
            self.neg_piv = self.neg_piv / self.neg_piv.norm(dim=-1, keepdim=True)

            warm_up = warm_up_factor * (1 - np.sqrt(frame / len(assigs)))
            iters = int(iter_per_frame + warm_up * iter_per_frame)
            self.train(name, iters)

    @torch.no_grad()
    def interpolate_frames(self, multiplier: int):
        paths = list(map(torch.load, sorted(self.path["keyframes"].glob("*.p"))))
        i = 0
        for start, end in tqdm(zip(paths[:-1], paths[1:]), total=len(paths)):
            for sample in stats.norm.ppf(np.linspace(stats.norm.cdf(start), stats.norm.cdf(end), multiplier + 1))[:multiplier]:
                self.checkin(Tensor(sample), self.path["interpolations"] / f"{i:06d}")
                i += 1
