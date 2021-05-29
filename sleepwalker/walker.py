import clip
from numpy import product
from rich import get_console
import torch
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
import pandas as pd
from torch import Tensor
from collections import namedtuple
import random
from itertools import product
from tqdm import tqdm
from torch.nn import functional as F
import torchvision
from pathlib import Path

console = get_console()
print = console.print


DATA_DIR = Path("~/data/").expanduser()


def load_perceptor(device):
    clip.available_models()
    perceptor, _ = clip.load("ViT-B/32", device=device, jit=False)
    perceptor = perceptor.eval()
    return perceptor


def _load_vqgan(config, ckpt_path=None) -> VQModel:
    model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        # NOTE: Whats up with this line:??
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()


def load_vqgan() -> VQModel:
    config16384 = OmegaConf.load(
        DATA_DIR / "vqgan_imagenet_f16_16384/configs/model.yaml"
    )
    model16384 = _load_vqgan(
        config16384,
        ckpt_path=DATA_DIR / "vqgan_imagenet_f16_16384/checkpoints/last.ckpt",
    )
    return model16384


class Pars(torch.nn.Module):
    def __init__(self, width, height, batch_size=1):
        super(Pars, self).__init__()
        self.normu = (
            0.5 * torch.randn(batch_size, 256, width // 16, height // 16).cuda()
        )
        self.normu = torch.nn.Parameter(torch.sinh(1.9 * torch.arcsinh(self.normu)))

    def forward(self):
        return self.normu.clip(-6, 6)


class SleepWalker:
    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        device: str = "cuda:0",
        only_decode: bool = False,
    ) -> None:
        self.width, self.height = width, height
        self.device = device
        if not only_decode:
            self.perceptor = load_perceptor(self.device)
        self.vqgan = load_vqgan().to(device)
        self.weight_decay = 0.1
        self.batch_size = 1
        self.up_noise = 0.1
        self.lr = 0.5
        self.augmentations = torch.nn.Sequential(
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomAffine(24, (0.1, 0.1), fill=0),
        ).to(self.device)
        self.normalization = torchvision.transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        ).to(self.device)

    def reset(self):
        self.lats = Pars(self.width, self.height, self.batch_size).to(self.device)
        self.optimizer = torch.optim.AdamW(
            [{"params": [self.lats.normu], "lr": self.lr}],
            weight_decay=self.weight_decay,
        )

    def clip_image(self, tensor: Tensor) -> Tensor:
        return (tensor.clip(-1, 1) + 1) / 2

    def current_image(self) -> Tensor:
        return self.decode_image(self.lats())

    def encode_text(self, text: str) -> Tensor:
        text = clip.tokenize(text).to(self.device)
        return self.perceptor.encode_text(text).detach().clone()

    def decode_image(self, z: Tensor) -> Tensor:
        quant_embd = self.vqgan.post_quant_conv(z)
        return self.vqgan.decoder(quant_embd)

    def encode_image(self, x: Tensor) -> Tensor:
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

    def ascend_txt(self):
        current_image = self.current_image()
        current_image = self.clip_image(self.augment_image(current_image))
        current_image = self.normalization(current_image)
        image_semantic_embedding = self.encode_image(current_image)

        pos_loss = -10 * torch.cosine_similarity(
            self.pos_piv, image_semantic_embedding, -1
        )
        neg_loss = 5 * torch.cosine_similarity(
            self.neg_piv, image_semantic_embedding, -1
        )
        return pos_loss + neg_loss

    def iterate(self) -> Tensor:
        loss = self.ascend_txt()

        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with_decay = torch.abs(self.lats()).max() > 5
        for g in self.optimizer.param_groups:
            g["weight_decay"] = self.weight_decay if with_decay else 0

        return loss

    def train(self, basename: str, n_steps: int):
        for i in range(n_steps):
            loss = self.iterate()
            if i % 20 == 0:
                for g in self.optimizer.param_groups:
                    print(
                        f"step={i:05d}, loss={loss.item():.3e}, lr={g['lr']}, decay={g['weight_decay']}"
                    )
                self.checkin(f"samples/{basename}_{i:04d}")
        self.checkin(basename, True)

    @torch.no_grad()
    def checkin(self, path: str, save_lats: bool = False):
        lats = self.lats()
        if save_lats:
            torch.save(lats.cpu(), f"{path}.p")
        for image in self.decode_image(lats).cpu():
            torchvision.utils.save_image(image, f"{path}.png")

    def generate_keyframes(
        self, song: pd.DataFrame, fps: int, iter_per_frame: int = 100
    ):
        self.reset()

        neg_txt = "disconnected, confusing, incoherent"
        neg_emb = self.encode_text(neg_txt)

        total_length = int(song.iloc[-1].cue) + 1
        n_frames = int(total_length * fps)
        Line = namedtuple("Line", ["text", "emb", "marker"])
        lines = []
        for l in song.itertuples():
            lines.append(Line(l.line, self.encode_text(l.line), int(l.cue * fps)))

        left, right = None, None
        for frame in range(n_frames):
            name = f"{frame:05d}"

            for line in lines:
                if line.marker == frame:
                    left = line
                if line.marker > frame:
                    right = line
                    break

            progress = (frame - left.marker) / (right.marker - left.marker)
            self.pos_piv = (1.0 - progress) * left.emb + progress * right.emb
            self.pos_piv = self.pos_piv / self.pos_piv.norm(dim=-1, keepdim=True)

            self.neg_piv = progress * left.emb + neg_emb
            self.neg_piv = self.neg_piv / self.neg_piv.norm(dim=-1, keepdim=True)

            print(
                f"{name}\n\t{1-progress:.3f}  {left.text}\n\t{progress:.3f}  {right.text}"
            )
            self.train(name, iter_per_frame)

    @torch.no_grad()
    def interpolate_frames(self, multiplier: int):
        paths = list(map(torch.load, sorted(Path(".").glob("*.p"))))
        i = 0
        for (start, end), p in tqdm(
            product(zip(paths[:-1], paths[1:]), torch.linspace(0, 1, multiplier)),
            total=(len(paths) - 1) * multiplier,
        ):
            emb = (1 - p) * start + p * end
            for image in self.decode_image(emb.to(self.device)).cpu():
                torchvision.utils.save_image(image, f"interpolations/{i:04d}.png")
            i += 1
