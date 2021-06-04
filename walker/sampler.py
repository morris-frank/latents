import numpy as np
import pandas as pd
import torch
from torch import Tensor
from rich import get_console

from .perceptor import Perceptor
from .generator import Generator

console = get_console()
print = console.print


class Pivot(torch.nn.Module):
    def __init__(self, width, height, batch_size=1):
        super().__init__()
        self.embed = (
            0.5 * torch.randn(batch_size, 256, width // 16, height // 16).cuda()
        )
        self.embed = torch.nn.Parameter(torch.sinh(1.9 * torch.arcsinh(self.embed)))

    def forward(self):
        return self.embed.clip(-6, 6)


class Sampler:
    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        device: str = "cuda:0",
    ) -> None:
        self.width, self.height = width, height
        self.device = device

        self.generator = Generator(width, height, device)
        self.perceptor = Perceptor(device)

        self.weight_decay = 0.1
        self.batch_size = 1
        self.lr = 0.5

        self.perception_attractor = None
        self.perception_repeller = None
        self.generation_attractor = None
        self.generation_repeller = None

    def reset(self):
        self.pivot = Pivot(self.width, self.height, self.batch_size).to(
            self.device
        )
        self.optimizer = torch.optim.AdamW(
            [{"params": [self.pivot.embed], "lr": self.lr}],
            weight_decay=self.weight_decay,
        )

    def imsave(self, path: str, **kwargs):
        self.generator.imsave(self.pivot(), path, **kwargs)

    def step(self) -> Tensor:
        image_at_pivot = self.generator.embed2img_augmented(self.pivot())
        perception_at_pivot = self.perceptor.img2embed(image_at_pivot)

        loss = 0

        if self.perception_attractor is not None:
            loss -= 10 * torch.cosine_similarity(self.perception_attractor, perception_at_pivot).mean()

        if self.perception_repeller is not None:
            loss += 5 * torch.cosine_similarity(self.perception_repeller, perception_at_pivot).mean()

        if self.generation_attractor is not None:
            loss -= 1 * torch.cosine_similarity(self.generation_attractor, self.pivot()).mean()

        if self.generation_repeller is not None:
            loss += 1 * torch.cosine_similarity(self.generation_repeller, self.pivot()).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        use_weight_decay = torch.abs(self.pivot()).max() > 5
        for g in self.optimizer.param_groups:
            g["weight_decay"] = self.weight_decay if use_weight_decay else 0

        return loss

    def train(self, img_name: str, n_steps: int):
        for i in range(n_steps):
            loss = self.step()
            if i % 100 == 0:
                for g in self.optimizer.param_groups:
                    print(
                        f"step={i:05d}, loss={loss.item():.3e}, lr={g['lr']}, decay={g['weight_decay']}"
                    )
                self.imsave(f"samples/{img_name}_{i:04d}")
        self.imsave(f"keyframes/{img_name}", save_embed=True)

    def generate_keyframes_v3(
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
                self.pivot.embed.data = latent.to(self.device)
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
