"""Whole-image context and native-pixel patches, shared by training and inference."""

import hashlib
import random

import numpy as np
from PIL import Image, ImageOps
import torch


def canonical_image(image):
    image.seek(0)
    oriented = ImageOps.exif_transpose(image).convert("RGBA")
    background = Image.new("RGBA", oriented.size, "white")
    background.alpha_composite(oriented)
    return background.convert("RGB")


def image_tensor(image):
    return (
        torch.from_numpy(np.array(image, dtype=np.uint8, copy=True))
        .permute(2, 0, 1)
        .float()
        / 255
    )


class ImageViews:
    def __init__(self, global_size, local_size, patches, seed, training=False):
        if min(global_size, local_size) < 32 or patches not in (1, 4):
            raise ValueError("View sizes must be >= 32; patches must be 1 or 4")
        self.global_size = global_size
        self.local_size = local_size
        self.patches = patches
        self.seed = seed
        self.training = training
        self.epoch = 0

    def __call__(self, image, sample_id):
        width, height = image.size
        scale = self.global_size / max(width, height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = image.resize(size, Image.Resampling.BILINEAR)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        global_view = mean.expand(3, self.global_size, self.global_size).clone()
        x, y = (self.global_size - size[0]) // 2, (self.global_size - size[1]) // 2
        global_view[:, y : y + size[1], x : x + size[0]] = image_tensor(resized)

        local_views = torch.zeros(self.patches, 3, self.local_size, self.local_size)
        mask = torch.zeros(self.patches, dtype=torch.bool)
        boxes = torch.full((self.patches, 4), -1, dtype=torch.long)
        if width >= self.local_size and height >= self.local_size:
            max_x, max_y = width - self.local_size, height - self.local_size
            if self.training:
                seed_text = f"{self.seed}:{self.epoch}:{sample_id}".encode()
                rng = random.Random(
                    int.from_bytes(hashlib.sha256(seed_text).digest()[:8])
                )
                count = (max_x + 1) * (max_y + 1)
                positions = rng.sample(range(count), min(self.patches, count))
                coordinates = [(p % (max_x + 1), p // (max_x + 1)) for p in positions]
            elif self.patches == 1:
                coordinates = [(max_x // 2, max_y // 2)]
            else:
                coordinates = list(
                    dict.fromkeys([(0, 0), (max_x, 0), (0, max_y), (max_x, max_y)])
                )
            for index, (x, y) in enumerate(coordinates):
                patch = image.crop((x, y, x + self.local_size, y + self.local_size))
                local_views[index] = image_tensor(patch)
                mask[index] = True
                boxes[index] = torch.tensor([x, y, self.local_size, self.local_size])
        return {
            "global": global_view,
            "local": local_views,
            "local_mask": mask,
            "crop_boxes": boxes,
        }
