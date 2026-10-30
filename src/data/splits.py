"""Shared label / split rules for GenImage.

GenImage layout (inside the folders or ZIPs):
    <generator>/train/ai/...      AI-generated, training data
    <generator>/train/nature/...  real ImageNet photos, training data
    <generator>/val/ai/...        AI-generated, official test data
    <generator>/val/nature/...    real ImageNet photos, official test data

GenImage has no separate validation set, so we make one:
    GenImage "train" -> our "train" (95 %) or "val" (5 %)
    GenImage "val"   -> our "test"  (never used for training or model choice)

The train/val decision is a hash of the image FILE NAME (not the full path).
ImageNet photos can appear in the folders of several generators; hashing the
file name puts every copy of the same photo into the same split, so no photo
can be in train and val at the same time. The rule is deterministic, so the
global and local branches get identical splits without sharing a file.
"""

import hashlib
import re

LABELS = {"nature": 0, "ai": 1}  # 0 = real, 1 = AI-generated
LABEL_NAMES = {value: key for key, value in LABELS.items()}


def path_parts(source):
    """Split 'dir/archive.zip!inner/path.png' into its path components."""
    return [part for part in re.split(r"[/\\!]", source) if part]


def parse_source(source):
    """Return (label, generator, genimage_split) or None if not recognised.

    label is 0 (nature/real) or 1 (ai). generator is the folder directly above
    'train'/'val' (e.g. 'Midjourney' or 'imagenet_ai_0419_sdv4'); for loose
    files directly under train/ the first path component is used.
    """
    parts = path_parts(source)
    lowered = [part.lower() for part in parts]
    label_index = max((i for i, p in enumerate(lowered) if p in LABELS), default=None)
    if label_index is None:
        return None
    split_index = max((i for i, p in enumerate(lowered[:label_index])
                       if p in ("train", "val")), default=None)
    if split_index is None:
        return None
    if split_index > 0:
        generator = parts[split_index - 1]
        if generator.lower().endswith(".zip"):
            generator = generator[:-4]
    else:
        generator = "unknown"
    return LABELS[lowered[label_index]], generator, lowered[split_index]


def assign_split(source, genimage_split, val_fraction=0.05, salt="genimage-v1"):
    """Map GenImage's train/val to our train/val/test (see module docstring)."""
    if genimage_split == "val":
        return "test"
    filename = path_parts(source)[-1].lower()
    digest = hashlib.sha256(f"{salt}:{filename}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2 ** 64
    return "val" if fraction < val_fraction else "train"


def image_id(source):
    """Short stable id for joining z1 (global) and z2 (local) in stage B."""
    return hashlib.sha1(source.encode()).hexdigest()[:16]
