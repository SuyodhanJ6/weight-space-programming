"""Image loading and the fixed train/val/test split. No network access."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image

from paths import IS_V2, V2_CACHE_DIR, V2_MANIFEST

IMAGE_DIR = Path("experiments/pilot20")
SPLITS_PATH = Path("dataset/processed/splits.json")
IMAGE_SIZE = 64
PIXEL_MEAN = 0.5   # fixed constants, not dataset statistics
PIXEL_STD = 0.25
LABELS = {"cat": 0, "dog": 1}
SPLIT_SIZES = {"train": 6, "val": 2, "test": 2}  # per class
SPLIT_SEED = 0


def make_splits() -> dict[str, list[str]]:
    """Deterministic per-class split of the 20 pilot images; writes splits.json."""
    rng = random.Random(SPLIT_SEED)
    splits: dict[str, list[str]] = {name: [] for name in SPLIT_SIZES}
    for cls in LABELS:
        files = sorted(p.name for p in IMAGE_DIR.glob(f"{cls}_*.jpg"))
        rng.shuffle(files)
        start = 0
        for name, size in SPLIT_SIZES.items():
            splits[name].extend(files[start:start + size])
            start += size
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_PATH.write_text(json.dumps(splits, indent=2))
    return splits


def load_image(path: Path) -> torch.Tensor:
    """Resize shortest side to IMAGE_SIZE, center-crop, normalize -> (3, H, W)."""
    with Image.open(path) as raw:
        img = raw.convert("RGB")
    w, h = img.size
    scale = IMAGE_SIZE / min(w, h)
    img = img.resize((max(IMAGE_SIZE, round(w * scale)), max(IMAGE_SIZE, round(h * scale))),
                     Image.BICUBIC)
    w, h = img.size
    left, top = (w - IMAGE_SIZE) // 2, (h - IMAGE_SIZE) // 2
    img = img.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
    data = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    tensor = data.view(IMAGE_SIZE, IMAGE_SIZE, 3).permute(2, 0, 1).float() / 255.0
    return (tensor - PIXEL_MEAN) / PIXEL_STD


_TEST_UNLOCKED = False


def unlock_test() -> None:
    """wsp-002 only: called solely by the one-time `final` command."""
    global _TEST_UNLOCKED
    _TEST_UNLOCKED = True


def _load_split_v2(split: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    if split == "test" and not _TEST_UNLOCKED:
        raise PermissionError("wsp-002 test split is sealed until the one-time final evaluation")
    cache = V2_CACHE_DIR / f"{split}.pt"
    if not cache.exists():
        entries = [e for e in json.loads(V2_MANIFEST.read_text()) if e["split"] == split]
        images = torch.stack([load_image(Path(e["path"])) for e in entries])
        labels = torch.tensor([e["label"] for e in entries])
        V2_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"images": images, "labels": labels, "names": [e["path"] for e in entries]}, cache)
    blob = torch.load(cache, weights_only=True)
    return blob["images"], blob["labels"], blob["names"]


def load_split(split: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Return (images N,3,H,W), labels N (0=cat,1=dog), file names."""
    if IS_V2:
        return _load_split_v2(split)
    if not SPLITS_PATH.exists():
        make_splits()
    names = json.loads(SPLITS_PATH.read_text())[split]
    images = torch.stack([load_image(IMAGE_DIR / n) for n in names])
    labels = torch.tensor([LABELS[n.split("_")[0]] for n in names])
    return images, labels, names
