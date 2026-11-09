"""Build the WSP-006 dataset: the SAME 20 local pilot images (10 cats, 10 dogs), re-split with a new seed (12/4/4).

Offline, no downloads. Originals are copied (never altered). Test images are sealed (chmod 000) until `final`.
Disclosure: the agent has already seen all 20 images and WSP-001's results; WSP-006 is a replication on a different split.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from pathlib import Path

os.environ.setdefault("WSP_EXPERIMENT", "wsp-006")

from PIL import Image  # noqa: E402

import paths  # noqa: E402
from data import load_split  # noqa: E402

PILOT = Path("experiments/pilot20")
SEED = 6
PER_CLASS = {"train": 6, "val": 2, "test": 2}


def main() -> None:
    if paths.V2_MANIFEST.exists():
        raise SystemExit("dataset already built; refusing to rebuild")
    rng = random.Random(SEED)
    entries = []
    for cls, label in (("cat", 0), ("dog", 1)):
        files = sorted(PILOT.glob(f"{cls}_*.jpg"))
        rng.shuffle(files)
        start = 0
        for split, n in PER_CLASS.items():
            base = paths.V2_SEALED_DIR if split == "test" else paths.V2_IMAGE_DIR / split
            base.mkdir(parents=True, exist_ok=True)
            for k, src in enumerate(files[start:start + n]):
                dest = base / f"{cls}_{src.stem.split('_')[1]}.jpg"
                shutil.copy2(src, dest)
                os.chmod(dest, 0o444)
                with Image.open(dest) as im:
                    w, h = im.size
                entries.append({"path": str(dest), "label": label, "split": split,
                                "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(), "w": w, "h": h})
            start += n
    paths.EXP.mkdir(parents=True, exist_ok=True)
    paths.V2_MANIFEST.write_text(json.dumps(entries, indent=2))
    for split in ("train", "val"):
        load_split(split)                      # builds caches (test cache is NOT built)
    report = {"seed": SEED, "per_class": PER_CLASS, "total": len(entries),
              "counts": {s: sum(e["split"] == s for e in entries) for s in PER_CLASS},
              "source": str(PILOT), "note": "same 20 images as WSP-001/005, new split (seed 6); test sealed; images were seen in earlier experiments"}
    (paths.EXP / "dataset_report.json").write_text(json.dumps(report, indent=2))
    os.chmod(paths.V2_SEALED_DIR, 0o000)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
