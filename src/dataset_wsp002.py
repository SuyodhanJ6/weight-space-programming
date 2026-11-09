"""Build the WSP-002 dataset from the locally downloaded Microsoft Cats vs Dogs zip.

Offline. Validates every image, removes exact/near duplicates (and the 20 WSP-001 images),
samples with a fixed seed independent of image content, copies (never alters) originals,
writes the manifest, builds train/val caches, and SEALS the test images (chmod 000).
Only train+val content statistics are computed for the report.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("WSP_EXPERIMENT", "wsp-002")

from PIL import Image  # noqa: E402

import paths  # noqa: E402
from data import load_split  # noqa: E402
from image_inspector import _dhash, _stats  # noqa: E402

ZIP = paths.DATA_DIR / "download" / "kagglecatsanddogs.zip"
EXTRACT = paths.DATA_DIR / "download" / "extracted"
CONFIG = {"wsp-002": (2, {"train": 500, "val": 300, "test": 1000}),
          "wsp-003": (3, {"train": 4000, "val": 1500, "test": 3000}),
          "wsp-004": (4, {"train": 4000, "val": 1500, "test": 0})}   # test = ALL leftover unseen images
SEED, PER_CLASS = CONFIG[paths.EXPERIMENT]
PREVIOUS_MANIFESTS = {"wsp-003": [Path("experiments/wsp-002/manifest.json")],
                      "wsp-004": [Path("experiments/wsp-002/manifest.json"),
                                  Path("experiments/wsp-003/manifest.json")]}.get(paths.EXPERIMENT, [])
MIN_SIDE = 100
NEAR_HAMMING = 4
WSP001_PILOT = Path("experiments/pilot20")


def extract() -> None:
    EXTRACT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as zf:
        for member in zf.namelist():
            target = (EXTRACT / member).resolve()
            if not str(target).startswith(str(EXTRACT.resolve())):
                raise ValueError(f"unsafe zip member {member}")
        zf.extractall(EXTRACT)


def validate(path: Path) -> dict | str:
    """Return a record, or a reject reason string."""
    raw = path.read_bytes()
    if len(raw) < 1024:
        return "too_small_file"
    try:
        with Image.open(path) as img:
            fmt = img.format
            img.load()
            if fmt != "JPEG":
                return "not_jpeg"
            if min(img.size) < MIN_SIDE:
                return "too_small_image"
            return {"sha256": hashlib.sha256(raw).hexdigest(), "dhash": _dhash(img),
                    "w": img.width, "h": img.height, "mode": img.mode}
    except Exception:
        return "undecodable"


def pool_for(label_name: str, banned: set[str]) -> tuple[list[dict], Counter]:
    rejects: Counter = Counter()
    kept: list[dict] = []
    seen_sha: set[str] = set()
    for path in sorted((EXTRACT / "PetImages" / label_name).glob("*.jpg")):
        res = validate(path)
        if isinstance(res, str):
            rejects[res] += 1
            continue
        if res["sha256"] in banned or res["sha256"] in seen_sha:
            rejects["exact_duplicate_or_wsp001"] += 1
            continue
        seen_sha.add(res["sha256"])
        kept.append({"path": str(path), "label": 0 if label_name == "Cat" else 1, **res})
    return kept, rejects


def drop_near_duplicates(items: list[dict]) -> tuple[list[dict], int]:
    """Pigeonhole bucketing: 4x16-bit chunks; pairs sharing a chunk are compared (hamming<=4)."""
    buckets: dict[tuple[int, int], list[int]] = {}
    drop: set[int] = set()
    for i, it in enumerate(items):
        for c in range(4):
            buckets.setdefault((c, (it["dhash"] >> (16 * c)) & 0xFFFF), []).append(i)
    for members in buckets.values():
        for a_pos, a in enumerate(members):
            for b in members[a_pos + 1:]:
                if b not in drop and a not in drop and \
                        bin(items[a]["dhash"] ^ items[b]["dhash"]).count("1") <= NEAR_HAMMING:
                    drop.add(b)
    return [it for i, it in enumerate(items) if i not in drop], len(drop)


def sample(items: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    items = sorted(items, key=lambda it: it["sha256"])
    rng.shuffle(items)
    out, start = {}, 0
    for split, n in PER_CLASS.items():
        out[split] = items[start:start + n]
        start += n
    return out


def copy_split(split: str, items: list[dict], cls_name: str, start: int) -> list[dict]:
    base = paths.V2_SEALED_DIR if split == "test" else paths.V2_IMAGE_DIR / split
    base.mkdir(parents=True, exist_ok=True)
    entries = []
    for k, it in enumerate(items):
        dest = base / f"{cls_name}_{start + k:04d}.jpg"
        shutil.copy2(it["path"], dest)
        os.chmod(dest, 0o444)
        entries.append({"path": str(dest), "label": it["label"], "split": split,
                        "sha256": it["sha256"], "w": it["w"], "h": it["h"]})
    return entries


def inspection_report(entries: list[dict], rejects: dict, near: dict) -> dict:
    """Content statistics for TRAIN and VAL only. Test: counts only."""
    report = {"counts": {s: Counter(e["label"] for e in entries if e["split"] == s) for s in PER_CLASS},
              "rejects_by_reason": rejects, "near_duplicates_removed": near, "content": {}}
    report["counts"] = {s: {"cat": c[0], "dog": c[1]} for s, c in report["counts"].items()}
    for split in ("train", "val"):
        for label, name in ((0, "cat"), (1, "dog")):
            rows = [e for e in entries if e["split"] == split and e["label"] == label]
            stats = [_stats(Image.open(e["path"])) for e in rows[:150]]
            report["content"][f"{split}_{name}"] = {
                "n": len(rows),
                "mean_width": round(sum(e["w"] for e in rows) / len(rows), 1),
                "mean_height": round(sum(e["h"] for e in rows) / len(rows), 1),
                "min_side": min(min(e["w"], e["h"]) for e in rows),
                "max_side": max(max(e["w"], e["h"]) for e in rows),
                **{k: round(sum(s[k] for s in stats) / len(stats), 1)
                   for k in ("brightness", "contrast", "edge_density", "warmth_r_minus_b")},
                "brightness_range": [round(min(s["brightness"] for s in stats), 1),
                                     round(max(s["brightness"] for s in stats), 1)]}
    return report


def build_wsp004() -> None:
    """Train/val = exactly WSP-003's (like-for-like comparison); test = every leftover unseen image,
    after exact AND near-duplicate checks against everything used so far."""
    banned = {hashlib.sha256(p.read_bytes()).hexdigest() for p in WSP001_PILOT.glob("*.jpg")}
    for manifest in PREVIOUS_MANIFESTS:
        banned |= {e["sha256"] for e in json.loads(manifest.read_text())}
    old = [e for e in json.loads(Path("experiments/wsp-003/manifest.json").read_text()) if e["split"] != "test"]
    old_hashes = [_dhash(Image.open(e["path"])) for e in old]
    buckets: dict[tuple[int, int], list[int]] = {}
    for h in old_hashes:
        for c in range(4):
            buckets.setdefault((c, (h >> (16 * c)) & 0xFFFF), []).append(h)
    zip_sha = hashlib.sha256(ZIP.read_bytes()).hexdigest()
    extract()
    rng = random.Random(SEED)
    pools, rejects, near, leak = {}, {}, {}, {}
    for label_name, cls in (("Cat", "cat"), ("Dog", "dog")):
        pool, rej = pool_for(label_name, banned)
        pool, n_near = drop_near_duplicates(pool)
        kept = []
        for it in pool:   # near-duplicate of an already-used train/val image -> drop (leakage guard)
            hit = any(bin(it["dhash"] ^ h).count("1") <= NEAR_HAMMING
                      for c in range(4) for h in buckets.get((c, (it["dhash"] >> (16 * c)) & 0xFFFF), []))
            if not hit:
                kept.append(it)
        rejects[cls], near[cls], leak[cls] = dict(rej), n_near, len(pool) - len(kept)
        pools[cls] = sorted(kept, key=lambda it: it["sha256"])
    n = min(len(pools["cat"]), len(pools["dog"]))
    entries = [{k: e[k] for k in ("path", "label", "split", "sha256", "w", "h") if k in e} for e in old]
    for cls in ("cat", "dog"):
        rng.shuffle(pools[cls])
        entries += copy_split("test", pools[cls][:n], cls, 0)
    paths.EXP.mkdir(parents=True, exist_ok=True)
    paths.V2_MANIFEST.write_text(json.dumps(entries, indent=2))
    for split in ("train", "val"):
        load_split(split)
    report = {"counts": {s: dict(Counter("cat" if e["label"] == 0 else "dog" for e in entries if e["split"] == s))
                         for s in ("train", "val", "test")},
              "rejects_by_reason": rejects, "near_duplicates_removed": near,
              "near_duplicates_of_trainval_removed": leak, "zip_sha256": zip_sha, "seed": SEED,
              "total": len(entries), "note": "train/val identical to wsp-003; test = all leftover unseen images"}
    (paths.EXP / "dataset_report.json").write_text(json.dumps(report, indent=2))
    os.chmod(paths.V2_SEALED_DIR, 0o000)
    shutil.rmtree(EXTRACT)
    print(json.dumps(report, indent=2))


def main() -> None:
    if paths.V2_MANIFEST.exists():
        raise SystemExit("dataset already built; refusing to rebuild")
    if paths.EXPERIMENT == "wsp-004":
        return build_wsp004()
    banned = {hashlib.sha256(p.read_bytes()).hexdigest() for p in WSP001_PILOT.glob("*.jpg")}
    for manifest in PREVIOUS_MANIFESTS:   # earlier experiments' images (incl. their sealed tests): never reuse
        banned |= {e["sha256"] for e in json.loads(manifest.read_text())}
    zip_sha = hashlib.sha256(ZIP.read_bytes()).hexdigest()
    extract()
    rng = random.Random(SEED)
    entries, rejects, near = [], {}, {}
    for label_name, cls in (("Cat", "cat"), ("Dog", "dog")):
        pool, rej = pool_for(label_name, banned)
        pool, n_near = drop_near_duplicates(pool)
        rejects[cls], near[cls] = dict(rej), n_near
        split_items = sample(pool, rng)
        start = 0
        for split, items in split_items.items():
            entries += copy_split(split, items, cls, start)
            start = 0  # per-split numbering restarts per class folder prefix
    paths.EXP.mkdir(parents=True, exist_ok=True)
    paths.V2_MANIFEST.write_text(json.dumps(entries, indent=2))
    for split in ("train", "val"):
        load_split(split)  # builds the cache
    report = inspection_report(entries, rejects, near)
    report.update({"zip_sha256": zip_sha, "seed": SEED, "per_class": PER_CLASS,
                   "total": len(entries)})
    (paths.EXP / "dataset_report.json").write_text(json.dumps(report, indent=2))
    os.chmod(paths.V2_SEALED_DIR, 0o000)          # seal
    shutil.rmtree(EXTRACT)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
