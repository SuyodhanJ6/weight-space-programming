"""Inspect the 20-image dataset (read-only) and write the manifest + report.

Checks: counts, dimensions, channels, formats, corruption, exact and near
duplicates (8x8 difference hash), simple colour/contrast/edge statistics.
Never modifies an image.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from data import IMAGE_DIR, LABELS, make_splits

MANIFEST_PATH = Path("experiments/manifest.json")
REPORT_JSON = Path("experiments/dataset_report.json")
NEAR_DUP_HAMMING = 6


def _dhash(img: Image.Image) -> int:
    small = img.convert("L").resize((9, 8), Image.BILINEAR)
    px = list(small.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | int(px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def _stats(img: Image.Image) -> dict:
    rgb = img.convert("RGB").resize((64, 64), Image.BILINEAR)
    px = list(rgb.getdata())
    n = len(px)
    mean = [sum(p[c] for p in px) / n for c in range(3)]
    gray = [0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2] for p in px]
    gm = sum(gray) / n
    contrast = (sum((g - gm) ** 2 for g in gray) / n) ** 0.5
    grad = sum(abs(gray[i + 1] - gray[i]) for r in range(64) for i in range(r * 64, r * 64 + 63))
    return {"mean_rgb": [round(m, 1) for m in mean], "brightness": round(gm, 1),
            "contrast": round(contrast, 1), "edge_density": round(grad / (64 * 63), 2),
            "warmth_r_minus_b": round(mean[0] - mean[2], 1)}


def inspect_one(path: Path) -> dict:
    raw = path.read_bytes()
    rec = {"path": str(path), "label": LABELS[path.name.split("_")[0]],
           "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    try:
        with Image.open(path) as img:
            fmt, mode = img.format, img.mode
            img.load()
            rec.update({"format": fmt, "mode": mode, "channels": len(img.getbands()),
                        "width": img.width, "height": img.height, "corrupted": False,
                        "dhash": _dhash(img), **_stats(img)})
    except Exception as exc:  # report, never hide, a corrupt file
        rec.update({"corrupted": True, "error": f"{type(exc).__name__}: {exc}"})
    return rec


def find_duplicates(records: list[dict]) -> dict:
    ok = [r for r in records if not r["corrupted"]]
    exact, near = [], []
    for i, a in enumerate(ok):
        for b in ok[i + 1:]:
            if a["sha256"] == b["sha256"]:
                exact.append([a["path"], b["path"]])
            elif bin(a["dhash"] ^ b["dhash"]).count("1") <= NEAR_DUP_HAMMING:
                near.append([a["path"], b["path"]])
    return {"exact": exact, "near": near}


def class_summary(records: list[dict], label: int) -> dict:
    rows = [r for r in records if r["label"] == label and not r["corrupted"]]
    keys = ["brightness", "contrast", "edge_density", "warmth_r_minus_b", "width", "height"]
    return {"n": len(rows), **{f"mean_{k}": round(sum(r[k] for r in rows) / len(rows), 1)
                                for k in keys}}


def main() -> None:
    splits = make_splits()
    split_of = {n: s for s, names in splits.items() for n in names}
    records = [inspect_one(p) for p in sorted(IMAGE_DIR.glob("*.jpg"))]
    for r in records:
        r["split"] = split_of[Path(r["path"]).name]
    dups = find_duplicates(records)
    report = {"counts": {"cat": sum(r["label"] == 0 for r in records),
                         "dog": sum(r["label"] == 1 for r in records)},
              "corrupted": [r["path"] for r in records if r["corrupted"]],
              "formats": sorted({r.get("format", "?") for r in records}),
              "modes": sorted({r.get("mode", "?") for r in records}),
              "duplicates": dups,
              "cat_summary": class_summary(records, 0), "dog_summary": class_summary(records, 1)}
    MANIFEST_PATH.write_text(json.dumps(
        [{k: r[k] for k in ("path", "label", "split", "sha256")} for r in records], indent=2))
    REPORT_JSON.write_text(json.dumps({"report": report, "images": records}, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
