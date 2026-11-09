"""Inspect a cat/dog image dataset.

Reports: cat count, dog count, image dimensions, corrupted images, duplicates.
Read-only: never modifies or deletes images. No network access.

Expected layout (class folder names are matched case-insensitively):

    dataset/raw/Cat/*.jpg
    dataset/raw/Dog/*.jpg

Usage:
    python src/inspect_dataset.py [--root dataset/raw]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"})
CLASSES = ("cat", "dog")
HASH_CHUNK_BYTES = 1 << 20
MAX_LISTED = 20


def find_class_dirs(root: Path) -> dict[str, Path]:
    """Map each class name to its directory (case-insensitive match)."""
    found: dict[str, Path] = {}
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() in CLASSES:
            found[child.name.lower()] = child
    return found


def list_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def check_image(path: Path) -> tuple[tuple[int, int] | None, str | None]:
    """Return ((width, height), None) if decodable, else (None, error)."""
    try:
        with Image.open(path) as img:
            img.load()  # force full decode; verify() alone misses truncation
            return img.size, None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def inspect(root: Path) -> int:
    if not root.is_dir():
        print(f"Dataset root not found: {root}")
        return 1

    class_dirs = find_class_dirs(root)
    missing = [name for name in CLASSES if name not in class_dirs]
    if missing:
        print(f"Missing class folder(s) in {root}: {', '.join(missing)}")
        return 1

    sizes: Counter[tuple[int, int]] = Counter()
    corrupted: list[tuple[Path, str]] = []
    by_hash: dict[str, list[Path]] = defaultdict(list)
    counts: dict[str, int] = {}

    for name in CLASSES:
        images = list_images(class_dirs[name])
        counts[name] = len(images)
        for path in images:
            size, error = check_image(path)
            if error is not None:
                corrupted.append((path, error))
                continue
            sizes[size] += 1
            by_hash[file_digest(path)].append(path)

    duplicates = [paths for paths in by_hash.values() if len(paths) > 1]
    extra_copies = sum(len(paths) - 1 for paths in duplicates)

    print(f"Dataset root : {root}")
    print(f"Cat images   : {counts['cat']}")
    print(f"Dog images   : {counts['dog']}")
    print(f"Corrupted    : {len(corrupted)}")
    print(f"Duplicate groups: {len(duplicates)} ({extra_copies} redundant files)")

    if sizes:
        widths = [w for (w, _) in sizes.elements()]
        heights = [h for (_, h) in sizes.elements()]
        print(f"Width  min/max: {min(widths)} / {max(widths)}")
        print(f"Height min/max: {min(heights)} / {max(heights)}")
        print(f"Distinct dimensions: {len(sizes)}")
        print("Most common dimensions:")
        for (w, h), n in sizes.most_common(5):
            print(f"  {w}x{h}: {n}")

    for path, error in corrupted[:MAX_LISTED]:
        print(f"  CORRUPT {path}: {error}")
    for group in duplicates[:MAX_LISTED]:
        print("  DUPLICATE " + " == ".join(str(p) for p in group))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("dataset/raw"))
    args = parser.parse_args(argv)
    return inspect(args.root)


if __name__ == "__main__":
    sys.exit(main())
