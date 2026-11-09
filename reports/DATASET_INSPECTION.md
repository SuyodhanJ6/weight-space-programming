# Dataset inspection (20 images)

Source: Microsoft Cats vs. Dogs (`PetImages`). 10 cats + 10 dogs were selected with a seeded
random draw (seed 0) from valid, RGB, >=100 px images and copied to `experiments/pilot20/`
(read-only). The full 25k-image download was deleted at the user's request; only these 20 remain.

| Check | Result |
|---|---|
| Count | 10 cat, 10 dog |
| Format / channels | all JPEG, RGB (3 channels) |
| Size | 237-500 px per side (mean ~449x403 cat, ~402x407 dog) |
| Corrupted | 0 |
| Exact duplicates (SHA-256) | 0 |
| Near duplicates (8x8 dHash, Hamming <=6) | 0 |

Manifest with labels and split: `experiments/manifest.json`; per-image stats:
`experiments/dataset_report.json`.

## Split (fixed, seed 0)
train 12 (6+6) - used to set the head in closed form and as examples for reasoning;
val 4 (2+2) - used with leave-one-out (LOO) to accept/reject edits;
test 4 (2+2) - locked; read once, at the very end.

## Visual observations (all 20 viewed on contact sheets)
- Cats: roundish face, small triangular ears, short muzzle, large widely spaced eyes (flash glow
  common); 7/10 dark or black-and-white; compact, curled poses; cages/beds/chairs as background.
- Dogs: longer protruding muzzle, dark nose, varied ear shape (floppy/folded/upright), full
  standing body with long legs; tan/brown/black fur; dirt, grass, fences and kennels behind;
  one image (dog_02) has a person in a Santa suit.
- Colour is only a weak cue: both classes contain black animals. In this sample dogs are
  warmer (mean R-B 27.5 vs 13.4; same on the 16 dev images: 27.7 vs 12.4), brighter
  (117 vs 95) and have more edges (11.4 vs 9.7 mean abs gradient).
- Background correlates with setting (cages/fences in both classes), a leakage risk.
- Disclosure: these 20-image statistics (incl. the 4 test images) informed hypothesis iter004
  (colour). The dev-only (16-image) statistics give the same numbers; no model output on the test
  split was used.
