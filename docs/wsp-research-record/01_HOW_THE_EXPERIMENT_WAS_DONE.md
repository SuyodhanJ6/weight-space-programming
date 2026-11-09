# 01 - How the experiment was done (full method)

This document describes, step by step, exactly how the Weight-Space Programming (WSP) research was run:
the question, the environment, the data, the network, how every weight was initialised, how weights were
edited, how edits were accepted or rolled back, how results were measured, and how integrity was protected.

---------------------------------------------------------------------------------------------------

## 1. Research question

**Can an LLM agent build and improve a cat-vs-dog image classifier by reasoning about the network's
weights, biases and activations and editing them directly, without any gradient-based training?**

Label convention everywhere: `0 = cat`, `1 = dog`; the network outputs one logit, `sigmoid(logit) = P(dog)`.

Forbidden throughout (and enforced by the audit): backpropagation, `loss.backward()`, autograd,
gradient calculation, finite-difference gradients, SGD/Adam/any optimizer, fine-tuning, conventional
training loops, random or evolutionary weight search, pretrained models/weights, transfer learning,
external classifiers or model APIs. Loss values may be reported as a *metric* only.

Allowed: PyTorch used purely as a tensor/convolution library (autograd disabled), closed-form
statistical estimators computed once from training data (clearly reported as data fitting, see 6).

## 2. Environment

| Item | Value |
|---|---|
| Machine | macOS (Darwin), CPU only, no GPU |
| Python | 3.13 (uv-managed virtualenv `.venv`) |
| Libraries | torch 2.15.0, numpy 2.5.4, pillow 13.0.0 (declared in `pyproject.toml`, locked in `uv.lock`) |
| Autograd | disabled globally (`torch.set_grad_enabled(False)` in `src/model.py`), every parameter `requires_grad=False` |
| Network | the runner monkey-patches `socket.connect` / `create_connection` to raise; internet used only for the explicitly approved dataset downloads (section 3) |
| Source size | about 3,400 lines in `src/` |

## 3. Data

All data come from the **Microsoft Cats vs. Dogs (`PetImages`) zip** (about 25,000 JPEGs, 12,500 per class, sha256
`b7974bd00a84a99921f36ee4403f089853777b5ae8d151c76a86e64900334af9`). It was downloaded **four times, each time only after the
user explicitly approved it** (the project's `CLAUDE.md` otherwise forbids downloads; every exception is
recorded in the relevant report). After each build the zip and extraction were deleted.

### 3.1 Common validation and cleaning
For every candidate image the builder (`src/dataset_wsp002.py`) required: file >= 1 KB, decodes fully,
format JPEG, shorter side >= 100 px. It then removed **exact duplicates** (SHA-256) and
**near-duplicates** (8x8 difference hash, Hamming distance <= 4, pigeonhole bucketing over four 16-bit chunks).
Images are never modified; they are copied byte-for-byte, files made read-only (mode 444), and the manifest records each
file's SHA-256.

### 3.2 The four datasets

| Experiment | Source of images | Train | Validation | Sealed test | Sampling seed |
|---|---|---|---|---|---|
| WSP-001 | 20 images (10 cat + 10 dog) drawn from the first download with a seeded shuffle, valid RGB and >= 100 px | 12 (6+6) | 4 (2+2) | 4 (2+2) | 0 |
| WSP-002 | second download; excludes WSP-001's 20 images | 1,000 (500+500) | 600 (300+300) | 2,000 (1,000+1,000) | 2 |
| WSP-003 | third download; excludes WSP-001 and all WSP-002 images (including its sealed test) | 8,000 (4,000+4,000) | 3,000 (1,500+1,500) | 6,000 (3,000+3,000) | 3 |
| WSP-004 | fourth download; train/val are **identical to WSP-003's**; test = every leftover image not used anywhere before | 8,000 | 3,000 | 3,884 (1,942+1,942) | 4 |

Rejects (typical, per class): about 94-135 non-JPEG files, about 78-80 images smaller than 100 px, 2-3 near-empty files,
27-35 near-duplicates. WSP-004's test additionally drops near-duplicates of any already-used train/val image (18 cat + 14 dog removed).
The independent audits verified by SHA-256 that **train, validation and test never overlap within an experiment, and that
WSP-002, WSP-003 and WSP-004 tests do not overlap any earlier experiment's images.** One byte-identical pair with opposite
labels exists in the source dataset (cat_2054/dog_1703 in WSP-003's test), a source-data quirk.

### 3.3 Sealing the test split (WSP-002 onward)
* Sampling is content-independent (seeded shuffle over hash-sorted lists).
* Test images are copied into `dataset/<exp>/sealed_test/` and the directory is set to **mode 000**.
* No tensor cache for the test split exists until the single final run.
* `src/data.py::_load_split_v2` raises `PermissionError` for `"test"` unless `unlock_test()` has been called; the only caller is the
  one-time `final` command, which also re-hashes every test image against the manifest right after opening the directory and aborts (re-sealing)
  on any mismatch (added before WSP-004's final).
* During iterations only `train` and `val` (and cross-validation over `train`) are ever read; the audit proves it at runtime by wrapping the loader.

### 3.4 Preprocessing (identical for every experiment)
Resize the shorter side to 64 px (bicubic), center-crop to 64x64, scale to [0,1], then normalise with the **fixed** constants
`(x - 0.5) / 0.25` (not dataset statistics). Tensors are cached as float32 under `dataset/<exp>/cache/`.

## 4. The network (`src/model.py`)

`CatDogNet(widths=(c1, c2, c3))`:

```
input 3x64x64
-> conv 5x5 (3 -> c1, pad 2) -> ReLU -> maxpool 2          (32x32)
-> conv 3x3 (c1 -> c2, pad 1) -> ReLU -> maxpool 2         (16x16)
-> conv 3x3 (c2 -> c3, pad 1) -> ReLU -> adaptive-avgpool 4x4
-> flatten (c3 * 16 features) -> linear (-> 1 logit)
```

* WSP-001/002/003: widths (16, 32, 32), 512 head features, about 15 k parameters.
* WSP-004 (a documented, evidence-based design change, see 02 section 8): widths **(48, 96, 96)**, 1,536 head features.
* Constructor zeros every parameter and sets `requires_grad=False`: **nothing is random; every number is assigned deliberately.**

## 5. Hand-programmed initialisation (iter000)

### 5.1 WSP-001/002/003 baseline (`src/init_weights.py`)
**conv1 (16 filters, 5x5; Gabor/DoG luminance-weighted by (0.299, 0.587, 0.114) across R,G,B)**
* 0-7: even-phase (cosine) **Gabor** at orientations k*pi/8, sigma 1.5, wavelength 4 px, zero-mean, L1-normalised.
* 8-11: **difference-of-Gaussians** blob detectors: (sigma_in, sigma_out, polarity) = (1,2,+), (1,2,-), (0.8,1.6,+), (0.8,1.6,-).
* 12-15: **colour opponent**: R-G, G-R, B-Y, Y-B (vectors (1,-1,0), (-0.5,-0.5,1) and negatives) multiplied by a Gaussian blur (sigma 1.5).

**conv2 (32 channels over 16)**: 0-7 line-continuation along each orientation (3-tap line, 0.5 from neighbouring orientations);
8-15 corners (orientation k + k+90 degrees, uniform 1/9); 16-23 parallel pairs (taps across the edge, centre -0.5); 24-31 blob+edge
(DoG channel 8+k%4 at the centre + Gabor k/9). Each filter L1-normalised, biases 0.
**conv3 (32 channels over 32)**: centre-weighted 3x3 blur kernel [1 2 1; 2 4 2; 1 2 1]/16 over groups (all four types per orientation,
line pairs, corner/parallel mixes, isotropic energy). Biases 0.
(Known flaws found later by inspection: colour channels had zero weight into conv2; conv3 filters 24-31 were identical; ReLUs almost never gated.)

### 5.2 WSP-004 baseline (wide)
* conv1 **48** filters: even Gabors x 8 orientations x 2 polarities (16), odd-phase (sine) Gabors x 8 x 2 (16), DoG (4), colour axes (4),
  diagonal hue sectors (4: orange, yellow-green, blue-green, purple), coarse-frequency even Gabors (wavelength 6, sigma 2, orientations 0,2,4,6) (4).
* conv2: 96 channels = fine passthrough of all 48 conv1 maps (centre tap 1) + 48 coarse copies (3x3 box average).
* conv3: 96 channels, identity routing with the 3x3 blur kernel.

## 6. The head: closed-form statistical fits (data fitting, *not* hand edits)

The head (`linear`) is set once per model by a **closed-form estimator from TRAIN features only** (no iteration, no gradients):

| Rule | Formula |
|---|---|
| `lda` (diagonal LDA) | `w = (mu_dog - mu_cat) / (pooled per-feature variance + 1e-3)`, bias at the class-mean midpoint |
| `lda_demean` | diagonal LDA with `mean(w)` subtracted |
| `centroid` | `w = mu_dog - mu_cat` |
| `lda_pooled`, `lda_pooled_cb` | diagonal LDA over per-channel features averaged over all 16 cells, or over centre-4 vs border-12 cells (weights shared across cells) |
| `lda_full` (from WSP-003) | `w = (Sigma_within + 0.05 * mean_variance * I)^-1 (mu_dog - mu_cat)` via **one** `torch.linalg.solve`; ridge fixed in advance, never tuned |

After solving, weights and bias are rescaled so the train logits have standard deviation 2. `cell_mask` (16 multipliers) can down-weight pooled cells.
Anything computed from training statistics (head weights, normalisation scales, bias centring, channel-pruning choices) is **labelled "fit" or "derived"**
in the reports, separately from **"hand"** values the agent wrote by reasoning.

## 7. The WSP engine: how a weight edit happens (`src/wsp_engine.py`, `src/weight_programmer.py`)

1. **IterationSpec** (in `src/iterations*.py`) holds: id, title, observed behaviour, problem, hypothesis, selected parameters, expected effect, risks, and an `edit(ctx)` function. All were written by the LLM agent *after* inspecting the current model; they are append-only.
2. The engine loads the **current accepted checkpoint** and verifies its SHA-256 against the state file.
3. `edit(ctx)` runs. It may only change weights through `WeightProgrammer`: `modify_parameter`, `modify_parameter_group`, `replace_parameter`, `apply_parameter_transform` (also `inspect_parameters`, `get_parameter`, `compare_parameters`, `validate_parameter_changes`, `save_checkpoint`, `restore_checkpoint`). Every operation logs parameter, selector, reason, number of changed elements and sample old/new values. `ctx.refit_head()` re-derives the head in closed form and logs it as two `replace_parameter` operations.
4. **Validation**: shapes unchanged, all values finite, no autograd state, and a **runtime tripwire** rejects the iteration if any tensor changed beyond what the logged operations explain.
5. **Evaluation on train and validation only** (plus a 5-fold CV diagnostic over train for WSP-002+, leave-one-out over train+val for WSP-001): accuracy, precision, recall, F1, confusion matrix, signed margins, BCE (metric only).
6. **Decision by the pre-registered rule** (written before any edit of that experiment, stored in `experiment_state.json` and `decision_log.md`).
7. The candidate checkpoint is **always saved** (`checkpoints/<exp>/iterNNN.pt`) together with the full old/new diff (`experiments/<exp>/param_changes/iterNNN.pt`) and a hash.
   If rejected, the next iteration simply starts from the previous accepted checkpoint (**rollback by construction**); rejected experiments are never deleted.
8. State is written atomically (`tmp` + `os.replace`): `experiment_state.json`, `experiment_history.json`, `current_iteration.json`, `parameter_changes.json`, `decision_log.md`, `logs/.../execution.log`, `reports/.../PROGRESS.md`.
9. **Restart safety**: on start the runner reads the state, marks any `in_progress` iteration as interrupted (rolled back), skips completed iterations, and never re-initialises a valid baseline.

### 7.1 Acceptance rules (pre-registered)
* **WSP-001**: reject on any regression of validation accuracy or leave-one-out accuracy; accept on +1 validation image (0.25) or +1 LOO image (0.0625), or on a margin-only gain (val margin +0.25 with LOO margin not down more than 0.10).
* **WSP-002/003/004**: accept iff **(a)** validation accuracy rises by **>= 0.010** AND **(b)** 5-fold train-CV accuracy does not fall by more than 0.010 AND **(c)** validation F1 does not fall by more than 0.010. Train accuracy is recorded but **never** used. Thresholds were never altered (iter019 of WSP-003 reached +0.0097 and was correctly rejected).

## 8. Measurements and statistics

* Accuracy, precision, recall, F1, confusion matrix, mean signed margin (logit x label sign), BCE (as metric).
* 95% **Wilson** confidence intervals for accuracies; **exact McNemar** test (paired, two-sided) for comparing two models on the same images.
* **5-fold CV on train**: the head rule is re-fitted on 4/5 of train and applied to the held-out 1/5 (closed-form refit per fold).
* **Controls** (`src/controls.py`, `src/controls_wsp002.py`): random convolution filters + the same closed-form head (does hand design beat random features?), label-shuffled head (is accuracy above chance?), head-versus-extractor attribution (swap diagonal/full head on the same extractor).

## 9. Integrity protections (summary; details in 03_AUDIT_RECORD.md)
* Test split sealed (section 3.3); single `final` run per experiment; test never used for any decision.
* Previous experiments' artifacts hashed and made read-only before each new experiment; the audit re-verifies them.
* `src/paths.py` namespaces each experiment (`experiments/`, `checkpoints/`, `logs/`, `reports/`, `dataset/` per experiment).
* The static AST audit, in-place-mutation scan, runtime checks, hash-exact replay of every iteration, and independent sub-agent reviews.

## 10. Timeline of what was done (phases)

1. **Setup**: project structure, `uv` environment, dataset inspection script, README. Dataset not local -> asked -> approved -> downloaded the Microsoft zip (787 MB) -> deleted the 25 k images at the user's request after selecting 20.
2. **Look at the 20 images** (contact sheets) and write down visual differences (02 section 1); propose architecture and strategy; user approved PyTorch install.
3. **WSP-001 (20 images)**: build model, hand-initialise, baseline, 9 reasoned edits, controls, own audit + independent audit, one-time test, report.
4. **WSP-002 (3,600 images)**: second download (approved); refactor for namespaces; protect WSP-001; baseline; 27 iterations across 6 rounds; controls; audit upgrades (weight-update audit: tripwire, mutation scan, hash-exact replay, trail); test; report with addenda.
5. **WSP-003 (17,000 images)**: third download (approved) after finding the 600-image validation set could not resolve effects; 21 iterations; discovery that the diagonal head was the bottleneck; debugging of invalid iter011/012; controls (attribution, random features); independent audit; test; report.
6. **WSP-004 (wide network)**: fourth download (approved) for a fresh, unseen test set; architecture widened; 3 iterations; audit bug fixed; independent audit; controls; one-time test; report.
7. **This record** (`docs/wsp-research-record/`).

## 11. Reproduction

```bash
uv sync
export WSP_EXPERIMENT=wsp-003            # wsp-001 (default), wsp-002, wsp-003, wsp-004
python src/dataset_wsp002.py             # needs the downloaded zip in dataset/<exp>/download/ (WSP-001's 20 images are in experiments/pilot20)
python src/experiment_runner.py init     # baseline checkpoint + state
python src/experiment_runner.py run      # resumable; executes src/iterations_wsp003.py
python src/weight_audit.py               # full audit (use --no-write for protected experiments)
python src/controls_wsp002.py            # null controls (WSP-002+)
python src/experiment_runner.py final    # ONE-TIME sealed-test evaluation
scripts/run_wsp.sh                       # restart supervisor around `run`
```
