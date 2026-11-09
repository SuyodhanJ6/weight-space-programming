"""WSP-004 iterations: wide network (conv1 48 / conv2 96 / conv3 96 -> 1,536 head features).

Train/val = WSP-003's; sealed test = 3,884 fresh images (+ WSP-003's final model scored on it once as a reference).
Hand-written, hypothesis-driven edits judged on train/val/CV only. Edits marked [closed-form head] reassign head
weights by a one-shot formula over TRAIN statistics (data fitting). Append-only.
"""

from __future__ import annotations

import torch

from wsp_engine import EditContext, IterationSpec

C1 = 48
BASELINE_OBS = ("WSP-004 baseline (hand-written 48-filter conv1, fine+coarse passthrough, full-covariance head): train 0.795, "
                "val 0.784, CV 0.777. Ablation: fine-only 0.781, coarse-only 0.778 (coarse copies add ~0.3 pt); odd Gabors "
                "most valuable (drop -> 0.768); coarse-freq Gabors, DoG, colour, hue each add <=0.4 pt.")


def _pool1_means(ctx: EditContext) -> torch.Tensor:
    from evaluator import split_data
    images, _, _ = split_data("train")
    total, count = torch.zeros(C1), 0
    for i in range(0, len(images), 500):
        batch = ctx.model.pool(torch.relu(ctx.model.conv1(images[i:i + 500])))
        total += batch.sum(dim=(0, 2, 3))
        count += batch.shape[0] * batch.shape[2] * batch.shape[3]
    return (total / count).clamp_min(1e-4)


def edit_peaks_replace_coarse(ctx: EditContext) -> None:
    w2 = torch.zeros(48, C1, 3, 3)
    for j in range(C1):
        w2[j] = -1.0 / 9
        w2[j, j, 1, 1] += 1.0                                    # ReLU(f_j - 3x3 box mean of f_j)
    ctx.programmer.modify_parameter_group(
        "conv2.weight", (slice(48, 96), slice(None), slice(None), slice(None)), w2,
        "conv2 ch 48-95: local peakiness ReLU(f_j - 3x3 box mean) replaces the coarse box copies")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_conjunctions_replace_coarse(ctx: EditContext) -> None:
    scale = _pool1_means(ctx)
    w2 = torch.zeros(48, C1, 3, 3)
    b2 = torch.full((48,), -2.0)
    slot = 0
    def conj(a: int, b: int) -> None:
        nonlocal slot
        w2[slot, a, 1, 1] = 1.0 / scale[a]
        w2[slot, b, 1, 1] = 1.0 / scale[b]
        slot += 1
    for colour in range(36, 40):                       # 4 colour axes x 8 even-Gabor orientations = 32
        for k in range(8):
            conj(colour, k)
    for k in range(8):                                 # 8 oriented-bar corner pairs (k, k+45 deg)
        conj(k, (k + 2) % 8)
    for k in range(8):                                 # 8 odd-phase junction pairs (k, k+90 deg)
        conj(16 + k, 16 + (k + 4) % 8)
    ctx.notes.append(f"train-mean scales for conv1 channels (first 8): {[round(float(v), 4) for v in scale[:8]]}")
    ctx.programmer.modify_parameter_group(
        "conv2.weight", (slice(48, 96), slice(None), slice(None), slice(None)), w2,
        "conv2 ch 48-95: soft-AND conjunctions (colour x orientation, corner pairs, junction pairs), train-mean scales")
    ctx.programmer.modify_parameter_group("conv2.bias", (slice(48, 96),), b2,
                                          "conv2.bias[48:96] = -2.0: ReLU(a/s_a + b/s_b - 2) soft AND")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_coarse_odd_gabors(ctx: EditContext) -> None:
    import math
    from init_weights import LUMA, _odd_gabor
    for i, k in enumerate((0, 2, 4, 6)):
        kernel = LUMA.view(3, 1, 1) * _odd_gabor(math.pi * k / 8, sigma=2.0, wavelength=6.0)
        ctx.programmer.modify_parameter_group(
            "conv1.weight", (44 + i, slice(None), slice(None), slice(None)), kernel,
            f"conv1 filter {44 + i}: coarse-frequency (wavelength 6, sigma 2) ODD-phase Gabor at orientation {k}")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_more_odd_gabors(ctx: EditContext) -> None:
    """Replace DoG (32-35) and diagonal hues (40-43) with odd-phase Gabors at wavelength 6 for orientations 1,3,5,7 (+/-)."""
    import math
    from init_weights import LUMA, _odd_gabor
    slots = [(32, 1, +1.0), (33, 3, +1.0), (34, 5, +1.0), (35, 7, +1.0), (40, 1, -1.0), (41, 3, -1.0), (42, 5, -1.0), (43, 7, -1.0)]
    for slot, k, sign in slots:
        kernel = sign * LUMA.view(3, 1, 1) * _odd_gabor(math.pi * k / 8, sigma=2.0, wavelength=6.0)
        ctx.programmer.modify_parameter_group(
            "conv1.weight", (slot, slice(None), slice(None), slice(None)), kernel,
            f"conv1 filter {slot}: coarse odd-phase Gabor, orientation {k}, polarity {sign:+.0f}")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


ITERATIONS = [
    IterationSpec(
        id="iter001", title="Local-peakiness channels replace the coarse box copies (48 channels)",
        observed=BASELINE_OBS,
        problem="Coarse copies are near-linear functions of the fine features: only ~0.3 pt of new information for 768 "
                "extra head features (variance cost).",
        hypothesis="ReLU(f - 3x3 box mean of f) per conv1 map measures how much a feature stands out from its surroundings "
                   "(nonlinear, not available linearly) and raises val accuracy by >=1 point over 0.7843. "
                   "(WSP-003: peaks gave +0.7 on the 16-map net.)",
        selected="conv2.weight[48:96] (48x48x3x3 = 20,736) + head refit (1,537, lda_full)",
        expected="Gain if local peakiness is informative; fewer redundant features.",
        risks="Loses coarse marginals (+0.3 pt).",
        edit=edit_peaks_replace_coarse),
    IterationSpec(
        id="iter002", title="Soft-AND conjunction channels replace the coarse box copies (48 channels)",
        observed=BASELINE_OBS + " WSP-003: colour x orientation conjunctions were accepted under the diagonal head.",
        problem="Fine marginals cannot express co-occurrence (colour with orientation; neighbouring orientations).",
        hypothesis="48 conjunction channels (4 colour axes x 8 even orientations, 8 corner pairs k/k+45deg, 8 odd-phase "
                   "junction pairs k/k+90deg) add nonlinear information and raise val accuracy by >=1 point.",
        selected="conv2.weight+bias[48:96] (20,736+48) + head refit (1,537, lda_full)",
        expected="Gain if co-occurrence is informative.",
        risks="Scale constants are train-derived; scene-context colour cues.",
        edit=edit_conjunctions_replace_coarse),
    IterationSpec(
        id="iter003", title="Coarse-frequency ODD Gabors replace the weakest group (coarse even Gabors)",
        observed=BASELINE_OBS + " iter001/002 (nonlinear replacements of the coarse copies) both regressed.",
        problem="The 4 coarse-frequency EVEN Gabors are the weakest group (dropping them: val 0.787, +0.3), while odd-phase "
                "(step-edge) filters are the most valuable (dropping them: val 0.768).",
        hypothesis="Replacing those 4 slots with coarse ODD-phase Gabors (orientations 0,2,4,6, wavelength 6) adds edge "
                   "information and raises val accuracy by >=1 point over 0.7843.",
        selected="conv1.weight[44:48] (4x3x5x5 = 300) + head refit (1,537, lda_full)",
        expected="Gain if coarse step edges are informative.", risks="Only 4 orientations at the coarse band.",
        edit=edit_coarse_odd_gabors),
]
