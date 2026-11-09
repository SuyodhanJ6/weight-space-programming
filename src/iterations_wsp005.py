"""WSP-005 iterations: replication of the 20-image mission on a NEW 12/4/4 split (seed 5).

Hand-written, hypothesis-driven edits judged on train / val(4) / 5-fold train-CV only. Edits marked [closed-form head]
re-derive head weights by a one-shot formula over TRAIN statistics (data fitting, not hand programming).
With 4 validation images a one-image change is 25 points: conclusions are exploratory. Append-only.
"""

from __future__ import annotations

import torch

from evaluator import split_data
from wsp_engine import EditContext, IterationSpec

CENTRE_CELLS = {5, 6, 9, 10}
BASE_OBS = ("WSP-005 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500 (2 of 4 wrong: "
            "cat_07 scored dog +1.18, dog_09 scored cat -1.07), 5-fold CV 0.917. Head weights nearly all positive (mean +0.113); "
            "conv2 ReLU 93% active, conv3 100% active (near-linear); logit r=+0.80 with label, +0.17 with pixel energy.")


def _train_acts(ctx: EditContext) -> dict:
    images, _, _ = split_data("train")
    return ctx.model.forward_with_activations(images)


def edit_demean_head(ctx: EditContext) -> None:
    ctx.refit_head(rule="lda_demean", reason="remove global-energy direction from head")


def edit_conv2_gate(ctx: EditContext) -> None:
    bias = -_train_acts(ctx)["conv2"].mean(dim=(0, 2, 3))
    ctx.programmer.modify_parameter_group("conv2.bias", (slice(None),), bias,
                                          "bias_c = -mean(pre-activation_c over train): ReLU gates on above-average evidence")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_colour_wiring(ctx: EditContext) -> None:
    for k in range(8):
        ctx.programmer.modify_parameter_group(
            "conv2.weight", (24 + k, 12 + k % 4, slice(None), slice(None)), torch.full((3, 3), 1.0 / 9),
            f"wire colour-opponent channel {12 + k % 4} into conv2 blob+edge filter {24 + k}")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_box_conv3(ctx: EditContext) -> None:
    w3 = torch.zeros(32, 32, 3, 3)
    for c in range(32):
        w3[c, c] = 1.0 / 9
    ctx.programmer.replace_parameter("conv3.weight", w3,
                                     "conv3: 3x3 box (1/9) per channel instead of the 1-2-1 blur: wider spatial pooling")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_pooled_head(ctx: EditContext) -> None:
    ctx.refit_head(rule="lda_pooled", reason="pooled-over-cells head: weight shared across the 16 pooled cells")


def edit_centre_cells(ctx: EditContext) -> None:
    mask = [1.0 if c in CENTRE_CELLS else 0.25 for c in range(16)]
    ctx.refit_head(cell_mask=mask, reason="down-weight border cells (background) to 0.25")


def edit_centroid(ctx: EditContext) -> None:
    ctx.refit_head(rule="centroid", reason="class-mean difference without per-feature variance scaling")


def edit_conv3_gate_half(ctx: EditContext) -> None:
    bias = -0.5 * _train_acts(ctx)["conv3"].mean(dim=(0, 2, 3))
    ctx.programmer.modify_parameter_group("conv3.bias", (slice(None),), bias,
                                          "half-strength centring: bias_c = -0.5*mean(pre-activation_c)")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_lda_full(ctx: EditContext) -> None:
    ctx.refit_head(rule="lda_full", reason="full-covariance LDA (ridge) on train")


ITERATIONS = [
    IterationSpec(
        id="iter001", title="De-mean the head (replication of WSP-001 iter001 on a new split)",
        observed=BASE_OBS,
        problem="Mostly-positive head weights make the logit partly a feature-energy meter.",
        hypothesis="Subtracting the mean head weight removes the energy direction and raises val accuracy by >=1 image "
                   "(0.25). Replication: WSP-001 saw lower train accuracy and no val gain; the new split tests whether that was chance.",
        selected="head.weight (512), head.bias (1) [closed-form head]",
        expected="Lower train accuracy, possible val gain.", risks="Discards a legitimate dog-leaning energy signal.",
        edit=edit_demean_head),
    IterationSpec(
        id="iter002", title="Negative conv2 biases to restore ReLU gating",
        observed=BASE_OBS,
        problem="conv2 is 93% active and conv3 100% active, so the stack is nearly linear and adds no selectivity.",
        hypothesis="bias_c = -mean(pre-activation_c over train) makes conv2 fire only for above-average evidence and lifts val >= 1 image.",
        selected="conv2.bias (32) + head refit (513)",
        expected="Sparser conv2 activations, cleaner separation.", risks="Train-mean centring on 12 images may encode them.",
        edit=edit_conv2_gate),
    IterationSpec(
        id="iter003", title="Wire the dead colour pathway into conv2",
        observed=BASE_OBS + " conv2 weights from colour channels 12-15 are exactly zero: the net is colour-blind.",
        problem="Colour is computed in conv1 and discarded.",
        hypothesis="Wiring colour-opponent channels 12-15 into conv2 filters 24-31 lets warm/cool evidence reach the head (val >= +1 image).",
        selected="conv2.weight[24:32, 12:16] (72) + head refit (513)",
        expected="Possible gain; colour may be a spurious cue in 12 images.", risks="Black dogs/cats; background colour.",
        edit=edit_colour_wiring),
    IterationSpec(
        id="iter004", title="Wider spatial pooling in conv3 (3x3 box instead of 1-2-1 blur)",
        observed=BASE_OBS + " Web photos vary in pose and position; the 1-2-1 blur has an effective footprint of about 2 pixels.",
        problem="Only 12 training images: features tied to exact position transfer badly to new poses.",
        hypothesis="A uniform 3x3 box in conv3 widens the receptive field (more position tolerance) and raises val by >= 1 image.",
        selected="conv3.weight (9,216) + head refit (513)",
        expected="Small spatial smoothing effect.", risks="Blurs fine texture cues.",
        edit=edit_box_conv3),
    IterationSpec(
        id="iter005", title="Pooled-cell head (shared weights over the 16 pooled cells)",
        observed=BASE_OBS + " The 512-weight head has 513 free values fit to 12 images.",
        problem="Per-cell head weights are far more parameters than the data supports.",
        hypothesis="Sharing one weight per channel over the 16 cells (32 weights) reduces variance and raises val by >= 1 image.",
        selected="head.weight (512), head.bias (1) [closed-form head]",
        expected="Less overfitting (lower train accuracy).", risks="Loses spatial layout information.",
        edit=edit_pooled_head),
    IterationSpec(
        id="iter006", title="Down-weight border cells of the head (background suppression)",
        observed=BASE_OBS + " Border cells hold fences, cages and grass.",
        problem="Background may correlate with class in 12 images.",
        hypothesis="Scaling border-cell head weights by 0.25 reduces background reliance and lifts val by >= 1 image.",
        selected="head.weight border-cell entries (384 of 512), head.bias [closed-form head]",
        expected="Less background sensitivity.", risks="Off-centre animals lose signal.",
        edit=edit_centre_cells),
    IterationSpec(
        id="iter007", title="Centroid head rule (no variance scaling)",
        observed=BASE_OBS + " With n=12 the per-feature variance estimates are very noisy.",
        problem="Dividing by noisy variances amplifies unreliable low-variance features.",
        hypothesis="The plain class-mean difference transfers better and lifts val by >= 1 image.",
        selected="head.weight (512), head.bias (1) [closed-form head]",
        expected="Higher val if variance estimates were the noise.", risks="Ignores feature scale differences.",
        edit=edit_centroid),
    IterationSpec(
        id="iter008", title="Half-strength conv3 bias centring",
        observed=BASE_OBS + " conv3 is 100% active: the last layer never gates.",
        problem="Last-layer features are linear in conv2 output.",
        hypothesis="Half-strength centring gates only the weakest conv3 evidence without killing channels; val +>=1 image.",
        selected="conv3.bias (32) + head refit (513)",
        expected="Mild sparsification.", risks="Interactions with earlier accepted edits unknown.",
        edit=edit_conv3_gate_half),
    IterationSpec(
        id="iter009", title="Full-covariance LDA head (ridge)",
        observed=BASE_OBS + " Diagonal LDA ignores correlation between channels; WSP-003 found full-covariance LDA gave most of its gain at n=8,000.",
        problem="Highly correlated features are double-counted by a diagonal head.",
        hypothesis="Full-covariance LDA with ridge 0.05 x mean variance de-correlates them. With only 12 samples (rank <= 11) it may overfit; "
                   "tested as a directional check.",
        selected="head.weight (512), head.bias (1) [closed-form head]",
        expected="Likely train ~100% and val unchanged or worse (n << features).", risks="Severe overfit.",
        edit=edit_lda_full),
]


# ---------------------------------------------------------------------------------------------------------------
# Round 2 (iter010-iter019): written AFTER the sealed test was read (WSP-005 test: initial = final = 3/4). These are
# EXPLORATORY: judged on train/val/CV only, the test result is not used and cannot change. All start from the baseline
# (no edit was ever accepted). Each is one directional test, not a sweep.
# ---------------------------------------------------------------------------------------------------------------
import math  # noqa: E402

R2_OBS = BASE_OBS + " Round 1: 9/9 edits rejected; only iter001 raised val (0.5->0.75) but CV fell 0.917->0.5."


def edit_cb_head(ctx: EditContext) -> None:
    ctx.refit_head(rule="lda_pooled_cb", reason="centre/border-pooled head: 64 shared weights (4 centre cells, 12 border cells)")


def edit_drop_colour(ctx: EditContext) -> None:
    ctx.programmer.modify_parameter_group(
        "conv1.weight", (slice(12, 16), slice(None), slice(None), slice(None)), torch.zeros(4, 3, 5, 5),
        "conv1 filters 12-15 (colour opponent) set to 0: remove colour evidence")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_drop_dog(ctx: EditContext) -> None:
    ctx.programmer.modify_parameter_group(
        "conv1.weight", (slice(8, 12), slice(None), slice(None), slice(None)), torch.zeros(4, 3, 5, 5),
        "conv1 filters 8-11 (centre-surround blobs) set to 0: remove blob evidence")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_odd_gabors(ctx: EditContext) -> None:
    from init_weights import LUMA, _odd_gabor
    for i, k in enumerate((0, 2, 4, 6)):
        kernel = LUMA.view(3, 1, 1) * _odd_gabor(math.pi * k / 8, sigma=2.0, wavelength=6.0)
        ctx.programmer.modify_parameter_group(
            "conv1.weight", (8 + i, slice(None), slice(None), slice(None)), kernel,
            f"conv1 filter {8 + i}: coarse odd-phase (step-edge) Gabor, orientation {k}/8*pi, replaces a blob filter")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_drop_parallel(ctx: EditContext) -> None:
    ctx.programmer.modify_parameter_group(
        "conv2.weight", (slice(16, 24), slice(None), slice(None), slice(None)), torch.zeros(8, 16, 3, 3),
        "conv2 filters 16-23 (parallel-pair detectors) set to 0")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_drop_energy(ctx: EditContext) -> None:
    ctx.programmer.modify_parameter_group(
        "conv3.weight", (slice(24, 32), slice(None), slice(None), slice(None)), torch.zeros(8, 32, 3, 3),
        "conv3 filters 24-31 (isotropic energy channels) set to 0")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_centre_only(ctx: EditContext) -> None:
    ctx.refit_head(cell_mask=[1.0 if c in CENTRE_CELLS else 0.0 for c in range(16)],
                   reason="head uses only the 4 centre pooled cells (border weight 0)")


def edit_top_half(ctx: EditContext) -> None:
    ctx.refit_head(cell_mask=[1.0 if c < 8 else 0.25 for c in range(16)],
                   reason="head emphasises the top two pooled rows (heads/faces), bottom rows x0.25")


def edit_conv3_kernels(ctx: EditContext) -> None:
    kernels = [torch.tensor(k, dtype=torch.float32) for k in (
        [[0, 1, 0], [0, 1, 0], [0, 1, 0]], [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        [[1, 0, 0], [1, 0, 0], [1, 0, 0]], [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        [[1, 1, 1], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [1, 1, 1]])]
    for i, kern in enumerate(kernels):
        block = (kern / kern.sum() / 8).expand(8, 3, 3)
        ctx.programmer.modify_parameter_group(
            "conv3.weight", (24 + i, slice(24, 32), slice(None), slice(None)), block,
            f"conv3 filter {24 + i}: distinct spatial kernel over the 8 blob+edge channels")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_demean_pooled_cb(ctx: EditContext) -> None:
    ctx.refit_head(rule="lda_pooled", cell_mask=[1.0] * 16, reason="pooled head again with explicit uniform mask (control for mask path)")


ROUND2 = [
    IterationSpec(
        id="iter010", title="[exploratory] Centre/border pooled head (64 shared weights)",
        observed=R2_OBS + " iter005 (32 shared weights) lowered CV 0.917->0.833 but kept val.",
        problem="Fully shared cells lose the centre-vs-border distinction that separates subject from background.",
        hypothesis="Two shared weights per channel (centre cells, border cells) keep subject/background while cutting parameters; val +>=1 image.",
        selected="head.weight (512), head.bias (1) [closed-form head]", expected="Between iter005 and baseline.",
        risks="64 features still large for 12 images.", edit=edit_cb_head),
    IterationSpec(
        id="iter011", title="[exploratory] Remove colour-opponent filters from conv1",
        observed=R2_OBS + " iter003 (wiring colour into conv2) only raised train to 1.0, a memorisation sign.",
        problem="Colour may be a scene cue (fur colour, background) that does not generalise from 12 images.",
        hypothesis="Zeroing conv1 filters 12-15 removes colour evidence from the pooled features; val +>=1 image if colour was noise.",
        selected="conv1.weight[12:16] (300) + head refit (513)", expected="Less memorisation.", risks="Loses a real cue.",
        edit=edit_drop_colour),
    IterationSpec(
        id="iter012", title="[exploratory] Remove centre-surround blob filters from conv1",
        observed=R2_OBS,
        problem="Blob filters 8-11 respond to local contrast spots (eyes, noses, but also fur speckle).",
        hypothesis="Removing them reduces noise features; val +>=1 image.",
        selected="conv1.weight[8:12] (300) + head refit (513)", expected="Fewer redundant features.", risks="Loses eye/nose cues.",
        edit=edit_drop_dog),
    IterationSpec(
        id="iter013", title="[exploratory] Coarse odd-phase Gabors replace the blob filters",
        observed=R2_OBS + " WSP-004 found odd-phase (step-edge) filters the most valuable group at n=8,000.",
        problem="Even-phase Gabors (0-7) respond to bars; step edges (object boundaries) are not represented.",
        hypothesis="Odd-phase Gabors at 4 orientations replace blobs 8-11 and add silhouette-edge information; val +>=1 image.",
        selected="conv1.weight[8:12] (300) + head refit (513)", expected="Edge-boundary evidence.", risks="n=12 cannot reveal a small gain.",
        edit=edit_odd_gabors),
    IterationSpec(
        id="iter014", title="[exploratory] Remove conv2 parallel-pair detectors",
        observed=R2_OBS,
        problem="Parallel-pair filters (16-23) respond to texture stripes common to both classes.",
        hypothesis="Zeroing them lowers texture noise; val +>=1 image.",
        selected="conv2.weight[16:24] (1,152) + head refit (513)", expected="Less texture dependence.", risks="Loses fur-texture cue.",
        edit=edit_drop_parallel),
    IterationSpec(
        id="iter015", title="[exploratory] Remove conv3 isotropic-energy channels",
        observed=R2_OBS + " Baseline logit correlates only +0.17 with pixel energy, but 8 of 32 conv3 channels are pure energy meters.",
        problem="Energy channels (24-31) are the same quantity repeated 8 times.",
        hypothesis="Zeroing conv3 24-31 removes redundant energy features; val +>=1 image.",
        selected="conv3.weight[24:32] (2,304) + head refit (513)", expected="Smaller redundant block.", risks="Energy is partly informative.",
        edit=edit_drop_energy),
    IterationSpec(
        id="iter016", title="[exploratory] Centre-only head (border weight 0)",
        observed=R2_OBS + " iter006 (border x0.25) changed nothing.",
        problem="iter006 may simply have been too weak a test of the background hypothesis.",
        hypothesis="Setting border weights to exactly 0 removes background entirely; val +>=1 image if background is a spurious cue.",
        selected="head.weight (512 incl. 384 zeroed), head.bias", expected="Subject-only decision.", risks="Off-centre subjects.",
        edit=edit_centre_only),
    IterationSpec(
        id="iter017", title="[exploratory] Top-half emphasis in the head",
        observed=R2_OBS + " Faces and heads sit in the upper half of most framed pet photos; bodies/floors in the lower half.",
        problem="Floor/grass texture in the lower rows is class-uninformative.",
        hypothesis="Weighting the top two pooled rows x1 and the bottom two x0.25 sharpens face evidence; val +>=1 image.",
        selected="head.weight (512), head.bias [closed-form head]", expected="Less floor influence.", risks="Arbitrary prior about framing.",
        edit=edit_top_half),
    IterationSpec(
        id="iter018", title="[exploratory] Break conv3 filter symmetry (replication of WSP-001 iter006)",
        observed=R2_OBS + " conv3 filters 24-31 produce identical outputs, so the head counts one feature 8 times.",
        problem="8 identical channels waste capacity.",
        hypothesis="Eight distinct spatial kernels (vertical, horizontal, diagonals, left, right, top, bottom) give distinct layout features; val +>=1 image.",
        selected="conv3.weight[24:32, 24:32] (576) + head refit (513)", expected="More informative last-layer features.",
        risks="Encodes arbitrary position.", edit=edit_conv3_kernels),
    IterationSpec(
        id="iter019", title="[exploratory] Control: pooled head with explicit uniform cell mask",
        observed=R2_OBS + " iter005's regression could come from the pooled rule or from the head-fitting path.",
        problem="Need to separate the effect of the pooled rule from a bookkeeping artefact.",
        hypothesis="A pooled head built with an explicit all-ones mask reproduces iter005 exactly (train .833, CV .833); a difference would mean a code-path bug.",
        selected="head.weight (512), head.bias [closed-form head]", expected="Identical to iter005 -> rejected again.", risks="None; diagnostic.",
        edit=edit_demean_pooled_cb),
]
ITERATIONS = ITERATIONS + ROUND2
