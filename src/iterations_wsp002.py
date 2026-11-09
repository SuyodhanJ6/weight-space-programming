"""WSP-002 iterations: explicit, hypothesis-driven edits authored by the reasoning agent.

Append-only. Evaluated on train + validation (600 images) only; the sealed test split is never
read. Edits marked [closed-form head] reassign head weights by a one-shot formula over TRAIN
statistics: that is data fitting, not hand programming, and is reported separately.
"""

from __future__ import annotations

import torch

import iterations as v1  # reuse the (unchanged) wsp-001 edit functions for replications
from wsp_engine import EditContext, IterationSpec

BOTTOM_ROW = {12, 13, 14, 15}


def edit_bottom_row(ctx: EditContext) -> None:
    mask = [0.25 if c in BOTTOM_ROW else 1.0 for c in range(16)]
    ctx.refit_head(cell_mask=mask, reason="down-weight bottom pooled row (floor/ground) to 0.25")


def edit_identity_routing(ctx: EditContext) -> None:
    spatial = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16
    for j in range(8):
        zero = torch.zeros(32, 3, 3)
        zero[24 + j] = spatial
        ctx.programmer.modify_parameter_group(
            "conv3.weight", (24 + j, slice(None), slice(None), slice(None)), zero,
            f"conv3 filter {24 + j} reads ONLY conv2 channel {24 + j} (identity routing)")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_colour_corner(ctx: EditContext) -> None:
    for k in range(8):
        ctx.programmer.modify_parameter_group(
            "conv2.weight", (8 + k, 12 + k % 4, slice(None), slice(None)), torch.full((3, 3), 1.0 / 9),
            f"wire colour-opponent channel {12 + k % 4} into conv2 corner filter {8 + k}")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_coarse_gabor(ctx: EditContext) -> None:
    import math
    from init_weights import LUMA, NUM_ORIENTATIONS, _gabor
    for k in range(NUM_ORIENTATIONS):
        kernel = LUMA.view(3, 1, 1) * _gabor(math.pi * k / NUM_ORIENTATIONS, sigma=2.0, wavelength=6.0)
        ctx.programmer.modify_parameter_group(
            "conv1.weight", (k, slice(None), slice(None), slice(None)), kernel,
            f"conv1 oriented filter {k}: Gabor wavelength 4->6 px, sigma 1.5->2.0 (coarser structure)")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_border_quarter(ctx: EditContext) -> None:
    mask = [1.0 if c in {5, 6, 9, 10} else 0.25 for c in range(16)]
    ctx.refit_head(cell_mask=mask, reason="scale the 12 border cells' head weights by 0.25")


PRUNE_COUNT = 8


def edit_prune_weak_channels(ctx: EditContext) -> None:
    from evaluator import split_data
    images, labels, _ = split_data("train")                 # TRAIN only
    gap = ctx.model.forward_with_activations(images)["gap"].view(-1, 32, 16)
    cat, dog = gap[labels == 0], gap[labels == 1]
    d = ((dog.mean(0) - cat.mean(0)) / ((cat.var(0) + dog.var(0)) / 2).sqrt().clamp_min(1e-6)).abs().mean(1)
    weakest = d[:24].argsort()[:PRUNE_COUNT].tolist()       # channels 24-31 are identical: excluded
    ctx.notes.append(f"pruned conv3 channels {weakest} (lowest mean |d| on train)")
    zeros = torch.zeros(len(weakest), 32, 3, 3)
    ctx.programmer.modify_parameter_group("conv3.weight", (torch.tensor(weakest),), zeros,
                                          f"zero the {PRUNE_COUNT} weakest conv3 filters {weakest}")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_colour_box(ctx: EditContext) -> None:
    from init_weights import _l1
    box = _l1(torch.ones(5, 5))
    opp = [torch.tensor([1.0, -1.0, 0.0]), torch.tensor([-0.5, -0.5, 1.0])]
    for i, vec in enumerate(opp):
        for pol, idx in ((1.0, 12 + 2 * i), (-1.0, 13 + 2 * i)):
            ctx.programmer.modify_parameter_group(
                "conv1.weight", (idx, slice(None), slice(None), slice(None)),
                pol * vec.view(3, 1, 1) * box, f"conv1 colour filter {idx}: 5x5 box average (was sigma 1.5 Gaussian)")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


# ---- Round 2: post-test exploratory (sealed test already read once; judged on train/val only) ----

def edit_colour_parallel(ctx: EditContext) -> None:
    for k in range(8):
        ctx.programmer.modify_parameter_group(
            "conv2.weight", (16 + k, 12 + k % 4, slice(None), slice(None)), torch.full((3, 3), 1.0 / 9),
            f"wire colour-opponent channel {12 + k % 4} into conv2 parallel-pair filter {16 + k}")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_half_bias(ctx: EditContext) -> None:
    pre = _pre_conv2(ctx)
    ctx.programmer.modify_parameter_group(
        "conv2.bias", (slice(None),), -0.5 * pre.mean(dim=(0, 2, 3)),
        "half-strength centring: bias_c = -0.5*mean(pre-activation_c over train)")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def _pre_conv2(ctx: EditContext) -> torch.Tensor:
    from evaluator import split_data
    images, _, _ = split_data("train")
    return ctx.model.forward_with_activations(images)["conv2"]


def edit_pure_colour_blob(ctx: EditContext) -> None:
    ctx.programmer.modify_parameter_group(
        "conv2.weight", (slice(24, 32), slice(0, 8), slice(None), slice(None)), 0.0,
        "remove the oriented-edge inputs from blob+colour filters 24-31 (keep DoG + colour only)")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


PAIR_OFFSETS = [((1, 0), (1, 2)),   # horizontal pair
                ((0, 1), (2, 1)),   # vertical pair
                ((0, 0), (2, 2)),   # diagonal pair
                ((0, 2), (2, 0))]   # anti-diagonal pair


def _blob_pair_filters(ctx: EditContext, filters: list[int], source_channel: int, what: str) -> None:
    """conv3 filter f fires where two blobs of conv2 channel `source_channel` sit symmetrically
    about the centre (positive on both, negative at the centre), i.e. a blob-pair detector."""
    for f, (p1, p2) in zip(filters, PAIR_OFFSETS):
        block = torch.zeros(32, 3, 3)
        block[source_channel, p1[0], p1[1]] = 0.5
        block[source_channel, p2[0], p2[1]] = 0.5
        block[source_channel, 1, 1] = -0.25
        ctx.programmer.modify_parameter_group(
            "conv3.weight", (f, slice(None), slice(None), slice(None)), block,
            f"conv3 filter {f}: {what} pair detector at offsets {p1},{p2}")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_bright_blob_pairs(ctx: EditContext) -> None:
    _blob_pair_filters(ctx, [25, 26, 27, 28], 24, "bright-blob (centre-surround on-centre, eye/glint)")


def edit_dark_blob_pairs(ctx: EditContext) -> None:
    _blob_pair_filters(ctx, [29, 30, 31], 25, "dark-blob (off-centre, nostril/dark eye)")


def edit_larger_blobs(ctx: EditContext) -> None:
    from init_weights import LUMA, _dog
    for idx, pol in ((10, 1), (11, -1)):
        kernel = LUMA.view(3, 1, 1) * _dog(1.5, 3.0, pol)
        ctx.programmer.modify_parameter_group(
            "conv1.weight", (idx, slice(None), slice(None), slice(None)), kernel,
            f"conv1 centre-surround filter {idx}: DoG sigma (0.8,1.6) -> (1.5,3.0) (eye/nose-sized blobs)")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


# ---- Round 4: user-directed continuation at n=600 (gains cannot be verified on unseen data) ----

def edit_odd_gabor(ctx: EditContext) -> None:
    import math
    from init_weights import LUMA, NUM_ORIENTATIONS, _grid, _l1
    x, y = _grid(5)
    for k in range(NUM_ORIENTATIONS):
        theta = math.pi * k / NUM_ORIENTATIONS
        xr = x * math.cos(theta) + y * math.sin(theta)
        kern = torch.exp(-(x**2 + y**2) / (2 * 1.5**2)) * torch.sin(2 * math.pi * xr / 4.0)
        ctx.programmer.modify_parameter_group(
            "conv1.weight", (k, slice(None), slice(None), slice(None)), LUMA.view(3, 1, 1) * _l1(kern),
            f"conv1 oriented filter {k}: even (cosine) -> odd (sine) Gabor: step-edge/outline detector")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_sharp_orientation(ctx: EditContext) -> None:
    n = 8
    for k in range(n):
        for nb in ((k + 1) % n, (k - 1) % n):
            ctx.programmer.modify_parameter(
                "conv2.weight", (k, nb, 1, 1), 0.0,
                f"line filter {k}: drop the 0.5 tap from neighbouring orientation channel {nb}")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_half_bias_conv3(ctx: EditContext) -> None:
    from evaluator import split_data
    images, _, _ = split_data("train")
    pre = ctx.model.forward_with_activations(images)["conv3"]
    ctx.programmer.modify_parameter_group(
        "conv3.bias", (slice(None),), -0.5 * pre.mean(dim=(0, 2, 3)),
        "half-strength centring: bias_c = -0.5*mean(pre-activation_c over train)")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_passthrough(ctx: EditContext) -> None:
    """conv2/conv3 become identity channel routes for the 16 conv1 features (hierarchy ablation)."""
    w2 = torch.zeros(32, 16, 3, 3)
    w3 = torch.zeros(32, 32, 3, 3)
    spatial = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16
    for j in range(16):
        w2[j, j, 1, 1] = 1.0
        w3[j, j] = spatial
    ctx.programmer.replace_parameter("conv2.weight", w2, "conv2: channel j <- conv1 channel j (centre tap), channels 16-31 = 0")
    ctx.programmer.replace_parameter("conv3.weight", w3, "conv3: channel j <- conv2 channel j (blur), channels 16-31 = 0")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_multiscale(ctx: EditContext) -> None:
    """conv2: channels 0-15 pass conv1 features; channels 16-31 hold their 3x3 box average (coarser scale)."""
    w2 = torch.zeros(32, 16, 3, 3)
    w3 = torch.zeros(32, 32, 3, 3)
    spatial = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16
    for j in range(16):
        w2[j, j, 1, 1] = 1.0
        w2[16 + j, j] = 1.0 / 9
    for c in range(32):
        w3[c, c] = spatial
    ctx.programmer.replace_parameter("conv2.weight", w2, "conv2: ch j <- conv1 ch j (fine); ch 16+j <- 3x3 box mean of conv1 ch j (coarse)")
    ctx.programmer.replace_parameter("conv3.weight", w3, "conv3: identity routing with blur for all 32 channels")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_luminance_ramps(ctx: EditContext) -> None:
    """conv1 filters 8-11 become single-pixel luminance ramps with thresholds (soft histogram bins)."""
    from init_weights import LUMA
    specs = [(+1.0, 0.4), (+1.0, 1.1), (-1.0, 0.4), (-1.0, 1.1)]  # (sign, threshold) in normalised luminance
    for i, (sign, thr) in enumerate(specs):
        kernel = torch.zeros(3, 5, 5)
        kernel[:, 2, 2] = sign * LUMA
        ctx.programmer.modify_parameter_group(
            "conv1.weight", (8 + i, slice(None), slice(None), slice(None)), kernel,
            f"conv1 filter {8 + i}: single-pixel luminance ramp, sign {sign:+.0f}")
        ctx.programmer.modify_parameter("conv1.bias", (8 + i,), -thr,
                                        f"conv1 filter {8 + i}: threshold {thr} (ReLU(sign*L - {thr}))")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


ITERATIONS = [
    IterationSpec(
        id="iter001", title="Remove global-energy direction from the head (replication at scale)",
        observed="On train (n=1000) the baseline logit correlates r=+0.91 with image edge density but "
                 "the label only +0.25; corr(logit,label)=0.26. Head weights are almost all positive.",
        problem="The classifier is an edge-density meter, not a cat/dog detector.",
        hypothesis="Subtracting the mean head weight removes the energy-following direction, so the "
                   "logit depends on which features differ between classes. (Rejected on 4 val images in "
                   "WSP-001; re-tested because 600 val images can now resolve a 1-point effect.)",
        selected="head.weight (512), head.bias (1) [closed-form head]",
        expected="Lower correlation of logit with edge density; val accuracy up if energy was a distractor.",
        risks="May remove legitimate dog-leaning energy (dogs have slightly more edges: r=+0.25).",
        edit=v1.edit_001),
    IterationSpec(
        id="iter002", title="Centre conv2 pre-activations with negative biases (replication)",
        observed="relu2 is 95% active and relu3 100%: ReLUs do not gate, so the network is nearly linear.",
        problem="No nonlinear feature selection occurs.",
        hypothesis="conv2.bias_c = -mean(pre-activation_c over train) makes units fire only for "
                   "above-average local evidence. (Accepted in WSP-001 on 16 images; re-tested at scale.)",
        selected="conv2.bias (32) + head refit (513) [bias is a data-derived constant; head closed-form]",
        expected="Sparser conv2 activations and more selective features.",
        risks="Mean centring uses train statistics; can zero out weak channels.",
        edit=v1.edit_002),
    IterationSpec(
        id="iter003", title="Centre conv3 pre-activations with negative biases (replication)",
        observed="relu3 is 100% active; every final feature contributes.",
        problem="Same linearity one layer deeper.",
        hypothesis="Centring conv3 pre-activations gates the final features. (Rejected in WSP-001 on 16 images.)",
        selected="conv3.bias (32) + head refit (513)",
        expected="More selective final descriptors.",
        risks="Over-sparsification of final features.",
        edit=v1.edit_003),
    IterationSpec(
        id="iter004", title="Wire the dead colour pathway into conv2 (replication)",
        observed="conv2 receives exactly 0.0 weight from the 4 colour channels. On this larger train set "
                 "warmth barely relates to label (r=+0.08), unlike the 20-image WSP-001 sample.",
        problem="Colour is discarded; but its value is doubtful at scale.",
        hypothesis="Connecting colour channels to conv2 filters 24-31 will NOT help (WSP-001's val gain was "
                   "one image). This is a falsification test of the WSP-001 colour finding.",
        selected="conv2.weight[24:32,12:16] (72) + head refit (513)",
        expected="No val improvement >=1 point; if one appears the WSP-001 finding replicates.",
        risks="Spurious colour correlation on 600 val images is unlikely but possible.",
        edit=v1.edit_004),
    IterationSpec(
        id="iter005", title="Suppress the bottom pooled row in the head",
        observed="Per-cell mean |effect size| on train: 0.43/0.42 for centre cells vs 0.10-0.20 for the "
                 "bottom row (ground/floor/furniture), 0.22-0.37 elsewhere.",
        problem="Bottom-row features carry little class signal but add variance (and background).",
        hypothesis="Scaling bottom-row head weights by 0.25 reduces background noise.",
        selected="head.weight bottom-row entries (128 of 512), head.bias [closed-form head with mask]",
        expected="Small val gain; risk of losing legs/paw cues.",
        risks="Animals whose lower body is informative.",
        edit=edit_bottom_row),
    IterationSpec(
        id="iter006", title="Identity routing for conv3 filters 24-31 (de-duplicate colour/blob features)",
        observed="After iter004, conv3 filters 24-31 each sum ALL eight conv2 blob/colour channels with "
                 "the same kernel, so they are identical: colour evidence reaches the head as one feature "
                 "repeated 8x (16 distinct values of 512).",
        problem="Colour/blob information is collapsed; the head cannot weight individual colour channels.",
        hypothesis="Routing conv3 filter 24+j to conv2 channel 24+j only gives eight distinct colour/blob "
                   "descriptors, so the closed-form head can use them separately and val accuracy rises.",
        selected="conv3.weight[24:32] (8x32x3x3 = 2304 values) + head refit (513)",
        expected="Val accuracy gain >=1 point if colour descriptors are individually informative.",
        risks="Position-specific loss of the pooled colour sum; more head variance (more distinct features).",
        edit=edit_identity_routing),
    IterationSpec(
        id="iter007", title="Amplify colour pathway x3 (replication of WSP-001 iter007)",
        observed="iter004 (colour) was the only accepted edit; colour weights are 1/9 per tap vs edge "
                 "weights up to 0.5.",
        problem="Colour evidence may be under-weighted relative to edge channels inside conv2.",
        hypothesis="Scaling the 72 wired colour weights by 3 strengthens colour influence and raises val accuracy.",
        selected="conv2.weight[24:32,12:16] (72) + head refit (513)",
        expected="Further gain if colour is genuinely informative; regression if it adds noise.",
        risks="Overweights a noisy cue.",
        edit=v1.edit_007),
    IterationSpec(
        id="iter008", title="Wire colour into conv2 corner filters 8-15",
        observed="Colour reaches conv3 only through the blob+edge path (filters 24-31).",
        problem="Colour is not combined with the corner/junction detectors.",
        hypothesis="Adding colour-opponent input to the corner filters lets the network represent "
                   "coloured corner/junction regions (e.g. nose, eyes against fur).",
        selected="conv2.weight[8:16,12:16] (72) + head refit (513)",
        expected="Val gain >=1 point if colour-structure conjunctions matter.",
        risks="Dilutes pure corner geometry.",
        edit=edit_colour_corner),
    IterationSpec(
        id="iter009", title="Coarser oriented edge filters in conv1 (wavelength 4 -> 6 px)",
        observed="The accepted model's logit still follows fine-scale edge density (baseline r=+0.91 with "
                 "mean |dI/dx|, only +0.25 with the label). Fine 4-px Gabors mostly measure texture/JPEG detail.",
        problem="Fine-scale edge energy is a weak, confounded cue; class-relevant structure (ear, eye, muzzle "
                "contours) lives at coarser scales (~6-8 px at 64x64).",
        hypothesis="Replacing the 8 oriented filters with wavelength-6 px, sigma-2.0 Gabors lowers dependence "
                   "on fine texture and raises val accuracy by >=1 point.",
        selected="conv1.weight[0:8] (8x3x5x5 = 600 values) + head refit (513)",
        expected="Lower correlation with fine edge density, modest val gain.",
        risks="Loses fine whisker/fur detail; 5x5 support truncates a 6-px wavelength.",
        edit=edit_coarse_gabor),
    IterationSpec(
        id="iter010", title="Centroid head rule (no per-feature variance scaling)",
        observed="Head has 512 features fitted from 1000 train images; centre-only edge features (96) match "
                 "the full head on val (59.5 vs 60.7%), so many features add noise.",
        problem="Dividing by noisy per-feature variances amplifies low-variance, uninformative features.",
        hypothesis="The plain class-mean difference (centroid rule) is more robust and raises val accuracy "
                   "by >=1 point. (Rejected in WSP-001 on 16 images.)",
        selected="head.weight (512), head.bias (1) [closed-form head]",
        expected="Higher val accuracy and CV accuracy if variance estimates were the noise source.",
        risks="Ignores feature scale; may favour high-variance features.",
        edit=lambda ctx: ctx.refit_head(rule="centroid", reason="class-mean difference without variance scaling")),
    IterationSpec(
        id="iter011", title="Scale border-cell head weights by 0.25",
        observed="Edge features restricted to the 4 centre cells reach val 59.5%, same as all 16 cells "
                 "within noise (+-2%), with 4x fewer features.",
        problem="Border cells add many correlated, background-dominated features and head variance.",
        hypothesis="Down-weighting the 12 border cells to 0.25 reduces variance and background reliance.",
        selected="head.weight border-cell entries (384 of 512), head.bias [closed-form head with mask]",
        expected="Small val gain (~1 point).",
        risks="Animal extremities near the border are lost.",
        edit=edit_border_quarter),
    IterationSpec(
        id="iter012", title="Prune the 8 weakest conv3 channels",
        observed="Head variance dominates: centre-only edge features (96) match all 512 on val. Per-channel "
                 "mean |effect size| on train varies ~2x between channels.",
        problem="Low-signal channels add variance to a head fitted from 1000 images.",
        hypothesis="Zeroing the 8 conv3 filters (among 0-23) with the lowest mean |d| on TRAIN lowers head "
                   "variance and raises val accuracy by >=1 point.",
        selected="conv3.weight rows for the 8 weakest channels (8x32x3x3 = 2304 values) + head refit (513)",
        expected="Small val gain; CV roughly unchanged.",
        risks="Selection uses train effect sizes (data-derived); may remove channels useful in combination.",
        edit=edit_prune_weak_channels),
    IterationSpec(
        id="iter013", title="Average colour over a larger area (Gaussian sigma 1.5 -> 5x5 box)",
        observed="Colour wiring (iter004) was the only accepted edit (+3.5 pts val). Colour is a region "
                 "property; conv1 colour filters are small Gaussians, exposed to JPEG chroma noise.",
        problem="Colour channels have low signal-to-noise at pixel level.",
        hypothesis="A flat 5x5 box average of the colour-opponent signal yields a more reliable colour "
                   "descriptor and a further >=1 point val gain.",
        selected="conv1.weight[12:16] (4x3x5x5 = 300 values) + head refit (513)",
        expected="Modest additional gain from cleaner colour.",
        risks="Blurs colour boundaries; box filter has hard edges.",
        edit=edit_colour_box),
    IterationSpec(
        id="iter014", title="[round 2] Wire colour into conv2 parallel-pair filters 16-23",
        observed="Colour entered via the blob path (iter004, accepted) but not via corner filters (iter008, rejected).",
        problem="Colour is available to only one of four conv2 filter families.",
        hypothesis="Giving the parallel-pair (stripe/texture) filters colour-opponent input adds colour-textured "
                   "structure and raises val accuracy by >=1 point. Completes the colour-routing family.",
        selected="conv2.weight[16:24,12:16] (72) + head refit (513)",
        expected="Small gain if stripe+colour conjunctions matter (e.g. fur patterns).",
        risks="Same dilution that sank iter008.",
        edit=edit_colour_parallel),
    IterationSpec(
        id="iter015", title="[round 2] Half-strength conv2 bias gating on the accepted model",
        observed="Full centring (iter002) moved val +0.7 pts but CV -0.5 pts: borderline, not rejected for harm.",
        problem="Full centring may over-sparsify; ReLU2 is still ~95% active.",
        hypothesis="Half-strength centring keeps some selectivity without regressions and gives a >=1 point val gain.",
        selected="conv2.bias (32) + head refit (513)",
        expected="Small gain.", risks="Likely within noise.", edit=edit_half_bias),
    IterationSpec(
        id="iter016", title="[round 2] Remove oriented-edge inputs from the colour/blob filters",
        observed="Blob+colour filters 24-31 carry gabor/9 edge input plus DoG plus colour; the edge part duplicates "
                 "channels 0-23, and colour is the only modality with proven value.",
        problem="The colour signal is diluted by a redundant edge term inside the same filters.",
        hypothesis="Zeroing the gabor taps (conv2.weight[24:32,0:8]) yields cleaner colour/DoG descriptors and "
                   "raises val accuracy by >=1 point.",
        selected="conv2.weight[24:32,0:8] (8x8x3x3 = 576 values) + head refit (513)",
        expected="Cleaner colour features; possible loss of edge-conditioned blobs.",
        risks="Loses blob+edge conjunction.", edit=edit_pure_colour_blob),
    IterationSpec(
        id="iter017", title="[round 3] Bright blob-pair detectors (eyes) in conv3 filters 25-28",
        observed="All earlier edits re-weighted generic edge/colour features; none encoded facial structure. conv3 "
                 "filters 25-31 are still redundant duplicates of filter 24 (wasted capacity).",
        problem="No feature represents two symmetric blobs (eyes), a face cue: cats have large widely-spaced "
                "bright/glinting eyes, dogs smaller darker ones.",
        hypothesis="Programming 4 conv3 filters as horizontal/vertical/diagonal pair detectors over the bright-blob "
                   "channel (conv2 ch 24) gives the head a face-structure feature and lifts val accuracy by >=1 point.",
        selected="conv3.weight[25:29] (4x32x3x3 = 1152 values) + head refit (513)",
        expected="Gain if eye pairs fall inside the pooled 3x3 neighbourhood at 16x16 (~8 px at 64x64).",
        risks="Pair separation (8 px) may be too small/large for most images; 24 itself is an edge+blob+colour mix.",
        edit=edit_bright_blob_pairs),
    IterationSpec(
        id="iter018", title="[round 3] Dark blob-pair detectors (nostril/dark eyes) in conv3 filters 29-31",
        observed="Dogs' eyes and noses are dark blobs on lighter fur; the off-centre DoG channel (conv2 ch 25) is unused as a pair.",
        problem="Dark paired structure is not represented.",
        hypothesis="Pair detectors over the dark-blob channel add the complementary face cue and raise val accuracy >=1 point.",
        selected="conv3.weight[29:32] (3x32x3x3 = 864 values) + head refit (513)",
        expected="Gain if dark blob pairs separate dogs from cats.",
        risks="Dark backgrounds create many spurious pairs.",
        edit=edit_dark_blob_pairs),
    IterationSpec(
        id="iter019", title="[round 3] Larger centre-surround blobs in conv1 (sigma 0.8 -> 1.5)",
        observed="The two small-scale DoG filters (sigma 0.8/1.6, ~3 px blobs) feed the blob+colour path; at 64x64 an "
                 "eye or nose is a ~4-8 px blob.",
        problem="The blob detectors respond to pixel-scale texture and JPEG noise more than to facial parts.",
        hypothesis="Matching DoG scale to part size (sigma 1.5/3.0) makes the blob channels respond to parts and "
                   "raises val accuracy by >=1 point.",
        selected="conv1.weight[10:12] (2x3x5x5 = 150 values) + head refit (513)",
        expected="Cleaner part-scale blob evidence.",
        risks="5x5 support truncates a sigma-3 surround; less fine detail.",
        edit=edit_larger_blobs),
    IterationSpec(
        id="iter020", title="[round 4] Odd-phase (sine) Gabors in conv1: step-edge detectors",
        observed="conv1 oriented filters are cos-phase (even): they respond to thin bars, not to step edges such as "
                 "body/ear/muzzle outlines against background.",
        problem="Outline evidence is under-represented.",
        hypothesis="Replacing the 8 even Gabors with odd sine-phase Gabors (same wavelength 4, sigma 1.5) detects "
                   "silhouette edges and raises val accuracy by >=1 point.",
        selected="conv1.weight[0:8] (600 values) + head refit (513)",
        expected="Outline-driven features; may lose bar/whisker response.",
        risks="Polarity-sensitive: ReLU keeps only one edge polarity per orientation (8 not 16 channels).",
        edit=edit_odd_gabor),
    IterationSpec(
        id="iter021", title="[round 4] Sharper orientation tuning in conv2 line filters",
        observed="conv2 line filters 0-7 also take 0.5 weight from the two neighbouring orientation channels at the centre tap.",
        problem="Broad tuning blurs orientation, reducing the diversity of the 8 line channels.",
        hypothesis="Removing the neighbour taps makes each line channel orientation-specific and raises val accuracy by >=1 point.",
        selected="conv2.weight line-filter neighbour taps (16 values) + head refit (513)",
        expected="More distinct line channels.", risks="Noisier, less shift/rotation tolerance.",
        edit=edit_sharp_orientation),
    IterationSpec(
        id="iter022", title="[round 4] Half-strength conv3 bias gating on the accepted model",
        observed="relu3 is 100% active; full centring (iter003) regressed CV by 2.4 pts.",
        problem="Final features never gate.",
        hypothesis="Half-strength centring adds mild gating without the regression and raises val accuracy by >=1 point.",
        selected="conv3.bias (32) + head refit (513)",
        expected="Small selectivity gain.", risks="Likely within noise.",
        edit=edit_half_bias_conv3),
    IterationSpec(
        id="iter023", title="[round 4] Hierarchy ablation: pass conv1 features straight through conv2/conv3",
        observed="22 edits of every kind left val at 60-61%; the head (closed-form on 512 features) may be doing all the work.",
        problem="Unknown whether the hand-composed line/corner/parallel/blob hierarchy contributes anything.",
        hypothesis="Replacing conv2/conv3 with identity routing of the 16 conv1 channels (256 informative features, "
                   "256 zero) gives accuracy within 1 point of the accepted model, i.e. the hierarchy adds nothing; "
                   "if it *raises* val by >=1 point (fewer noisy features) it is accepted.",
        selected="conv2.weight (4608) + conv3.weight (9216) replaced + head refit (513)",
        expected="Roughly equal accuracy (hierarchy decorative) or slightly better from lower head variance.",
        risks="Loses the colour wiring's blob-path structure (colour reaches the head only through channels 12-15).",
        edit=edit_passthrough),
    IterationSpec(
        id="iter024", title="[round 5] Multi-scale passthrough: 16 fine + 16 coarse conv1 feature maps",
        observed="iter023: the hand-composed hierarchy is decorative; the conv1 features + closed-form head give the same "
                 "accuracy. So information lives in conv1 features, and 16 of 32 final channels were empty in iter023.",
        problem="The head sees only fine-scale (~5 px) structure; coarse regional statistics (fur colour/texture over a "
                "region, silhouette) are missing, and half the channels are wasted.",
        hypothesis="Filling the 16 empty channels with 3x3-box (coarser) averages of the same conv1 features gives "
                   "512 informative features across two scales and lifts val accuracy by >=1 point over the accepted model.",
        selected="conv2.weight (4608) + conv3.weight (9216) replaced + head refit (513)",
        expected="Higher accuracy if scale diversity helps; head variance rises with more distinct features.",
        risks="Colour wiring from iter004 is dropped (colour enters only via channels 12-15 and their coarse copies).",
        edit=edit_multiscale),
    IterationSpec(
        id="iter025", title="[round 6] Spatially shared head (32 effective weights)",
        observed="24 extractor/head edits plateau at val ~60%, CV ~63%. The 512-feature head is fitted from only 1,000 "
                 "images, and centre-only edge features (96) already match it: variance, not features, may be the limit.",
        problem="512 free head weights from 1,000 samples: high estimation variance.",
        hypothesis="Sharing each channel's weight across all 16 cells (LDA over 32 cell-averaged features) cuts effective "
                   "parameters 16x, lowers variance and raises val accuracy by >=1 point, at the cost of spatial layout.",
        selected="head.weight (512 tied to 32 values) + head.bias [closed-form head, rule lda_pooled]",
        expected="Higher CV and val if variance dominates; lower if spatial layout carries the signal.",
        risks="Loses where-in-the-image information (centre vs border).",
        edit=lambda ctx: ctx.refit_head(rule="lda_pooled", reason="channel weights shared across the 16 cells")),
    IterationSpec(
        id="iter026", title="[round 6] Centre/border shared head (64 effective weights)",
        observed="Per-cell effect sizes: centre cells ~0.43, borders 0.2-0.37 (train).",
        problem="iter025 discards all spatial layout; centre vs border is the one spatial split the data supports.",
        hypothesis="LDA over 64 features (centre-cell mean, border-cell mean per channel) keeps the object-vs-background "
                   "split with 8x fewer weights than the full head and raises val accuracy by >=1 point.",
        selected="head.weight (512 tied to 64 values) + head.bias [closed-form head, rule lda_pooled_cb]",
        expected="Best of both if variance and coarse layout matter.",
        risks="Small gain, within noise.",
        edit=lambda ctx: ctx.refit_head(rule="lda_pooled_cb", reason="channel weights shared within centre / border cell groups")),
    IterationSpec(
        id="iter027", title="[round 6] Luminance-distribution (soft histogram) ramps replace the DoG blobs",
        observed="iter023/iter025/iter026: pooled low-level features plateau at CV ~63% whatever the head or hierarchy; "
                 "the DoG channels 8-11 are the least informative conv1 group and encode only local contrast.",
        problem="No feature describes the global brightness distribution (dark fur / bright background), the kind of "
                "colour-histogram cue known to separate pet photos.",
        hypothesis="Four thresholded single-pixel luminance ramps (bright 0.4, very bright 1.1, dark 0.4, very dark 1.1) "
                   "pooled over the grid give soft histogram masses and raise val accuracy by >=1 point.",
        selected="conv1.weight[8:12] (300 values), conv1.bias[8:12] (4) + head refit (513)",
        expected="Gain only if luminance distribution carries class information beyond edges and colour.",
        risks="Label correlation with brightness is only +0.09 on train; likely noise.",
        edit=edit_luminance_ramps),
]
