"""WSP-003 iterations (large split: 8,000 train / 3,000 val / sealed 6,000 test).

Hand-written, hypothesis-driven edits; evaluated on train/val/CV only. Edits marked [closed-form head]
reassign head weights by a one-shot formula over TRAIN statistics (data fitting, reported separately).
Append-only. Replications of WSP-001/002 ideas state the reason (resolve effects at 8x the data).
"""

from __future__ import annotations

import torch

import iterations as v1
import iterations_wsp002 as v2
from wsp_engine import EditContext, IterationSpec

BASELINE_OBS = ("WSP-003 baseline: train 0.602, val 0.601 (n=3000), 5-fold CV 0.601: all equal, so the model is "
                "bias-limited (features not rich enough), not variance-limited.")


def edit_pyramid(ctx: EditContext) -> None:
    """conv2: fine passthrough (0-15) + 3x3-box mid scale (16-31); conv3: fine/mid pass with blur (0-15 from fine,
    16-31 from mid but with a wide 3x3 box -> coarse ~12 px)."""
    w2 = torch.zeros(32, 16, 3, 3)
    w3 = torch.zeros(32, 32, 3, 3)
    blur = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16
    for j in range(16):
        w2[j, j, 1, 1] = 1.0
        w2[16 + j, j] = 1.0 / 9
        w3[j, j] = blur
        w3[16 + j, 16 + j] = 1.0 / 9
    ctx.programmer.replace_parameter("conv2.weight", w2, "conv2: ch j fine; ch 16+j 3x3 box of conv1 ch j")
    ctx.programmer.replace_parameter("conv3.weight", w3, "conv3: ch j fine (blur); ch 16+j coarse (3x3 box of the mid scale)")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


TEXTURE_GROUPS = {"horizontal": [0, 1, 7], "vertical": [3, 4, 5], "diagonal": [2, 6], "blob": [8, 9, 10, 11]}
COLOUR_CHANNELS = [12, 13, 14, 15]
AND_THRESHOLD = 2.0       # two scale-normalised inputs sum to 2 at their train means


def _pool1_channel_means(ctx: EditContext) -> torch.Tensor:
    """Mean post-ReLU pooled conv1 activation per channel over TRAIN (normalisation scales)."""
    from evaluator import split_data
    images, _, _ = split_data("train")
    total, count = torch.zeros(16), 0
    for i in range(0, len(images), 500):
        batch = ctx.model.pool(torch.relu(ctx.model.conv1(images[i:i + 500])))
        total += batch.sum(dim=(0, 2, 3))
        count += batch.shape[0] * batch.shape[2] * batch.shape[3]
    return total / count


def edit_conjunction(ctx: EditContext) -> None:
    """conv2 ch 0-15: fine passthrough. ch 16-31: ReLU(colour/s_c + texture/s_t - 2) conjunction detectors."""
    scale = _pool1_channel_means(ctx).clamp_min(1e-4)
    w2 = torch.zeros(32, 16, 3, 3)
    b2 = torch.zeros(32)
    for j in range(16):
        w2[j, j, 1, 1] = 1.0
    idx = 16
    for colour in COLOUR_CHANNELS:
        for group in TEXTURE_GROUPS.values():
            w2[idx, colour, 1, 1] = 1.0 / scale[colour]
            for ch in group:
                w2[idx, ch, 1, 1] = 1.0 / (len(group) * scale[ch])
            b2[idx] = -AND_THRESHOLD
            idx += 1
    w3 = torch.zeros(32, 32, 3, 3)
    blur = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16
    for c in range(32):
        w3[c, c] = blur
    ctx.notes.append(f"train-mean scales per conv1 channel: {[round(float(v), 4) for v in scale]}")
    ctx.programmer.replace_parameter("conv2.weight", w2, "conv2: ch<16 passthrough; ch 16-31 colour x texture conjunctions with train-mean scales")
    ctx.programmer.replace_parameter("conv2.bias", b2, f"conv2.bias[16:32] = -{AND_THRESHOLD}: ReLU(colour/s_c + texture/s_t - 2) = soft AND")
    ctx.programmer.replace_parameter("conv3.weight", w3, "conv3: identity routing with blur for all 32 channels")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_conjunction_fine(ctx: EditContext) -> None:
    """conv2 ch 0-31: ReLU(colour_c/s_c + gabor_k/s_k - 2) for every colour (4) x orientation (8) pair."""
    scale = _pool1_channel_means(ctx).clamp_min(1e-4)
    w2 = torch.zeros(32, 16, 3, 3)
    b2 = torch.zeros(32)
    idx = 0
    for colour in COLOUR_CHANNELS:
        for k in range(8):
            w2[idx, colour, 1, 1] = 1.0 / scale[colour]
            w2[idx, k, 1, 1] = 1.0 / scale[k]
            b2[idx] = -AND_THRESHOLD
            idx += 1
    w3 = torch.zeros(32, 32, 3, 3)
    blur = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16
    for c in range(32):
        w3[c, c] = blur
    ctx.programmer.replace_parameter("conv2.weight", w2, "conv2 ch (4*c+k): colour c x single orientation k, scale-normalised")
    ctx.programmer.replace_parameter("conv2.bias", b2, f"conv2.bias = -{AND_THRESHOLD}: soft AND")
    ctx.programmer.replace_parameter("conv3.weight", w3, "conv3: identity routing with blur for all 32 channels")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


HUE_BINS = [(13, 15), (12, 15), (13, 14), (12, 14)]   # G-R&Y-B (yellow-green), R-G&Y-B (orange), G-R&B-Y, R-G&B-Y
WEAK_ORIENTATIONS = (3, 4, 5)                          # lowest train |d| (0.16-0.18) among the 8 Gabors


def edit_hue_and_blob_conjunctions(ctx: EditContext) -> None:
    """Replace the 12 colour x {orientation 3,4,5} conjunction slots by 4 hue bins + 8 colour x DoG conjunctions."""
    scale = _pool1_channel_means(ctx).clamp_min(1e-4)
    slots = [8 * ci + k for ci in range(4) for k in WEAK_ORIENTATIONS]
    new_pairs = HUE_BINS + [(c, d) for c in (13, 15) for d in (8, 9, 10, 11)]   # 4 + 8 = 12
    for slot, (a, b) in zip(slots, new_pairs):
        block = torch.zeros(16, 3, 3)
        block[a, 1, 1] = 1.0 / scale[a]
        block[b, 1, 1] = 1.0 / scale[b]
        ctx.programmer.modify_parameter_group(
            "conv2.weight", (slot, slice(None), slice(None), slice(None)), block,
            f"conv2 ch {slot}: soft-AND of conv1 ch {a} and {b} (scale-normalised)")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_neighbourhood_conjunction(ctx: EditContext) -> None:
    """Colour at the centre tap AND the orientation channel averaged over the 3x3 neighbourhood."""
    scale = _pool1_channel_means(ctx).clamp_min(1e-4)
    for ci, colour in enumerate(COLOUR_CHANNELS):
        for k in range(8):
            slot = 8 * ci + k
            block = torch.zeros(16, 3, 3)
            block[colour, 1, 1] = 1.0 / scale[colour]
            block[k] = (1.0 / scale[k]) / 9.0
            ctx.programmer.modify_parameter_group(
                "conv2.weight", (slot, slice(None), slice(None), slice(None)), block,
                f"conv2 ch {slot}: colour {colour} at centre AND orientation {k} averaged over 3x3 (misregistration tolerance)")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_higher_threshold(ctx: EditContext) -> None:
    ctx.programmer.modify_parameter_group(
        "conv2.bias", (slice(None),), torch.full((32,), -3.0),
        "conjunction threshold -2.0 -> -3.0: fire only when summed scale-normalised evidence >= 3 (1.5x the mean each)")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def _reset_conv2_bias(ctx: EditContext) -> None:
    ctx.programmer.replace_parameter(
        "conv2.bias", torch.zeros(32),
        "reset conv2.bias to 0 (the accepted conjunction model carries -2.0, which would clip passthrough features)")


def edit_multiscale_reset(ctx: EditContext) -> None:
    _reset_conv2_bias(ctx)
    v2.edit_multiscale(ctx)


def edit_pyramid_reset(ctx: EditContext) -> None:
    _reset_conv2_bias(ctx)
    edit_pyramid(ctx)


def edit_two_frequency_gabors(ctx: EditContext) -> None:
    """Odd-index orientations (1,3,5,7) become coarse-frequency Gabors (wavelength 6, sigma 2); even stay fine (wavelength 4)."""
    import math
    from init_weights import LUMA, NUM_ORIENTATIONS, _gabor
    for k in range(1, NUM_ORIENTATIONS, 2):
        kernel = LUMA.view(3, 1, 1) * _gabor(math.pi * k / NUM_ORIENTATIONS, sigma=2.0, wavelength=6.0)
        ctx.programmer.modify_parameter_group(
            "conv1.weight", (k, slice(None), slice(None), slice(None)), kernel,
            f"conv1 filter {k}: Gabor wavelength 4 -> 6 px (sigma 2.0): second spatial frequency band")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_local_peaks(ctx: EditContext) -> None:
    """conv2 ch 0-15: fine passthrough; ch 16-31: ReLU(feature - 3x3 neighbourhood mean) (local peakiness)."""
    w2 = torch.zeros(32, 16, 3, 3)
    w3 = torch.zeros(32, 32, 3, 3)
    blur = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16
    for j in range(16):
        w2[j, j, 1, 1] = 1.0
        w2[16 + j, j] = -1.0 / 9
        w2[16 + j, j, 1, 1] += 1.0
    for c in range(32):
        w3[c, c] = blur
    _reset_conv2_bias(ctx)
    ctx.programmer.replace_parameter("conv2.weight", w2, "conv2: ch j fine; ch 16+j = ReLU(f_j - 3x3 box mean of f_j) local peakiness")
    ctx.programmer.replace_parameter("conv3.weight", w3, "conv3: identity routing with blur for all 32 channels")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_diagonal_hues(ctx: EditContext) -> None:
    """conv1 DoG filters 8-11 -> four diagonal hue-sector colour filters (45/135/225/315 deg in the RG/BY plane)."""
    from init_weights import _blur
    rg, by = torch.tensor([1.0, -1.0, 0.0]), torch.tensor([-0.5, -0.5, 1.0])
    vectors = [rg - by, -rg - by, -rg + by, rg + by]       # orange, yellow-green, cyan/blue-green, purple
    names = ["orange", "yellow-green", "blue-green", "purple"]
    for i, (vec, name) in enumerate(zip(vectors, names)):
        ctx.programmer.modify_parameter_group(
            "conv1.weight", (8 + i, slice(None), slice(None), slice(None)), vec.view(3, 1, 1) * _blur(),
            f"conv1 filter {8 + i}: diagonal hue sector '{name}' (Gaussian-blurred colour ramp), replaces a DoG blob")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_odd_phase_gabors(ctx: EditContext) -> None:
    """conv1 DoG filters 8-11 -> odd-phase (sine) Gabors at orientations 0, 2, 4, 6 (quadrature partners)."""
    import math
    from init_weights import LUMA, NUM_ORIENTATIONS, _grid, _l1
    x, y = _grid(5)
    for i, k in enumerate((0, 2, 4, 6)):
        theta = math.pi * k / NUM_ORIENTATIONS
        xr = x * math.cos(theta) + y * math.sin(theta)
        kern = torch.exp(-(x**2 + y**2) / (2 * 1.5**2)) * torch.sin(2 * math.pi * xr / 4.0)
        ctx.programmer.modify_parameter_group(
            "conv1.weight", (8 + i, slice(None), slice(None), slice(None)), LUMA.view(3, 1, 1) * _l1(kern),
            f"conv1 filter {8 + i}: odd-phase Gabor at orientation {k} (quadrature partner of filter {k}), replaces a DoG blob")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


def edit_peaks_and_odd_gabors(ctx: EditContext) -> None:
    edit_odd_phase_gabors(ctx)      # conv1[8:12] (ends with a head refit; superseded by the final one)
    edit_local_peaks(ctx)           # conv2/conv3 + bias reset, then final head refit


# ---- Round 2: continuation after the sealed test was spent (judged on val/CV only; unverified on test) ----

def edit_dark_bar_polarity(ctx: EditContext) -> None:
    """conv1 DoG 8-11 -> NEGATED even Gabors (dark-bar polarity) at orientations 0,2,4,6."""
    import math
    from init_weights import LUMA, NUM_ORIENTATIONS, _gabor
    for i, k in enumerate((0, 2, 4, 6)):
        kernel = -LUMA.view(3, 1, 1) * _gabor(math.pi * k / NUM_ORIENTATIONS)
        ctx.programmer.modify_parameter_group(
            "conv1.weight", (8 + i, slice(None), slice(None), slice(None)), kernel,
            f"conv1 filter {8 + i}: negated Gabor at orientation {k} (dark-bar polarity lost by the ReLU of filter {k})")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


TOP_CHANNELS = [0, 1, 2, 6, 7, 12, 13, 14, 15, 10]   # strongest Gabors, 4 colour channels, DoG on 0.8 (ablation-informed)


def edit_three_types_top_channels(ctx: EditContext) -> None:
    """30 conv2 channels: for each of 10 top conv1 maps -> fine, 3x3-box, local peak. Channels 30-31 unused."""
    w2 = torch.zeros(32, 16, 3, 3)
    w3 = torch.zeros(32, 32, 3, 3)
    blur = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16
    for i, ch in enumerate(TOP_CHANNELS):
        w2[i, ch, 1, 1] = 1.0                                   # fine
        w2[10 + i, ch] = 1.0 / 9                                # coarse box
        w2[20 + i, ch] = -1.0 / 9; w2[20 + i, ch, 1, 1] += 1.0  # ReLU(f - box): local peak
    for c in range(30):
        w3[c, c] = blur
    _reset_conv2_bias(ctx)
    ctx.programmer.replace_parameter("conv2.weight", w2, "conv2: top-10 conv1 maps x {fine, box, peak} (30 channels)")
    ctx.programmer.replace_parameter("conv3.weight", w3, "conv3: identity routing with blur for channels 0-29")
    ctx.refit_head(reason="extractor changed; re-derive head in closed form")


ITERATIONS = [
    IterationSpec(
        id="iter001", title="Wire the colour pathway into conv2 (replication at 8x data)",
        observed=BASELINE_OBS + " conv2 receives 0.0 weight from the 4 colour channels.",
        problem="Colour is discarded.",
        hypothesis="Wiring colour into conv2 blob filters 24-31 raises val accuracy by >=1 point. (WSP-002: +3.5 pts val "
                   "on 600 images but 0 on the 2,000-image test; re-tested where a 1-pt effect is resolvable, n=3000.)",
        selected="conv2.weight[24:32,12:16] (72) + head refit (513)",
        expected="Gain if colour is genuinely informative; none if the earlier gain was noise.",
        risks="Spurious colour cue.", edit=v1.edit_004),
    IterationSpec(
        id="iter002", title="Multi-scale passthrough: 16 fine + 16 coarse conv1 maps (replication at 8x data)",
        observed=BASELINE_OBS + " WSP-002 iter024 reached CV 0.640 vs 0.632 (small data); the hierarchy ablation showed "
                 "conv2/conv3 composition adds nothing over conv1 features.",
        problem="Features are too poor to separate the classes; capacity of the final descriptor is half empty/duplicated.",
        hypothesis="Filling all 32 final channels with informative fine+coarse conv1 features raises val accuracy by >=1 "
                   "point; with 8,000 train images the 512-weight head can now use them without variance penalty.",
        selected="conv2.weight (4608) + conv3.weight (9216) replaced + head refit (513)",
        expected="Gain if scale diversity helps.", risks="Colour wiring dropped; no composition.",
        edit=v2.edit_multiscale),
    IterationSpec(
        id="iter003", title="Three-scale pyramid: fine blur + mid + coarse box",
        observed="Features are scale-limited: the head sees ~6 px (fine) and ~12 px (mid) structure only.",
        problem="Object-level structure (head/body shape, ~20-40 px) is not represented in any final channel.",
        hypothesis="Adding a coarse (3x3 box over the mid scale, ~12-24 px) channel set raises val accuracy by >=1 point.",
        selected="conv2.weight (4608) + conv3.weight (9216) replaced + head refit (513)",
        expected="Gain if larger-scale structure carries class information.",
        risks="Coarse averages blur into global statistics.", edit=edit_pyramid),
    IterationSpec(
        id="iter004", title="Colour x texture conjunction detectors (thresholded soft-AND)",
        observed="On 8,000 train images: dogs have more of every channel (signed d>0), strongest for G-R (grass-green) "
                 "and Y-B (tan/yellow) colour channels and near-horizontal edges; coarse copies of channels carry the same "
                 "signal as fine ones (|d| 0.285 vs 0.284), so multi-scale added little. No ReLU ever gates: no feature "
                 "can express a conjunction.",
        problem="The head sees only marginal pooled colour and texture; co-occurrence (e.g. green+textured = grass, "
                "tan+textured = fur) is unrepresented, and all channels are near-duplicates.",
        hypothesis="16 conjunction channels ReLU(colour/s_c + texture/s_t - 2) over {4 colour} x {horizontal, vertical, "
                   "diagonal, blob} texture groups add genuinely new information and raise val accuracy by >=1 point. "
                   "(Likely exploits scene context such as grass; that would be a context cue, not animal recognition.)",
        selected="conv2.weight+bias (4608+32), conv3.weight (9216) + head refit (513)",
        expected="Gain if colour-texture co-occurrence is class-informative.",
        risks="Scale normalisation uses train means (data-derived); scene-context shortcut.",
        edit=edit_conjunction),
    IterationSpec(
        id="iter005", title="Fine conjunctions: every colour x every single orientation (32 channels)",
        observed="iter004 (grouped conjunctions + 16 passthrough channels): val +0.64, CV +1.00 vs the accepted model: "
                 "conjunctions add real information but fall just under the 1-pt rule.",
        problem="Texture groups average different orientations; passthrough channels are largely redundant.",
        hypothesis="Using all 32 channels for colour x single-orientation soft-ANDs gives a richer, non-redundant "
                   "descriptor and raises val accuracy by >=1 point over the accepted model.",
        selected="conv2.weight+bias (4608+32), conv3.weight (9216) + head refit (513)",
        expected="Larger gain than iter004 if orientation-specific colour co-occurrence is informative.",
        risks="Loses marginal (passthrough) features and the DoG/blob channels; scene-context shortcut.",
        edit=edit_conjunction_fine),
    IterationSpec(
        id="iter006", title="Hue-bin and colour x blob conjunctions replace the weakest-orientation slots",
        observed="iter005 accepted (val 0.6357): colour x orientation conjunctions work. Orientations 3-5 had the lowest "
                 "train |d| (0.16-0.18) so their 12 colour conjunctions are the weakest slots. Colour x colour (hue "
                 "combinations such as yellow-green, orange) and colour x blob are untried conjunction types.",
        problem="12 of 32 final channels carry low-signal conjunctions while hue and blob co-occurrence is missing.",
        hypothesis="Replacing those slots with 4 hue-bin ANDs (colour x colour) and 8 colour x DoG ANDs adds new "
                   "information and raises val accuracy by >=1 point.",
        selected="conv2.weight rows for 12 slots (12x16x3x3 = 1728) + head refit (513); conv2.bias unchanged (-2.0)",
        expected="Gain if hue distribution / colour-blob co-occurrence is informative.",
        risks="Hue bins capture scene colour (grass, indoor beige): context, not animal.",
        edit=edit_hue_and_blob_conjunctions),
    IterationSpec(
        id="iter007", title="Neighbourhood-tolerant conjunctions (colour at centre AND orientation over 3x3)",
        observed="iter005 accepted. Conjunction channels fire at 25-37% of positions (sensible gating). Colour (5x5 blurred) "
                 "and Gabor responses are not pixel-aligned, and boundaries (fur vs background) put edge and colour in "
                 "adjacent, not identical, cells.",
        problem="Centre-tap-only AND demands exact co-registration of colour and edge.",
        hypothesis="Averaging the orientation input over the 3x3 neighbourhood (misregistration tolerance) makes the "
                   "conjunction fire for colour regions bounded by oriented edges and raises val accuracy by >=1 point.",
        selected="conv2.weight rows 0-31 (32x16x3x3 = 4608) + head refit (513)",
        expected="Gain if boundary co-occurrence matters.", risks="Blurs exact co-location information.",
        edit=edit_neighbourhood_conjunction),
    IterationSpec(
        id="iter008", title="Higher AND threshold (-2.0 -> -3.0): rarer, stronger co-occurrences",
        observed="Pre-activations are heavy-tailed (ch0 quantiles: 50%=-0.70, 70%=0.33, 90%=2.48, 99%=7.9); units are active "
                 "25-37% of the time.",
        problem="At threshold 2 many weak coincidences pass; pooled averages are dominated by diffuse activity.",
        hypothesis="Requiring summed normalised evidence >= 3 isolates strong co-occurrences that are more class-specific "
                   "and raises val accuracy by >=1 point. (One directional test, not a sweep.)",
        selected="conv2.bias (32) + head refit (513)",
        expected="Sparser, more selective features.", risks="Too sparse: low-variance features.",
        edit=edit_higher_threshold),
    IterationSpec(
        id="iter009", title="Full-covariance LDA head (closed form) [data-fitted head, not a hand edit]",
        observed="train 0.629 / val 0.636 / CV 0.628 agree: bias-limited. The 512 pooled features are strongly "
                 "correlated (fine/coarse and conjunction families), yet the head divides by per-feature variance only.",
        problem="Diagonal LDA ignores feature correlations, so correlated features are double counted and "
                "complementary ones under-used.",
        hypothesis="w = (Sigma_within + 0.05*mean_var*I)^-1 (mu_dog - mu_cat), solved once from the 8,000 train images, "
                   "raises val accuracy by >=1 point. Ridge fixed a priori, not tuned.",
        selected="head.weight (512), head.bias [closed-form head, rule lda_full]",
        expected="A clear gain if correlation structure is the bottleneck.",
        risks="More of the accuracy comes from data fitting; 513 parameters estimated from 8,000 samples.",
        edit=lambda ctx: ctx.refit_head(rule="lda_full", reason="full-covariance closed-form discriminant")),
    IterationSpec(
        id="iter010", title="[full-LDA head] 16 marginal passthrough + 16 grouped conjunction channels",
        observed="iter009: full-covariance LDA lifts val 0.636 -> 0.744 (train 0.752, CV 0.742): correlated features were "
                 "the bottleneck, so redundancy no longer hurts and feature diversity may now pay off.",
        problem="The accepted extractor uses all 32 channels for colour x orientation conjunctions and drops marginal "
                "colour/texture/blob channels that were rejected as redundant under the diagonal head.",
        hypothesis="Mixing 16 marginal passthrough channels with 16 grouped conjunctions (iter004 design) adds information "
                   "the full-covariance head can exploit and raises val accuracy by >=1 point over 0.7443.",
        selected="conv2.weight+bias (4608+32), conv3.weight (9216) + head refit (513, lda_full)",
        expected="Gain if marginal features are complementary.", risks="Loses 16 finer conjunctions.",
        edit=edit_conjunction),
    IterationSpec(
        id="iter011", title="[full-LDA head] Multi-scale passthrough (16 fine + 16 coarse conv1 maps)",
        observed="Under the diagonal head multi-scale gave CV +0.9 pts but missed the rule; coarse copies were near "
                 "duplicates (|d| 0.285 vs 0.284). A full-covariance head treats duplicates correctly.",
        problem="Unknown whether scale diversity is useful once correlations are modelled.",
        hypothesis="Fine + coarse passthrough exceeds the conjunction extractor (val 0.744) by >=1 point.",
        selected="conv2.weight (4608) + conv3.weight (9216) replaced + head refit (513, lda_full)",
        expected="Probably no gain (conjunctions add information that marginals lack).",
        risks="Drops all conjunctions.", edit=v2.edit_multiscale),
    IterationSpec(
        id="iter012", title="[full-LDA head] Three-scale pyramid with a coarse box scale",
        observed="Same as iter011; features lack object-scale (20-40 px) structure.",
        problem="No final channel covers coarse structure.",
        hypothesis="A coarse 3x3-box-of-mid scale adds larger-scale information and beats 0.744 by >=1 point.",
        selected="conv2.weight (4608) + conv3.weight (9216) replaced + head refit (513, lda_full)",
        expected="Probably no gain over conjunctions.", risks="Drops conjunctions.", edit=edit_pyramid),
    IterationSpec(
        id="iter013", title="[full-LDA head] Multi-scale passthrough with conv2.bias reset (valid re-run of iter011)",
        observed="iter011/iter012 collapsed to 50% because the edit left conv2.bias=-2.0 from the conjunction model: 384 of "
                 "512 features were exactly zero (debugged). Those two results are INVALID tests, not evidence.",
        problem="The multi-scale hypothesis was never actually tested under the full-covariance head.",
        hypothesis="Fine + coarse passthrough (bias reset to 0) beats the conjunction extractor (val 0.744) by >=1 point.",
        selected="conv2.bias (32) reset, conv2.weight (4608), conv3.weight (9216) + head refit (513, lda_full)",
        expected="Probably below the conjunction extractor.", risks="Drops all conjunctions.",
        edit=edit_multiscale_reset),
    IterationSpec(
        id="iter014", title="[full-LDA head] Three-scale pyramid with conv2.bias reset (valid re-run of iter012)",
        observed="Same bug as iter011 (stale -2.0 bias).",
        problem="Pyramid hypothesis untested under the full-covariance head.",
        hypothesis="Fine + mid + coarse features beat val 0.744 by >=1 point.",
        selected="conv2.bias (32) reset, conv2.weight (4608), conv3.weight (9216) + head refit (513, lda_full)",
        expected="Probably below the conjunction extractor.", risks="Drops all conjunctions.",
        edit=edit_pyramid_reset),
    IterationSpec(
        id="iter015", title="[full-LDA head] Two spatial-frequency Gabor bands (odd orientations -> wavelength 6)",
        observed="Accepted model (iter013): val 0.766. Fine and coarse passthrough copies are near-duplicates (|d| 0.285 vs "
                 "0.284); a full-covariance head can use complementary features but gains nothing from copies.",
        problem="All 8 oriented filters share one frequency band, so texture-frequency information is absent.",
        hypothesis="Moving 4 of the 8 orientations (1,3,5,7) to wavelength 6 gives 4 angles x 2 frequency bands, adds "
                   "new texture information and raises val accuracy by >=1 point.",
        selected="conv1.weight[1,3,5,7] (4x3x5x5 = 300 values) + head refit (513, lda_full)",
        expected="Gain if texture frequency discriminates cat vs dog fur/background.",
        risks="Halves angular resolution at each frequency; 5x5 support truncates wavelength 6.",
        edit=edit_two_frequency_gabors),
    IterationSpec(
        id="iter016", title="[full-LDA head] Local-peakiness channels replace the coarse marginal copies",
        observed="Fine + coarse passthrough (val 0.766) beats fine + conjunctions (0.718): scale contrast is exploitable "
                 "by the full-covariance head. Coarse copies, however, are linear in the fine features before pooling, so "
                 "they add only what a linear average adds.",
        problem="No channel measures how much a feature STANDS OUT from its neighbourhood (a nonlinear quantity).",
        hypothesis="ReLU(f - 3x3 mean of f) per conv1 feature adds nonlinear local-contrast information not present in "
                   "linear fine/coarse averages, and raises val accuracy by >=1 point over 0.7663.",
        selected="conv2.bias reset, conv2.weight (4608), conv3.weight (9216) + head refit (513, lda_full)",
        expected="Gain if texture/colour 'peakiness' is class-informative.",
        risks="Loses the coarse marginals that currently help.",
        edit=edit_local_peaks),
    IterationSpec(
        id="iter017", title="[full-LDA head] Diagonal hue sectors replace the DoG blob filters",
        observed="Ablation (fit on train, val check): fine marginals alone reach val 0.765; dropping orientation filters "
                 "costs 7.5 pts, colour 2.2 pts, DoG only 0.9 pts; colour alone reaches 0.677. Information is in the 16 "
                 "conv1 filters, and the DoG group is the weakest.",
        problem="Colour is the second-most valuable group but is represented by only 4 axis-aligned ramps (4 hue sectors).",
        hypothesis="Replacing the 4 DoG filters with 4 diagonal hue-sector ramps (orange, yellow-green, blue-green, "
                   "purple) doubles hue resolution (ReLU of a sum differs from the sum of ReLUs) and raises val accuracy "
                   ">=1 point over 0.7663.",
        selected="conv1.weight[8:12] (4x3x5x5 = 300 values) + head refit (513, lda_full)",
        expected="Gain if hue resolution matters (grass, fur tones).",
        risks="Loses ~0.9 pts of DoG blob information; hue may be a scene-context cue.",
        edit=edit_diagonal_hues),
    IterationSpec(
        id="iter018", title="[full-LDA head] Odd-phase Gabors replace the DoG blob filters",
        observed="Same ablation: orientation filters carry the most information (without them val 0.698; only them 0.733).",
        problem="Oriented filters are all even-symmetric (bar detectors); step-edge (odd-phase) responses are absent.",
        hypothesis="Odd-phase partners at orientations 0,2,4,6 give quadrature phase information and raise val accuracy "
                   ">=1 point over 0.7663.",
        selected="conv1.weight[8:12] (300 values) + head refit (513, lda_full)",
        expected="Gain if edge-vs-bar polarity is informative.", risks="Loses DoG blob information.",
        edit=edit_odd_phase_gabors),
    IterationSpec(
        id="iter019", title="[full-LDA head] Combine local-peakiness channels (iter016) and odd-phase Gabors (iter018)",
        observed="iter016 (peaks) gave val +0.70 / CV +0.77 and iter018 (odd-phase Gabors) val +0.70 / CV +0.58, each just "
                 "under the 1-pt rule and consistent across val and 8,000-image CV. They modify different layers.",
        problem="Each small gain alone is below the acceptance resolution.",
        hypothesis="The two effects are additive: together they raise val accuracy by >=1 point over 0.7663.",
        selected="conv1.weight[8:12] (300) + conv2.bias (32) + conv2.weight (4608) + conv3.weight (9216) + head refit (513)",
        expected="About +1.4 pts if additive; less if they capture the same information.",
        risks="Overlap; losing coarse marginals and DoG blobs.",
        edit=edit_peaks_and_odd_gabors),
    IterationSpec(
        id="iter020", title="[round 2] Dark-bar polarity: negated Gabors replace the DoG blob filters",
        observed="Every even Gabor passes through a ReLU, so only the bright-bar half of its response survives; dark bars "
                 "(dark fur against light background) are invisible. DoG and colour channels already have both polarities.",
        problem="Half of the oriented-edge information is discarded by the ReLU.",
        hypothesis="Negated Gabors at orientations 0,2,4,6 recover dark-bar responses for the strongest orientations and "
                   "raise val accuracy by >=1 point over 0.7663. (Sibling of iter018, which tried odd phase: +0.7.)",
        selected="conv1.weight[8:12] (300 values) + head refit (513, lda_full)",
        expected="Gain if dark-bar structure is informative.", risks="Loses the DoG blob group (~0.9 pt).",
        edit=edit_dark_bar_polarity),
    IterationSpec(
        id="iter021", title="[round 2] Spend all channels on the 10 best conv1 maps x {fine, coarse, peak}",
        observed="Ablation: orientation filters 0,1,2,6,7 had the strongest |d|; weak ones (3,4,5) and the DoG group add "
                 "little. Currently 32 channels = 16 maps x 2 types, including the weak maps.",
        problem="Capacity is spread over low-information maps.",
        hypothesis="Using 30 channels as 10 strongest maps x 3 feature types (fine, 3x3 box, local peak) gives the "
                   "full-covariance head more informative features and raises val accuracy by >=1 point.",
        selected="conv2.bias reset, conv2.weight (4608), conv3.weight (9216) + head refit (513, lda_full)",
        expected="Modest gain; risks losing information in the dropped 6 maps.",
        risks="Selection of the 10 maps is ablation-informed (train fit, val check).",
        edit=edit_three_types_top_channels),
]
