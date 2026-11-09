# Decision log

## Pre-registered acceptance rule

wsp-002, pre-registered before any edit. Accept iff (a) validation accuracy rises by >= 0.010 (1 percentage point) over the accepted checkpoint AND (b) 5-fold CV accuracy on TRAIN does not fall by more than 0.010 AND (c) validation F1 does not fall by more than 0.010. Train accuracy is recorded but never used to accept. The sealed test split is never read during iterations.

## iter000: baseline

Hand-programmed initial network. {'train_acc': 0.622, 'val_acc': 0.5717, 'val_f1': 0.5403, 'val_margin': 0.398, 'cv_acc': 0.618, 'cv_margin': 0.5081}

## iter001: Remove global-energy direction from the head (replication at scale) -> REJECTED

- Observed: On train (n=1000) the baseline logit correlates r=+0.91 with image edge density but the label only +0.25; corr(logit,label)=0.26. Head weights are almost all positive.
- Hypothesis: Subtracting the mean head weight removes the energy-following direction, so the logit depends on which features differ between classes. (Rejected on 4 val images in WSP-001; re-tested because 600 val images can now resolve a 1-point effect.)
- Parameters: head.weight (512), head.bias (1) [closed-form head] (513 changed)
- Before: {'train_acc': 0.622, 'val_acc': 0.5717, 'val_f1': 0.5403, 'val_margin': 0.398, 'cv_acc': 0.618, 'cv_margin': 0.5081}
- After: {'train_acc': 0.593, 'val_acc': 0.5633, 'val_f1': 0.5619, 'val_margin': 0.3687, 'cv_acc': 0.586, 'cv_margin': 0.3878}
- Reason: diagnostic regression (cv -0.0320, val_f1 +0.0216)

## iter002: Centre conv2 pre-activations with negative biases (replication) -> REJECTED

- Observed: relu2 is 95% active and relu3 100%: ReLUs do not gate, so the network is nearly linear.
- Hypothesis: conv2.bias_c = -mean(pre-activation_c over train) makes units fire only for above-average local evidence. (Accepted in WSP-001 on 16 images; re-tested at scale.)
- Parameters: conv2.bias (32) + head refit (513) [bias is a data-derived constant; head closed-form] (545 changed)
- Before: {'train_acc': 0.622, 'val_acc': 0.5717, 'val_f1': 0.5403, 'val_margin': 0.398, 'cv_acc': 0.618, 'cv_margin': 0.5081}
- After: {'train_acc': 0.613, 'val_acc': 0.5783, 'val_f1': 0.5358, 'val_margin': 0.3726, 'cv_acc': 0.613, 'cv_margin': 0.4708}
- Reason: no meaningful val improvement (val +0.0066, cv -0.0050, val_f1 -0.0045)

## iter003: Centre conv3 pre-activations with negative biases (replication) -> REJECTED

- Observed: relu3 is 100% active; every final feature contributes.
- Hypothesis: Centring conv3 pre-activations gates the final features. (Rejected in WSP-001 on 16 images.)
- Parameters: conv3.bias (32) + head refit (513) (545 changed)
- Before: {'train_acc': 0.622, 'val_acc': 0.5717, 'val_f1': 0.5403, 'val_margin': 0.398, 'cv_acc': 0.618, 'cv_margin': 0.5081}
- After: {'train_acc': 0.594, 'val_acc': 0.575, 'val_f1': 0.5216, 'val_margin': 0.3504, 'cv_acc': 0.594, 'cv_margin': 0.4193}
- Reason: diagnostic regression (cv -0.0240, val_f1 -0.0187)

## iter004: Wire the dead colour pathway into conv2 (replication) -> ACCEPTED

- Observed: conv2 receives exactly 0.0 weight from the 4 colour channels. On this larger train set warmth barely relates to label (r=+0.08), unlike the 20-image WSP-001 sample.
- Hypothesis: Connecting colour channels to conv2 filters 24-31 will NOT help (WSP-001's val gain was one image). This is a falsification test of the WSP-001 colour finding.
- Parameters: conv2.weight[24:32,12:16] (72) + head refit (513) (585 changed)
- Before: {'train_acc': 0.622, 'val_acc': 0.5717, 'val_f1': 0.5403, 'val_margin': 0.398, 'cv_acc': 0.618, 'cv_margin': 0.5081}
- After: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- Reason: val improvement (val +0.0350, cv +0.0140, val_f1 +0.0486)

## iter005: Suppress the bottom pooled row in the head -> REJECTED

- Observed: Per-cell mean |effect size| on train: 0.43/0.42 for centre cells vs 0.10-0.20 for the bottom row (ground/floor/furniture), 0.22-0.37 elsewhere.
- Hypothesis: Scaling bottom-row head weights by 0.25 reduces background noise.
- Parameters: head.weight bottom-row entries (128 of 512), head.bias [closed-form head with mask] (513 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.636, 'val_acc': 0.6, 'val_f1': 0.5862, 'val_margin': 0.4039, 'cv_acc': 0.633, 'cv_margin': 0.6116}
- Reason: no meaningful val improvement (val -0.0067, cv +0.0010, val_f1 -0.0027)

## iter006: Identity routing for conv3 filters 24-31 (de-duplicate colour/blob features) -> REJECTED

- Observed: After iter004, conv3 filters 24-31 each sum ALL eight conv2 blob/colour channels with the same kernel, so they are identical: colour evidence reaches the head as one feature repeated 8x (16 distinct values of 512).
- Hypothesis: Routing conv3 filter 24+j to conv2 channel 24+j only gives eight distinct colour/blob descriptors, so the closed-form head can use them separately and val accuracy rises.
- Parameters: conv3.weight[24:32] (8x32x3x3 = 2304 values) + head refit (513) (1089 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.631, 'val_acc': 0.605, 'val_f1': 0.5864, 'val_margin': 0.4315, 'cv_acc': 0.632, 'cv_margin': 0.5995}
- Reason: no meaningful val improvement (val -0.0017, cv +0.0000, val_f1 -0.0025)

## iter007: Amplify colour pathway x3 (replication of WSP-001 iter007) -> REJECTED

- Observed: iter004 (colour) was the only accepted edit; colour weights are 1/9 per tap vs edge weights up to 0.5.
- Hypothesis: Scaling the 72 wired colour weights by 3 strengthens colour influence and raises val accuracy.
- Parameters: conv2.weight[24:32,12:16] (72) + head refit (513) (585 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.637, 'val_acc': 0.6, 'val_f1': 0.5789, 'val_margin': 0.3829, 'cv_acc': 0.633, 'cv_margin': 0.5889}
- Reason: no meaningful val improvement (val -0.0067, cv +0.0010, val_f1 -0.0100)

## iter008: Wire colour into conv2 corner filters 8-15 -> REJECTED

- Observed: Colour reaches conv3 only through the blob+edge path (filters 24-31).
- Hypothesis: Adding colour-opponent input to the corner filters lets the network represent coloured corner/junction regions (e.g. nose, eyes against fur).
- Parameters: conv2.weight[8:16,12:16] (72) + head refit (513) (585 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.628, 'val_acc': 0.5917, 'val_f1': 0.5601, 'val_margin': 0.3527, 'cv_acc': 0.624, 'cv_margin': 0.5682}
- Reason: diagnostic regression (cv -0.0080, val_f1 -0.0288)

## iter009: Coarser oriented edge filters in conv1 (wavelength 4 -> 6 px) -> REJECTED

- Observed: The accepted model's logit still follows fine-scale edge density (baseline r=+0.91 with mean |dI/dx|, only +0.25 with the label). Fine 4-px Gabors mostly measure texture/JPEG detail.
- Hypothesis: Replacing the 8 oriented filters with wavelength-6 px, sigma-2.0 Gabors lowers dependence on fine texture and raises val accuracy by >=1 point.
- Parameters: conv1.weight[0:8] (8x3x5x5 = 600 values) + head refit (513) (1113 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.635, 'val_acc': 0.5983, 'val_f1': 0.5852, 'val_margin': 0.3733, 'cv_acc': 0.631, 'cv_margin': 0.5916}
- Reason: no meaningful val improvement (val -0.0084, cv -0.0010, val_f1 -0.0037)

## iter010: Centroid head rule (no per-feature variance scaling) -> REJECTED

- Observed: Head has 512 features fitted from 1000 train images; centre-only edge features (96) match the full head on val (59.5 vs 60.7%), so many features add noise.
- Hypothesis: The plain class-mean difference (centroid rule) is more robust and raises val accuracy by >=1 point. (Rejected in WSP-001 on 16 images.)
- Parameters: head.weight (512), head.bias (1) [closed-form head] (513 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.61, 'val_acc': 0.5867, 'val_f1': 0.554, 'val_margin': 0.2861, 'cv_acc': 0.612, 'cv_margin': 0.5203}
- Reason: diagnostic regression (cv -0.0200, val_f1 -0.0349)

## iter011: Scale border-cell head weights by 0.25 -> REJECTED

- Observed: Edge features restricted to the 4 centre cells reach val 59.5%, same as all 16 cells within noise (+-2%), with 4x fewer features.
- Hypothesis: Down-weighting the 12 border cells to 0.25 reduces variance and background reliance.
- Parameters: head.weight border-cell entries (384 of 512), head.bias [closed-form head with mask] (513 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.634, 'val_acc': 0.5883, 'val_f1': 0.5704, 'val_margin': 0.3919, 'cv_acc': 0.636, 'cv_margin': 0.6162}
- Reason: diagnostic regression (cv +0.0040, val_f1 -0.0185)

## iter012: Prune the 8 weakest conv3 channels -> REJECTED

- Observed: Head variance dominates: centre-only edge features (96) match all 512 on val. Per-channel mean |effect size| on train varies ~2x between channels.
- Hypothesis: Zeroing the 8 conv3 filters (among 0-23) with the lowest mean |d| on TRAIN lowers head variance and raises val accuracy by >=1 point.
- Parameters: conv3.weight rows for the 8 weakest channels (8x32x3x3 = 2304 values) + head refit (513) (729 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.636, 'val_acc': 0.5967, 'val_f1': 0.5769, 'val_margin': 0.3683, 'cv_acc': 0.634, 'cv_margin': 0.5927}
- Reason: diagnostic regression (cv +0.0020, val_f1 -0.0120)

## iter013: Average colour over a larger area (Gaussian sigma 1.5 -> 5x5 box) -> REJECTED

- Observed: Colour wiring (iter004) was the only accepted edit (+3.5 pts val). Colour is a region property; conv1 colour filters are small Gaussians, exposed to JPEG chroma noise.
- Hypothesis: A flat 5x5 box average of the colour-opponent signal yields a more reliable colour descriptor and a further >=1 point val gain.
- Parameters: conv1.weight[12:16] (4x3x5x5 = 300 values) + head refit (513) (763 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3864, 'cv_acc': 0.632, 'cv_margin': 0.5931}
- Reason: no meaningful val improvement (val +0.0000, cv +0.0000, val_f1 +0.0000)

## iter014: [round 2] Wire colour into conv2 parallel-pair filters 16-23 -> REJECTED

- Observed: Colour entered via the blob path (iter004, accepted) but not via corner filters (iter008, rejected).
- Hypothesis: Giving the parallel-pair (stripe/texture) filters colour-opponent input adds colour-textured structure and raises val accuracy by >=1 point. Completes the colour-routing family.
- Parameters: conv2.weight[16:24,12:16] (72) + head refit (513) (585 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.625, 'val_acc': 0.5933, 'val_f1': 0.5627, 'val_margin': 0.3516, 'cv_acc': 0.623, 'cv_margin': 0.5668}
- Reason: diagnostic regression (cv -0.0090, val_f1 -0.0262)

## iter015: [round 2] Half-strength conv2 bias gating on the accepted model -> REJECTED

- Observed: Full centring (iter002) moved val +0.7 pts but CV -0.5 pts: borderline, not rejected for harm.
- Hypothesis: Half-strength centring keeps some selectivity without regressions and gives a >=1 point val gain.
- Parameters: conv2.bias (32) + head refit (513) (545 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.638, 'val_acc': 0.6033, 'val_f1': 0.5825, 'val_margin': 0.3841, 'cv_acc': 0.632, 'cv_margin': 0.5864}
- Reason: no meaningful val improvement (val -0.0034, cv +0.0000, val_f1 -0.0064)

## iter016: [round 2] Remove oriented-edge inputs from the colour/blob filters -> REJECTED

- Observed: Blob+colour filters 24-31 carry gabor/9 edge input plus DoG plus colour; the edge part duplicates channels 0-23, and colour is the only modality with proven value.
- Hypothesis: Zeroing the gabor taps (conv2.weight[24:32,0:8]) yields cleaner colour/DoG descriptors and raises val accuracy by >=1 point.
- Parameters: conv2.weight[24:32,0:8] (8x8x3x3 = 576 values) + head refit (513) (585 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.636, 'val_acc': 0.6033, 'val_f1': 0.5868, 'val_margin': 0.3867, 'cv_acc': 0.633, 'cv_margin': 0.5936}
- Reason: no meaningful val improvement (val -0.0034, cv +0.0010, val_f1 -0.0021)

## iter017: [round 3] Bright blob-pair detectors (eyes) in conv3 filters 25-28 -> REJECTED

- Observed: All earlier edits re-weighted generic edge/colour features; none encoded facial structure. conv3 filters 25-31 are still redundant duplicates of filter 24 (wasted capacity).
- Hypothesis: Programming 4 conv3 filters as horizontal/vertical/diagonal pair detectors over the bright-blob channel (conv2 ch 24) gives the head a face-structure feature and lifts val accuracy by >=1 point.
- Parameters: conv3.weight[25:29] (4x32x3x3 = 1152 values) + head refit (513) (801 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.636, 'val_acc': 0.6017, 'val_f1': 0.5872, 'val_margin': 0.3834, 'cv_acc': 0.631, 'cv_margin': 0.6009}
- Reason: no meaningful val improvement (val -0.0050, cv -0.0010, val_f1 -0.0017)

## iter018: [round 3] Dark blob-pair detectors (nostril/dark eyes) in conv3 filters 29-31 -> REJECTED

- Observed: Dogs' eyes and noses are dark blobs on lighter fur; the off-centre DoG channel (conv2 ch 25) is unused as a pair.
- Hypothesis: Pair detectors over the dark-blob channel add the complementary face cue and raise val accuracy >=1 point.
- Parameters: conv3.weight[29:32] (3x32x3x3 = 864 values) + head refit (513) (729 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.633, 'val_acc': 0.6133, 'val_f1': 0.5958, 'val_margin': 0.4223, 'cv_acc': 0.631, 'cv_margin': 0.6017}
- Reason: no meaningful val improvement (val +0.0066, cv -0.0010, val_f1 +0.0069)

## iter019: [round 3] Larger centre-surround blobs in conv1 (sigma 0.8 -> 1.5) -> REJECTED

- Observed: The two small-scale DoG filters (sigma 0.8/1.6, ~3 px blobs) feed the blob+colour path; at 64x64 an eye or nose is a ~4-8 px blob.
- Hypothesis: Matching DoG scale to part size (sigma 1.5/3.0) makes the blob channels respond to parts and raises val accuracy by >=1 point.
- Parameters: conv1.weight[10:12] (2x3x5x5 = 150 values) + head refit (513) (663 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.64, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.388, 'cv_acc': 0.634, 'cv_margin': 0.5955}
- Reason: no meaningful val improvement (val +0.0000, cv +0.0020, val_f1 +0.0000)

## iter020: [round 4] Odd-phase (sine) Gabors in conv1: step-edge detectors -> REJECTED

- Observed: conv1 oriented filters are cos-phase (even): they respond to thin bars, not to step edges such as body/ear/muzzle outlines against background.
- Hypothesis: Replacing the 8 even Gabors with odd sine-phase Gabors (same wavelength 4, sigma 1.5) detects silhouette edges and raises val accuracy by >=1 point.
- Parameters: conv1.weight[0:8] (600 values) + head refit (513) (1113 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.627, 'val_acc': 0.6, 'val_f1': 0.5918, 'val_margin': 0.3797, 'cv_acc': 0.624, 'cv_margin': 0.5835}
- Reason: no meaningful val improvement (val -0.0067, cv -0.0080, val_f1 +0.0029)

## iter021: [round 4] Sharper orientation tuning in conv2 line filters -> REJECTED

- Observed: conv2 line filters 0-7 also take 0.5 weight from the two neighbouring orientation channels at the centre tap.
- Hypothesis: Removing the neighbour taps makes each line channel orientation-specific and raises val accuracy by >=1 point.
- Parameters: conv2.weight line-filter neighbour taps (16 values) + head refit (513) (529 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.635, 'val_acc': 0.6017, 'val_f1': 0.5829, 'val_margin': 0.3827, 'cv_acc': 0.634, 'cv_margin': 0.5928}
- Reason: no meaningful val improvement (val -0.0050, cv +0.0020, val_f1 -0.0060)

## iter022: [round 4] Half-strength conv3 bias gating on the accepted model -> REJECTED

- Observed: relu3 is 100% active; full centring (iter003) regressed CV by 2.4 pts.
- Hypothesis: Half-strength centring adds mild gating without the regression and raises val accuracy by >=1 point.
- Parameters: conv3.bias (32) + head refit (513) (545 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.634, 'val_acc': 0.5933, 'val_f1': 0.5704, 'val_margin': 0.3801, 'cv_acc': 0.628, 'cv_margin': 0.5742}
- Reason: diagnostic regression (cv -0.0040, val_f1 -0.0185)

## iter023: [round 4] Hierarchy ablation: pass conv1 features straight through conv2/conv3 -> REJECTED

- Observed: 22 edits of every kind left val at 60-61%; the head (closed-form on 512 features) may be doing all the work.
- Hypothesis: Replacing conv2/conv3 with identity routing of the 16 conv1 channels (256 informative features, 256 zero) gives accuracy within 1 point of the accepted model, i.e. the hierarchy adds nothing; if it *raises* val by >=1 point (fewer noisy features) it is accepted.
- Parameters: conv2.weight (4608) + conv3.weight (9216) replaced + head refit (513) (2105 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.645, 'val_acc': 0.5983, 'val_f1': 0.5689, 'val_margin': 0.4593, 'cv_acc': 0.638, 'cv_margin': 0.6015}
- Reason: diagnostic regression (cv +0.0060, val_f1 -0.0200)

## iter024: [round 5] Multi-scale passthrough: 16 fine + 16 coarse conv1 feature maps -> REJECTED

- Observed: iter023: the hand-composed hierarchy is decorative; the conv1 features + closed-form head give the same accuracy. So information lives in conv1 features, and 16 of 32 final channels were empty in iter023.
- Hypothesis: Filling the 16 empty channels with 3x3-box (coarser) averages of the same conv1 features gives 512 informative features across two scales and lifts val accuracy by >=1 point over the accepted model.
- Parameters: conv2.weight (4608) + conv3.weight (9216) replaced + head refit (513) (2221 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.647, 'val_acc': 0.605, 'val_f1': 0.5775, 'val_margin': 0.4562, 'cv_acc': 0.64, 'cv_margin': 0.6075}
- Reason: diagnostic regression (cv +0.0080, val_f1 -0.0114)

## iter025: [round 6] Spatially shared head (32 effective weights) -> REJECTED

- Observed: 24 extractor/head edits plateau at val ~60%, CV ~63%. The 512-feature head is fitted from only 1,000 images, and centre-only edge features (96) already match it: variance, not features, may be the limit.
- Hypothesis: Sharing each channel's weight across all 16 cells (LDA over 32 cell-averaged features) cuts effective parameters 16x, lowers variance and raises val accuracy by >=1 point, at the cost of spatial layout.
- Parameters: head.weight (512 tied to 32 values) + head.bias [closed-form head, rule lda_pooled] (513 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.625, 'val_acc': 0.5917, 'val_f1': 0.5739, 'val_margin': 0.3455, 'cv_acc': 0.629, 'cv_margin': 0.5521}
- Reason: diagnostic regression (cv -0.0030, val_f1 -0.0150)

## iter026: [round 6] Centre/border shared head (64 effective weights) -> REJECTED

- Observed: Per-cell effect sizes: centre cells ~0.43, borders 0.2-0.37 (train).
- Hypothesis: LDA over 64 features (centre-cell mean, border-cell mean per channel) keeps the object-vs-background split with 8x fewer weights than the full head and raises val accuracy by >=1 point.
- Parameters: head.weight (512 tied to 64 values) + head.bias [closed-form head, rule lda_pooled_cb] (513 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.631, 'val_acc': 0.5933, 'val_f1': 0.5779, 'val_margin': 0.3785, 'cv_acc': 0.632, 'cv_margin': 0.6021}
- Reason: diagnostic regression (cv +0.0000, val_f1 -0.0110)

## iter027: [round 6] Luminance-distribution (soft histogram) ramps replace the DoG blobs -> REJECTED

- Observed: iter023/iter025/iter026: pooled low-level features plateau at CV ~63% whatever the head or hierarchy; the DoG channels 8-11 are the least informative conv1 group and encode only local contrast.
- Hypothesis: Four thresholded single-pixel luminance ramps (bright 0.4, very bright 1.1, dark 0.4, very dark 1.1) pooled over the grid give soft histogram masses and raise val accuracy by >=1 point.
- Parameters: conv1.weight[8:12] (300 values), conv1.bias[8:12] (4) + head refit (513) (817 changed)
- Before: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- After: {'train_acc': 0.608, 'val_acc': 0.5917, 'val_f1': 0.5724, 'val_margin': 0.3653, 'cv_acc': 0.606, 'cv_margin': 0.5589}
- Reason: diagnostic regression (cv -0.0260, val_f1 -0.0165)
