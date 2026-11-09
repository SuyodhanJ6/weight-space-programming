# Decision log

## Pre-registered acceptance rule

wsp-003, pre-registered before any edit. Accept iff (a) validation accuracy rises by >= 0.010 (1 percentage point) over the accepted checkpoint AND (b) 5-fold CV accuracy on TRAIN does not fall by more than 0.010 AND (c) validation F1 does not fall by more than 0.010. Train accuracy is recorded but never used to accept. The sealed test split is never read during iterations.

## iter000: baseline

Hand-programmed initial network. {'train_acc': 0.6021, 'val_acc': 0.6013, 'val_f1': 0.5862, 'val_margin': 0.473, 'cv_acc': 0.6014, 'cv_margin': 0.462}

## iter001: Wire the colour pathway into conv2 (replication at 8x data) -> ACCEPTED

- Observed: WSP-003 baseline: train 0.602, val 0.601 (n=3000), 5-fold CV 0.601: all equal, so the model is bias-limited (features not rich enough), not variance-limited. conv2 receives 0.0 weight from the 4 colour channels.
- Hypothesis: Wiring colour into conv2 blob filters 24-31 raises val accuracy by >=1 point. (WSP-002: +3.5 pts val on 600 images but 0 on the 2,000-image test; re-tested where a 1-pt effect is resolvable, n=3000.)
- Parameters: conv2.weight[24:32,12:16] (72) + head refit (513) (585 changed)
- Before: {'train_acc': 0.6021, 'val_acc': 0.6013, 'val_f1': 0.5862, 'val_margin': 0.473, 'cv_acc': 0.6014, 'cv_margin': 0.462}
- After: {'train_acc': 0.6204, 'val_acc': 0.6243, 'val_f1': 0.6094, 'val_margin': 0.5641, 'cv_acc': 0.6192, 'cv_margin': 0.5373}
- Reason: val improvement (val +0.0230, cv +0.0178, val_f1 +0.0232)

## iter002: Multi-scale passthrough: 16 fine + 16 coarse conv1 maps (replication at 8x data) -> REJECTED

- Observed: WSP-003 baseline: train 0.602, val 0.601 (n=3000), 5-fold CV 0.601: all equal, so the model is bias-limited (features not rich enough), not variance-limited. WSP-002 iter024 reached CV 0.640 vs 0.632 (small data); the hierarchy ablation showed conv2/conv3 composition adds nothing over conv1 features.
- Hypothesis: Filling all 32 final channels with informative fine+coarse conv1 features raises val accuracy by >=1 point; with 8,000 train images the 512-weight head can now use them without variance penalty.
- Parameters: conv2.weight (4608) + conv3.weight (9216) replaced + head refit (513) (2221 changed)
- Before: {'train_acc': 0.6204, 'val_acc': 0.6243, 'val_f1': 0.6094, 'val_margin': 0.5641, 'cv_acc': 0.6192, 'cv_margin': 0.5373}
- After: {'train_acc': 0.6285, 'val_acc': 0.6263, 'val_f1': 0.6141, 'val_margin': 0.5853, 'cv_acc': 0.6286, 'cv_margin': 0.5697}
- Reason: no meaningful val improvement (val +0.0020, cv +0.0094, val_f1 +0.0047)

## iter003: Three-scale pyramid: fine blur + mid + coarse box -> REJECTED

- Observed: Features are scale-limited: the head sees ~6 px (fine) and ~12 px (mid) structure only.
- Hypothesis: Adding a coarse (3x3 box over the mid scale, ~12-24 px) channel set raises val accuracy by >=1 point.
- Parameters: conv2.weight (4608) + conv3.weight (9216) replaced + head refit (513) (2221 changed)
- Before: {'train_acc': 0.6204, 'val_acc': 0.6243, 'val_f1': 0.6094, 'val_margin': 0.5641, 'cv_acc': 0.6192, 'cv_margin': 0.5373}
- After: {'train_acc': 0.6288, 'val_acc': 0.626, 'val_f1': 0.6134, 'val_margin': 0.5845, 'cv_acc': 0.6281, 'cv_margin': 0.5689}
- Reason: no meaningful val improvement (val +0.0017, cv +0.0089, val_f1 +0.0040)

## iter004: Colour x texture conjunction detectors (thresholded soft-AND) -> REJECTED

- Observed: On 8,000 train images: dogs have more of every channel (signed d>0), strongest for G-R (grass-green) and Y-B (tan/yellow) colour channels and near-horizontal edges; coarse copies of channels carry the same signal as fine ones (|d| 0.285 vs 0.284), so multi-scale added little. No ReLU ever gates: no feature can express a conjunction.
- Hypothesis: 16 conjunction channels ReLU(colour/s_c + texture/s_t - 2) over {4 colour} x {horizontal, vertical, diagonal, blob} texture groups add genuinely new information and raise val accuracy by >=1 point. (Likely exploits scene context such as grass; that would be a context cue, not animal recognition.)
- Parameters: conv2.weight+bias (4608+32), conv3.weight (9216) + head refit (513) (2245 changed)
- Before: {'train_acc': 0.6204, 'val_acc': 0.6243, 'val_f1': 0.6094, 'val_margin': 0.5641, 'cv_acc': 0.6192, 'cv_margin': 0.5373}
- After: {'train_acc': 0.6305, 'val_acc': 0.6307, 'val_f1': 0.6112, 'val_margin': 0.6053, 'cv_acc': 0.6292, 'cv_margin': 0.586}
- Reason: no meaningful val improvement (val +0.0064, cv +0.0100, val_f1 +0.0018)

## iter005: Fine conjunctions: every colour x every single orientation (32 channels) -> ACCEPTED

- Observed: iter004 (grouped conjunctions + 16 passthrough channels): val +0.64, CV +1.00 vs the accepted model: conjunctions add real information but fall just under the 1-pt rule.
- Hypothesis: Using all 32 channels for colour x single-orientation soft-ANDs gives a richer, non-redundant descriptor and raises val accuracy by >=1 point over the accepted model.
- Parameters: conv2.weight+bias (4608+32), conv3.weight (9216) + head refit (513) (2231 changed)
- Before: {'train_acc': 0.6204, 'val_acc': 0.6243, 'val_f1': 0.6094, 'val_margin': 0.5641, 'cv_acc': 0.6192, 'cv_margin': 0.5373}
- After: {'train_acc': 0.6288, 'val_acc': 0.6357, 'val_f1': 0.6064, 'val_margin': 0.5789, 'cv_acc': 0.6279, 'cv_margin': 0.5619}
- Reason: val improvement (val +0.0114, cv +0.0087, val_f1 -0.0030)

## iter006: Hue-bin and colour x blob conjunctions replace the weakest-orientation slots -> REJECTED

- Observed: iter005 accepted (val 0.6357): colour x orientation conjunctions work. Orientations 3-5 had the lowest train |d| (0.16-0.18) so their 12 colour conjunctions are the weakest slots. Colour x colour (hue combinations such as yellow-green, orange) and colour x blob are untried conjunction types.
- Hypothesis: Replacing those slots with 4 hue-bin ANDs (colour x colour) and 8 colour x DoG ANDs adds new information and raises val accuracy by >=1 point.
- Parameters: conv2.weight rows for 12 slots (12x16x3x3 = 1728) + head refit (513); conv2.bias unchanged (-2.0) (549 changed)
- Before: {'train_acc': 0.6288, 'val_acc': 0.6357, 'val_f1': 0.6064, 'val_margin': 0.5789, 'cv_acc': 0.6279, 'cv_margin': 0.5619}
- After: {'train_acc': 0.6211, 'val_acc': 0.6327, 'val_f1': 0.6001, 'val_margin': 0.5574, 'cv_acc': 0.6221, 'cv_margin': 0.5418}
- Reason: no meaningful val improvement (val -0.0030, cv -0.0058, val_f1 -0.0063)

## iter007: Neighbourhood-tolerant conjunctions (colour at centre AND orientation over 3x3) -> REJECTED

- Observed: iter005 accepted. Conjunction channels fire at 25-37% of positions (sensible gating). Colour (5x5 blurred) and Gabor responses are not pixel-aligned, and boundaries (fur vs background) put edge and colour in adjacent, not identical, cells.
- Hypothesis: Averaging the orientation input over the 3x3 neighbourhood (misregistration tolerance) makes the conjunction fire for colour regions bounded by oriented edges and raises val accuracy by >=1 point.
- Parameters: conv2.weight rows 0-31 (32x16x3x3 = 4608) + head refit (513) (801 changed)
- Before: {'train_acc': 0.6288, 'val_acc': 0.6357, 'val_f1': 0.6064, 'val_margin': 0.5789, 'cv_acc': 0.6279, 'cv_margin': 0.5619}
- After: {'train_acc': 0.6189, 'val_acc': 0.6233, 'val_f1': 0.5796, 'val_margin': 0.5086, 'cv_acc': 0.6171, 'cv_margin': 0.4964}
- Reason: diagnostic regression (cv -0.0108, val_f1 -0.0268)

## iter008: Higher AND threshold (-2.0 -> -3.0): rarer, stronger co-occurrences -> REJECTED

- Observed: Pre-activations are heavy-tailed (ch0 quantiles: 50%=-0.70, 70%=0.33, 90%=2.48, 99%=7.9); units are active 25-37% of the time.
- Hypothesis: Requiring summed normalised evidence >= 3 isolates strong co-occurrences that are more class-specific and raises val accuracy by >=1 point. (One directional test, not a sweep.)
- Parameters: conv2.bias (32) + head refit (513) (545 changed)
- Before: {'train_acc': 0.6288, 'val_acc': 0.6357, 'val_f1': 0.6064, 'val_margin': 0.5789, 'cv_acc': 0.6279, 'cv_margin': 0.5619}
- After: {'train_acc': 0.6308, 'val_acc': 0.6333, 'val_f1': 0.5974, 'val_margin': 0.5458, 'cv_acc': 0.6288, 'cv_margin': 0.5435}
- Reason: no meaningful val improvement (val -0.0024, cv +0.0009, val_f1 -0.0090)

## iter009: Full-covariance LDA head (closed form) [data-fitted head, not a hand edit] -> ACCEPTED

- Observed: train 0.629 / val 0.636 / CV 0.628 agree: bias-limited. The 512 pooled features are strongly correlated (fine/coarse and conjunction families), yet the head divides by per-feature variance only.
- Hypothesis: w = (Sigma_within + 0.05*mean_var*I)^-1 (mu_dog - mu_cat), solved once from the 8,000 train images, raises val accuracy by >=1 point. Ridge fixed a priori, not tuned.
- Parameters: head.weight (512), head.bias [closed-form head, rule lda_full] (513 changed)
- Before: {'train_acc': 0.6288, 'val_acc': 0.6357, 'val_f1': 0.6064, 'val_margin': 0.5789, 'cv_acc': 0.6279, 'cv_margin': 0.5619}
- After: {'train_acc': 0.7515, 'val_acc': 0.7443, 'val_f1': 0.7338, 'val_margin': 1.039, 'cv_acc': 0.7418, 'cv_margin': 1.0472}
- Reason: val improvement (val +0.1086, cv +0.1139, val_f1 +0.1274)

## iter010: [full-LDA head] 16 marginal passthrough + 16 grouped conjunction channels -> REJECTED

- Observed: iter009: full-covariance LDA lifts val 0.636 -> 0.744 (train 0.752, CV 0.742): correlated features were the bottleneck, so redundancy no longer hurts and feature diversity may now pay off.
- Hypothesis: Mixing 16 marginal passthrough channels with 16 grouped conjunctions (iter004 design) adds information the full-covariance head can exploit and raises val accuracy by >=1 point over 0.7443.
- Parameters: conv2.weight+bias (4608+32), conv3.weight (9216) + head refit (513, lda_full) (648 changed)
- Before: {'train_acc': 0.7515, 'val_acc': 0.7443, 'val_f1': 0.7338, 'val_margin': 1.039, 'cv_acc': 0.7418, 'cv_margin': 1.0472}
- After: {'train_acc': 0.7272, 'val_acc': 0.7177, 'val_f1': 0.7082, 'val_margin': 0.9427, 'cv_acc': 0.7177, 'cv_margin': 0.9528}
- Reason: diagnostic regression (cv -0.0241, val_f1 -0.0256)

## iter011: [full-LDA head] Multi-scale passthrough (16 fine + 16 coarse conv1 maps) -> REJECTED

- Observed: Under the diagonal head multi-scale gave CV +0.9 pts but missed the rule; coarse copies were near duplicates (|d| 0.285 vs 0.284). A full-covariance head treats duplicates correctly.
- Hypothesis: Fine + coarse passthrough exceeds the conjunction extractor (val 0.744) by >=1 point.
- Parameters: conv2.weight (4608) + conv3.weight (9216) replaced + head refit (513, lda_full) (719 changed)
- Before: {'train_acc': 0.7515, 'val_acc': 0.7443, 'val_f1': 0.7338, 'val_margin': 1.039, 'cv_acc': 0.7418, 'cv_margin': 1.0472}
- After: {'train_acc': 0.5152, 'val_acc': 0.5043, 'val_f1': 0.6601, 'val_margin': 0.1565, 'cv_acc': 0.5069, 'cv_margin': 0.0724}
- Reason: diagnostic regression (cv -0.2349, val_f1 -0.0737)

## iter012: [full-LDA head] Three-scale pyramid with a coarse box scale -> REJECTED

- Observed: Same as iter011; features lack object-scale (20-40 px) structure.
- Hypothesis: A coarse 3x3-box-of-mid scale adds larger-scale information and beats 0.744 by >=1 point.
- Parameters: conv2.weight (4608) + conv3.weight (9216) replaced + head refit (513, lda_full) (863 changed)
- Before: {'train_acc': 0.7515, 'val_acc': 0.7443, 'val_f1': 0.7338, 'val_margin': 1.039, 'cv_acc': 0.7418, 'cv_margin': 1.0472}
- After: {'train_acc': 0.5151, 'val_acc': 0.5043, 'val_f1': 0.6603, 'val_margin': 0.1553, 'cv_acc': 0.5072, 'cv_margin': 0.0755}
- Reason: diagnostic regression (cv -0.2346, val_f1 -0.0735)

## iter013: [full-LDA head] Multi-scale passthrough with conv2.bias reset (valid re-run of iter011) -> ACCEPTED

- Observed: iter011/iter012 collapsed to 50% because the edit left conv2.bias=-2.0 from the conjunction model: 384 of 512 features were exactly zero (debugged). Those two results are INVALID tests, not evidence.
- Hypothesis: Fine + coarse passthrough (bias reset to 0) beats the conjunction extractor (val 0.744) by >=1 point.
- Parameters: conv2.bias (32) reset, conv2.weight (4608), conv3.weight (9216) + head refit (513, lda_full) (751 changed)
- Before: {'train_acc': 0.7515, 'val_acc': 0.7443, 'val_f1': 0.7338, 'val_margin': 1.039, 'cv_acc': 0.7418, 'cv_margin': 1.0472}
- After: {'train_acc': 0.7674, 'val_acc': 0.7663, 'val_f1': 0.7587, 'val_margin': 1.1243, 'cv_acc': 0.7564, 'cv_margin': 1.1082}
- Reason: val improvement (val +0.0220, cv +0.0146, val_f1 +0.0249)

## iter014: [full-LDA head] Three-scale pyramid with conv2.bias reset (valid re-run of iter012) -> REJECTED

- Observed: Same bug as iter011 (stale -2.0 bias).
- Hypothesis: Fine + mid + coarse features beat val 0.744 by >=1 point.
- Parameters: conv2.bias (32) reset, conv2.weight (4608), conv3.weight (9216) + head refit (513, lda_full) (657 changed)
- Before: {'train_acc': 0.7674, 'val_acc': 0.7663, 'val_f1': 0.7587, 'val_margin': 1.1243, 'cv_acc': 0.7564, 'cv_margin': 1.1082}
- After: {'train_acc': 0.7664, 'val_acc': 0.7677, 'val_f1': 0.7604, 'val_margin': 1.125, 'cv_acc': 0.7566, 'cv_margin': 1.1084}
- Reason: no meaningful val improvement (val +0.0014, cv +0.0002, val_f1 +0.0017)

## iter015: [full-LDA head] Two spatial-frequency Gabor bands (odd orientations -> wavelength 6) -> REJECTED

- Observed: Accepted model (iter013): val 0.766. Fine and coarse passthrough copies are near-duplicates (|d| 0.285 vs 0.284); a full-covariance head can use complementary features but gains nothing from copies.
- Hypothesis: Moving 4 of the 8 orientations (1,3,5,7) to wavelength 6 gives 4 angles x 2 frequency bands, adds new texture information and raises val accuracy by >=1 point.
- Parameters: conv1.weight[1,3,5,7] (4x3x5x5 = 300 values) + head refit (513, lda_full) (813 changed)
- Before: {'train_acc': 0.7674, 'val_acc': 0.7663, 'val_f1': 0.7587, 'val_margin': 1.1243, 'cv_acc': 0.7564, 'cv_margin': 1.1082}
- After: {'train_acc': 0.766, 'val_acc': 0.7673, 'val_f1': 0.7603, 'val_margin': 1.1215, 'cv_acc': 0.758, 'cv_margin': 1.1081}
- Reason: no meaningful val improvement (val +0.0010, cv +0.0016, val_f1 +0.0016)

## iter016: [full-LDA head] Local-peakiness channels replace the coarse marginal copies -> REJECTED

- Observed: Fine + coarse passthrough (val 0.766) beats fine + conjunctions (0.718): scale contrast is exploitable by the full-covariance head. Coarse copies, however, are linear in the fine features before pooling, so they add only what a linear average adds.
- Hypothesis: ReLU(f - 3x3 mean of f) per conv1 feature adds nonlinear local-contrast information not present in linear fine/coarse averages, and raises val accuracy by >=1 point over 0.7663.
- Parameters: conv2.bias reset, conv2.weight (4608), conv3.weight (9216) + head refit (513, lda_full) (657 changed)
- Before: {'train_acc': 0.7674, 'val_acc': 0.7663, 'val_f1': 0.7587, 'val_margin': 1.1243, 'cv_acc': 0.7564, 'cv_margin': 1.1082}
- After: {'train_acc': 0.7738, 'val_acc': 0.7733, 'val_f1': 0.7645, 'val_margin': 1.1485, 'cv_acc': 0.7641, 'cv_margin': 1.1209}
- Reason: no meaningful val improvement (val +0.0070, cv +0.0077, val_f1 +0.0058)

## iter017: [full-LDA head] Diagonal hue sectors replace the DoG blob filters -> REJECTED

- Observed: Ablation (fit on train, val check): fine marginals alone reach val 0.765; dropping orientation filters costs 7.5 pts, colour 2.2 pts, DoG only 0.9 pts; colour alone reaches 0.677. Information is in the 16 conv1 filters, and the DoG group is the weakest.
- Hypothesis: Replacing the 4 DoG filters with 4 diagonal hue-sector ramps (orange, yellow-green, blue-green, purple) doubles hue resolution (ReLU of a sum differs from the sum of ReLUs) and raises val accuracy >=1 point over 0.7663.
- Parameters: conv1.weight[8:12] (4x3x5x5 = 300 values) + head refit (513, lda_full) (813 changed)
- Before: {'train_acc': 0.7674, 'val_acc': 0.7663, 'val_f1': 0.7587, 'val_margin': 1.1243, 'cv_acc': 0.7564, 'cv_margin': 1.1082}
- After: {'train_acc': 0.7624, 'val_acc': 0.7597, 'val_f1': 0.7508, 'val_margin': 1.0902, 'cv_acc': 0.751, 'cv_margin': 1.0811}
- Reason: no meaningful val improvement (val -0.0066, cv -0.0054, val_f1 -0.0079)

## iter018: [full-LDA head] Odd-phase Gabors replace the DoG blob filters -> REJECTED

- Observed: Same ablation: orientation filters carry the most information (without them val 0.698; only them 0.733).
- Hypothesis: Odd-phase partners at orientations 0,2,4,6 give quadrature phase information and raise val accuracy >=1 point over 0.7663.
- Parameters: conv1.weight[8:12] (300 values) + head refit (513, lda_full) (813 changed)
- Before: {'train_acc': 0.7674, 'val_acc': 0.7663, 'val_f1': 0.7587, 'val_margin': 1.1243, 'cv_acc': 0.7564, 'cv_margin': 1.1082}
- After: {'train_acc': 0.7724, 'val_acc': 0.7733, 'val_f1': 0.7662, 'val_margin': 1.139, 'cv_acc': 0.7622, 'cv_margin': 1.1192}
- Reason: no meaningful val improvement (val +0.0070, cv +0.0058, val_f1 +0.0075)

## iter019: [full-LDA head] Combine local-peakiness channels (iter016) and odd-phase Gabors (iter018) -> REJECTED

- Observed: iter016 (peaks) gave val +0.70 / CV +0.77 and iter018 (odd-phase Gabors) val +0.70 / CV +0.58, each just under the 1-pt rule and consistent across val and 8,000-image CV. They modify different layers.
- Hypothesis: The two effects are additive: together they raise val accuracy by >=1 point over 0.7663.
- Parameters: conv1.weight[8:12] (300) + conv2.bias (32) + conv2.weight (4608) + conv3.weight (9216) + head refit (513) (957 changed)
- Before: {'train_acc': 0.7674, 'val_acc': 0.7663, 'val_f1': 0.7587, 'val_margin': 1.1243, 'cv_acc': 0.7564, 'cv_margin': 1.1082}
- After: {'train_acc': 0.7772, 'val_acc': 0.776, 'val_f1': 0.7689, 'val_margin': 1.1615, 'cv_acc': 0.7634, 'cv_margin': 1.1337}
- Reason: no meaningful val improvement (val +0.0097, cv +0.0070, val_f1 +0.0102)

## iter020: [round 2] Dark-bar polarity: negated Gabors replace the DoG blob filters -> REJECTED

- Observed: Every even Gabor passes through a ReLU, so only the bright-bar half of its response survives; dark bars (dark fur against light background) are invisible. DoG and colour channels already have both polarities.
- Hypothesis: Negated Gabors at orientations 0,2,4,6 recover dark-bar responses for the strongest orientations and raise val accuracy by >=1 point over 0.7663. (Sibling of iter018, which tried odd phase: +0.7.)
- Parameters: conv1.weight[8:12] (300 values) + head refit (513, lda_full) (813 changed)
- Before: {'train_acc': 0.7674, 'val_acc': 0.7663, 'val_f1': 0.7587, 'val_margin': 1.1243, 'cv_acc': 0.7564, 'cv_margin': 1.1082}
- After: {'train_acc': 0.7676, 'val_acc': 0.763, 'val_f1': 0.7549, 'val_margin': 1.1137, 'cv_acc': 0.758, 'cv_margin': 1.0967}
- Reason: no meaningful val improvement (val -0.0033, cv +0.0016, val_f1 -0.0038)

## iter021: [round 2] Spend all channels on the 10 best conv1 maps x {fine, coarse, peak} -> REJECTED

- Observed: Ablation: orientation filters 0,1,2,6,7 had the strongest |d|; weak ones (3,4,5) and the DoG group add little. Currently 32 channels = 16 maps x 2 types, including the weak maps.
- Hypothesis: Using 30 channels as 10 strongest maps x 3 feature types (fine, 3x3 box, local peak) gives the full-covariance head more informative features and raises val accuracy by >=1 point.
- Parameters: conv2.bias reset, conv2.weight (4608), conv3.weight (9216) + head refit (513, lda_full) (875 changed)
- Before: {'train_acc': 0.7674, 'val_acc': 0.7663, 'val_f1': 0.7587, 'val_margin': 1.1243, 'cv_acc': 0.7564, 'cv_margin': 1.1082}
- After: {'train_acc': 0.7672, 'val_acc': 0.7657, 'val_f1': 0.756, 'val_margin': 1.1201, 'cv_acc': 0.7578, 'cv_margin': 1.1048}
- Reason: no meaningful val improvement (val -0.0006, cv +0.0014, val_f1 -0.0027)
