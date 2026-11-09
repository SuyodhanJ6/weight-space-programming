# Decision log

## Pre-registered acceptance rule

wsp-005, pre-registered before any edit. Accept iff (a) validation accuracy rises by >= 0.010 (1 percentage point) over the accepted checkpoint AND (b) 5-fold CV accuracy on TRAIN does not fall by more than 0.010 AND (c) validation F1 does not fall by more than 0.010. Train accuracy is recorded but never used to accept. The sealed test split is never read during iterations.

## iter000: baseline

Hand-programmed initial network. {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}

## iter001: De-mean the head (replication of WSP-001 iter001 on a new split) -> REJECTED

- Observed: WSP-005 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500 (2 of 4 wrong: cat_07 scored dog +1.18, dog_09 scored cat -1.07), 5-fold CV 0.917. Head weights nearly all positive (mean +0.113); conv2 ReLU 93% active, conv3 100% active (near-linear); logit r=+0.80 with label, +0.17 with pixel energy.
- Hypothesis: Subtracting the mean head weight removes the energy direction and raises val accuracy by >=1 image (0.25). Replication: WSP-001 saw lower train accuracy and no val gain; the new split tests whether that was chance.
- Parameters: head.weight (512), head.bias (1) [closed-form head] (513 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- After: {'train_acc': 0.6667, 'val_acc': 0.75, 'val_f1': 0.8, 'val_margin': 0.4591, 'cv_acc': 0.5, 'cv_margin': 0.4311}
- Reason: diagnostic regression (cv -0.4167, val_f1 +0.3000)

## iter002: Negative conv2 biases to restore ReLU gating -> REJECTED

- Observed: WSP-005 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500 (2 of 4 wrong: cat_07 scored dog +1.18, dog_09 scored cat -1.07), 5-fold CV 0.917. Head weights nearly all positive (mean +0.113); conv2 ReLU 93% active, conv3 100% active (near-linear); logit r=+0.80 with label, +0.17 with pixel energy.
- Hypothesis: bias_c = -mean(pre-activation_c over train) makes conv2 fire only for above-average evidence and lifts val >= 1 image.
- Parameters: conv2.bias (32) + head refit (513) (545 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- After: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.1335, 'cv_acc': 0.9167, 'cv_margin': 1.2856}
- Reason: no meaningful val improvement (val +0.0000, cv +0.0000, val_f1 +0.0000)

## iter003: Wire the dead colour pathway into conv2 -> REJECTED

- Observed: WSP-005 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500 (2 of 4 wrong: cat_07 scored dog +1.18, dog_09 scored cat -1.07), 5-fold CV 0.917. Head weights nearly all positive (mean +0.113); conv2 ReLU 93% active, conv3 100% active (near-linear); logit r=+0.80 with label, +0.17 with pixel energy. conv2 weights from colour channels 12-15 are exactly zero: the net is colour-blind.
- Hypothesis: Wiring colour-opponent channels 12-15 into conv2 filters 24-31 lets warm/cool evidence reach the head (val >= +1 image).
- Parameters: conv2.weight[24:32, 12:16] (72) + head refit (513) (585 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- After: {'train_acc': 1.0, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 1.0357, 'cv_acc': 0.9167, 'cv_margin': 1.278}
- Reason: no meaningful val improvement (val +0.0000, cv +0.0000, val_f1 +0.0000)

## iter004: Wider spatial pooling in conv3 (3x3 box instead of 1-2-1 blur) -> REJECTED

- Observed: WSP-005 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500 (2 of 4 wrong: cat_07 scored dog +1.18, dog_09 scored cat -1.07), 5-fold CV 0.917. Head weights nearly all positive (mean +0.113); conv2 ReLU 93% active, conv3 100% active (near-linear); logit r=+0.80 with label, +0.17 with pixel energy. Web photos vary in pose and position; the 1-2-1 blur has an effective footprint of about 2 pixels.
- Hypothesis: A uniform 3x3 box in conv3 widens the receptive field (more position tolerance) and raises val by >= 1 image.
- Parameters: conv3.weight (9,216) + head refit (513) (1809 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- After: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2634, 'cv_acc': 0.9167, 'cv_margin': 1.2911}
- Reason: no meaningful val improvement (val +0.0000, cv +0.0000, val_f1 +0.0000)

## iter005: Pooled-cell head (shared weights over the 16 pooled cells) -> REJECTED

- Observed: WSP-005 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500 (2 of 4 wrong: cat_07 scored dog +1.18, dog_09 scored cat -1.07), 5-fold CV 0.917. Head weights nearly all positive (mean +0.113); conv2 ReLU 93% active, conv3 100% active (near-linear); logit r=+0.80 with label, +0.17 with pixel energy. The 512-weight head has 513 free values fit to 12 images.
- Hypothesis: Sharing one weight per channel over the 16 cells (32 weights) reduces variance and raises val by >= 1 image.
- Parameters: head.weight (512), head.bias (1) [closed-form head] (513 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- After: {'train_acc': 0.8333, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.1352, 'cv_acc': 0.8333, 'cv_margin': 1.3793}
- Reason: diagnostic regression (cv -0.0834, val_f1 +0.0000)

## iter006: Down-weight border cells of the head (background suppression) -> REJECTED

- Observed: WSP-005 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500 (2 of 4 wrong: cat_07 scored dog +1.18, dog_09 scored cat -1.07), 5-fold CV 0.917. Head weights nearly all positive (mean +0.113); conv2 ReLU 93% active, conv3 100% active (near-linear); logit r=+0.80 with label, +0.17 with pixel energy. Border cells hold fences, cages and grass.
- Hypothesis: Scaling border-cell head weights by 0.25 reduces background reliance and lifts val by >= 1 image.
- Parameters: head.weight border-cell entries (384 of 512), head.bias [closed-form head] (513 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- After: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.3838, 'cv_acc': 0.9167, 'cv_margin': 1.3648}
- Reason: no meaningful val improvement (val +0.0000, cv +0.0000, val_f1 +0.0000)

## iter007: Centroid head rule (no variance scaling) -> REJECTED

- Observed: WSP-005 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500 (2 of 4 wrong: cat_07 scored dog +1.18, dog_09 scored cat -1.07), 5-fold CV 0.917. Head weights nearly all positive (mean +0.113); conv2 ReLU 93% active, conv3 100% active (near-linear); logit r=+0.80 with label, +0.17 with pixel energy. With n=12 the per-feature variance estimates are very noisy.
- Hypothesis: The plain class-mean difference transfers better and lifts val by >= 1 image.
- Parameters: head.weight (512), head.bias (1) [closed-form head] (513 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- After: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2717, 'cv_acc': 0.9167, 'cv_margin': 1.1265}
- Reason: no meaningful val improvement (val +0.0000, cv +0.0000, val_f1 +0.0000)

## iter008: Half-strength conv3 bias centring -> REJECTED

- Observed: WSP-005 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500 (2 of 4 wrong: cat_07 scored dog +1.18, dog_09 scored cat -1.07), 5-fold CV 0.917. Head weights nearly all positive (mean +0.113); conv2 ReLU 93% active, conv3 100% active (near-linear); logit r=+0.80 with label, +0.17 with pixel energy. conv3 is 100% active: the last layer never gates.
- Hypothesis: Half-strength centring gates only the weakest conv3 evidence without killing channels; val +>=1 image.
- Parameters: conv3.bias (32) + head refit (513) (545 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- After: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.1843, 'cv_acc': 0.9167, 'cv_margin': 1.2761}
- Reason: no meaningful val improvement (val +0.0000, cv +0.0000, val_f1 +0.0000)

## iter009: Full-covariance LDA head (ridge) -> REJECTED

- Observed: WSP-005 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500 (2 of 4 wrong: cat_07 scored dog +1.18, dog_09 scored cat -1.07), 5-fold CV 0.917. Head weights nearly all positive (mean +0.113); conv2 ReLU 93% active, conv3 100% active (near-linear); logit r=+0.80 with label, +0.17 with pixel energy. Diagonal LDA ignores correlation between channels; WSP-003 found full-covariance LDA gave most of its gain at n=8,000.
- Hypothesis: Full-covariance LDA with ridge 0.05 x mean variance de-correlates them. With only 12 samples (rank <= 11) it may overfit; tested as a directional check.
- Parameters: head.weight (512), head.bias (1) [closed-form head] (513 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- After: {'train_acc': 1.0, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.4502, 'cv_acc': 0.6667, 'cv_margin': 1.3113}
- Reason: diagnostic regression (cv -0.2500, val_f1 +0.0000)

## iter010: [exploratory] Centre/border pooled head (64 shared weights) -> REJECTED

- Observed: WSP-005 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500 (2 of 4 wrong: cat_07 scored dog +1.18, dog_09 scored cat -1.07), 5-fold CV 0.917. Head weights nearly all positive (mean +0.113); conv2 ReLU 93% active, conv3 100% active (near-linear); logit r=+0.80 with label, +0.17 with pixel energy. Round 1: 9/9 edits rejected; only iter001 raised val (0.5->0.75) but CV fell 0.917->0.5. iter005 (32 shared weights) lowered CV 0.917->0.833 but kept val.
- Hypothesis: Two shared weights per channel (centre cells, border cells) keep subject/background while cutting parameters; val +>=1 image.
- Parameters: head.weight (512), head.bias (1) [closed-form head] (513 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- After: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2436, 'cv_acc': 0.8333, 'cv_margin': 1.416}
- Reason: diagnostic regression (cv -0.0834, val_f1 +0.0000)

## iter011: [exploratory] Remove colour-opponent filters from conv1 -> REJECTED

- Observed: WSP-005 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500 (2 of 4 wrong: cat_07 scored dog +1.18, dog_09 scored cat -1.07), 5-fold CV 0.917. Head weights nearly all positive (mean +0.113); conv2 ReLU 93% active, conv3 100% active (near-linear); logit r=+0.80 with label, +0.17 with pixel energy. Round 1: 9/9 edits rejected; only iter001 raised val (0.5->0.75) but CV fell 0.917->0.5. iter003 (wiring colour into conv2) only raised train to 1.0, a memorisation sign.
- Hypothesis: Zeroing conv1 filters 12-15 removes colour evidence from the pooled features; val +>=1 image if colour was noise.
- Parameters: conv1.weight[12:16] (300) + head refit (513) (250 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- After: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- Reason: no meaningful val improvement (val +0.0000, cv +0.0000, val_f1 +0.0000)

## iter012: [exploratory] Remove centre-surround blob filters from conv1 -> REJECTED

- Observed: WSP-005 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500 (2 of 4 wrong: cat_07 scored dog +1.18, dog_09 scored cat -1.07), 5-fold CV 0.917. Head weights nearly all positive (mean +0.113); conv2 ReLU 93% active, conv3 100% active (near-linear); logit r=+0.80 with label, +0.17 with pixel energy. Round 1: 9/9 edits rejected; only iter001 raised val (0.5->0.75) but CV fell 0.917->0.5.
- Hypothesis: Removing them reduces noise features; val +>=1 image.
- Parameters: conv1.weight[8:12] (300) + head refit (513) (813 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- After: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2492, 'cv_acc': 0.9167, 'cv_margin': 1.2856}
- Reason: no meaningful val improvement (val +0.0000, cv +0.0000, val_f1 +0.0000)

## iter013: [exploratory] Coarse odd-phase Gabors replace the blob filters -> REJECTED

- Observed: WSP-005 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500 (2 of 4 wrong: cat_07 scored dog +1.18, dog_09 scored cat -1.07), 5-fold CV 0.917. Head weights nearly all positive (mean +0.113); conv2 ReLU 93% active, conv3 100% active (near-linear); logit r=+0.80 with label, +0.17 with pixel energy. Round 1: 9/9 edits rejected; only iter001 raised val (0.5->0.75) but CV fell 0.917->0.5. WSP-004 found odd-phase (step-edge) filters the most valuable group at n=8,000.
- Hypothesis: Odd-phase Gabors at 4 orientations replace blobs 8-11 and add silhouette-edge information; val +>=1 image.
- Parameters: conv1.weight[8:12] (300) + head refit (513) (813 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- After: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.293, 'cv_acc': 0.9167, 'cv_margin': 1.2472}
- Reason: no meaningful val improvement (val +0.0000, cv +0.0000, val_f1 +0.0000)

## iter014: [exploratory] Remove conv2 parallel-pair detectors -> REJECTED

- Observed: WSP-005 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500 (2 of 4 wrong: cat_07 scored dog +1.18, dog_09 scored cat -1.07), 5-fold CV 0.917. Head weights nearly all positive (mean +0.113); conv2 ReLU 93% active, conv3 100% active (near-linear); logit r=+0.80 with label, +0.17 with pixel energy. Round 1: 9/9 edits rejected; only iter001 raised val (0.5->0.75) but CV fell 0.917->0.5.
- Hypothesis: Zeroing them lowers texture noise; val +>=1 image.
- Parameters: conv2.weight[16:24] (1,152) + head refit (513) (537 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- After: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2913, 'cv_acc': 0.9167, 'cv_margin': 1.2839}
- Reason: no meaningful val improvement (val +0.0000, cv +0.0000, val_f1 +0.0000)

## iter015: [exploratory] Remove conv3 isotropic-energy channels -> REJECTED

- Observed: WSP-005 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500 (2 of 4 wrong: cat_07 scored dog +1.18, dog_09 scored cat -1.07), 5-fold CV 0.917. Head weights nearly all positive (mean +0.113); conv2 ReLU 93% active, conv3 100% active (near-linear); logit r=+0.80 with label, +0.17 with pixel energy. Round 1: 9/9 edits rejected; only iter001 raised val (0.5->0.75) but CV fell 0.917->0.5. Baseline logit correlates only +0.17 with pixel energy, but 8 of 32 conv3 channels are pure energy meters.
- Hypothesis: Zeroing conv3 24-31 removes redundant energy features; val +>=1 image.
- Parameters: conv3.weight[24:32] (2,304) + head refit (513) (1089 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- After: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2599, 'cv_acc': 0.9167, 'cv_margin': 1.3025}
- Reason: no meaningful val improvement (val +0.0000, cv +0.0000, val_f1 +0.0000)

## iter016: [exploratory] Centre-only head (border weight 0) -> REJECTED

- Observed: WSP-005 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500 (2 of 4 wrong: cat_07 scored dog +1.18, dog_09 scored cat -1.07), 5-fold CV 0.917. Head weights nearly all positive (mean +0.113); conv2 ReLU 93% active, conv3 100% active (near-linear); logit r=+0.80 with label, +0.17 with pixel energy. Round 1: 9/9 edits rejected; only iter001 raised val (0.5->0.75) but CV fell 0.917->0.5. iter006 (border x0.25) changed nothing.
- Hypothesis: Setting border weights to exactly 0 removes background entirely; val +>=1 image if background is a spurious cue.
- Parameters: head.weight (512 incl. 384 zeroed), head.bias (513 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- After: {'train_acc': 0.8333, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.4453, 'cv_acc': 0.8333, 'cv_margin': 1.3979}
- Reason: diagnostic regression (cv -0.0834, val_f1 +0.0000)

## iter017: [exploratory] Top-half emphasis in the head -> REJECTED

- Observed: WSP-005 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500 (2 of 4 wrong: cat_07 scored dog +1.18, dog_09 scored cat -1.07), 5-fold CV 0.917. Head weights nearly all positive (mean +0.113); conv2 ReLU 93% active, conv3 100% active (near-linear); logit r=+0.80 with label, +0.17 with pixel energy. Round 1: 9/9 edits rejected; only iter001 raised val (0.5->0.75) but CV fell 0.917->0.5. Faces and heads sit in the upper half of most framed pet photos; bodies/floors in the lower half.
- Hypothesis: Weighting the top two pooled rows x1 and the bottom two x0.25 sharpens face evidence; val +>=1 image.
- Parameters: head.weight (512), head.bias [closed-form head] (513 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- After: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.269, 'cv_acc': 0.8333, 'cv_margin': 1.19}
- Reason: diagnostic regression (cv -0.0834, val_f1 +0.0000)

## iter018: [exploratory] Break conv3 filter symmetry (replication of WSP-001 iter006) -> REJECTED

- Observed: WSP-005 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500 (2 of 4 wrong: cat_07 scored dog +1.18, dog_09 scored cat -1.07), 5-fold CV 0.917. Head weights nearly all positive (mean +0.113); conv2 ReLU 93% active, conv3 100% active (near-linear); logit r=+0.80 with label, +0.17 with pixel energy. Round 1: 9/9 edits rejected; only iter001 raised val (0.5->0.75) but CV fell 0.917->0.5. conv3 filters 24-31 produce identical outputs, so the head counts one feature 8 times.
- Hypothesis: Eight distinct spatial kernels (vertical, horizontal, diagonals, left, right, top, bottom) give distinct layout features; val +>=1 image.
- Parameters: conv3.weight[24:32, 24:32] (576) + head refit (513) (1089 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- After: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2754, 'cv_acc': 0.9167, 'cv_margin': 1.3019}
- Reason: no meaningful val improvement (val +0.0000, cv +0.0000, val_f1 +0.0000)

## iter019: [exploratory] Control: pooled head with explicit uniform cell mask -> REJECTED

- Observed: WSP-005 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500 (2 of 4 wrong: cat_07 scored dog +1.18, dog_09 scored cat -1.07), 5-fold CV 0.917. Head weights nearly all positive (mean +0.113); conv2 ReLU 93% active, conv3 100% active (near-linear); logit r=+0.80 with label, +0.17 with pixel energy. Round 1: 9/9 edits rejected; only iter001 raised val (0.5->0.75) but CV fell 0.917->0.5. iter005's regression could come from the pooled rule or from the head-fitting path.
- Hypothesis: A pooled head built with an explicit all-ones mask reproduces iter005 exactly (train .833, CV .833); a difference would mean a code-path bug.
- Parameters: head.weight (512), head.bias [closed-form head] (513 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- After: {'train_acc': 0.8333, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.1352, 'cv_acc': 0.8333, 'cv_margin': 1.3793}
- Reason: diagnostic regression (cv -0.0834, val_f1 +0.0000)
