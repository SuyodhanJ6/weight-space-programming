# Decision log

## Pre-registered acceptance rule

wsp-006, rule v3, pre-registered before any edit (replaces v2 after WSP-005 rejected 19 of 19 edits). Accept iff pooled leave-one-out accuracy over ALL 16 train+val images (head rule refit on 15, predict 1) rises by >= 1 image (0.0625) over the accepted checkpoint AND leave-one-out F1 does not fall. Train accuracy, the 4-image val split alone and 5-fold CV are recorded but never used to accept. The sealed test split is never read during iterations.

## iter000: baseline

Hand-programmed initial network. {'loo_acc': 0.75, 'loo_f1': 0.75, 'loo_margin': 0.8846, 'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': -0.133, 'cv_acc': 0.8333, 'cv_margin': 1.0472}

## iter001: De-mean the head (replication of WSP-001 iter001 on a new split) -> REJECTED

- Observed: WSP-006 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500, 5-fold CV 0.833, pooled leave-one-out over all 16 train+val images 0.750 (12/16), LOO F1 0.750. Decision metric is pooled LOO accuracy (rule v3). Same hypothesis as WSP-005 iter001 (De-mean the head (replication of WSP-001 iter001 on a new split)).
- Hypothesis: Subtracting the mean head weight removes the energy direction and raises val accuracy by >=1 image (0.25). Replication: WSP-001 saw lower train accuracy and no val gain; the new split tests whether that was chance.
- Parameters: head.weight (512), head.bias (1) [closed-form head] (513 changed)
- Before: {'loo_acc': 0.75, 'loo_f1': 0.75, 'loo_margin': 0.8846, 'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': -0.133, 'cv_acc': 0.8333, 'cv_margin': 1.0472}
- After: {'loo_acc': 0.6875, 'loo_f1': 0.7059, 'loo_margin': 0.722, 'train_acc': 0.5833, 'val_acc': 1.0, 'val_f1': 1.0, 'val_margin': 1.1126, 'cv_acc': 0.5833, 'cv_margin': 0.0824}
- Reason: LOO F1 regression (loo_acc -0.0625, loo_f1 -0.0441)

## iter002: Negative conv2 biases to restore ReLU gating -> REJECTED

- Observed: WSP-006 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500, 5-fold CV 0.833, pooled leave-one-out over all 16 train+val images 0.750 (12/16), LOO F1 0.750. Decision metric is pooled LOO accuracy (rule v3). Same hypothesis as WSP-005 iter002 (Negative conv2 biases to restore ReLU gating).
- Hypothesis: bias_c = -mean(pre-activation_c over train) makes conv2 fire only for above-average evidence and lifts val >= 1 image.
- Parameters: conv2.bias (32) + head refit (513) (545 changed)
- Before: {'loo_acc': 0.75, 'loo_f1': 0.75, 'loo_margin': 0.8846, 'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': -0.133, 'cv_acc': 0.8333, 'cv_margin': 1.0472}
- After: {'loo_acc': 0.75, 'loo_f1': 0.7143, 'loo_margin': 0.8817, 'train_acc': 0.9167, 'val_acc': 0.25, 'val_f1': 0.0, 'val_margin': 0.0039, 'cv_acc': 0.9167, 'cv_margin': 0.9901}
- Reason: LOO F1 regression (loo_acc +0.0000, loo_f1 -0.0357)

## iter003: Wire the dead colour pathway into conv2 -> ACCEPTED

- Observed: WSP-006 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500, 5-fold CV 0.833, pooled leave-one-out over all 16 train+val images 0.750 (12/16), LOO F1 0.750. Decision metric is pooled LOO accuracy (rule v3). Same hypothesis as WSP-005 iter003 (Wire the dead colour pathway into conv2).
- Hypothesis: Wiring colour-opponent channels 12-15 into conv2 filters 24-31 lets warm/cool evidence reach the head (val >= +1 image).
- Parameters: conv2.weight[24:32, 12:16] (72) + head refit (513) (585 changed)
- Before: {'loo_acc': 0.75, 'loo_f1': 0.75, 'loo_margin': 0.8846, 'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': -0.133, 'cv_acc': 0.8333, 'cv_margin': 1.0472}
- After: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0927, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.171, 'cv_acc': 0.9167, 'cv_margin': 1.1598}
- Reason: LOO improvement (loo_acc +0.0625, loo_f1 +0.0500)

## iter004: Wider spatial pooling in conv3 (3x3 box instead of 1-2-1 blur) -> REJECTED

- Observed: WSP-006 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500, 5-fold CV 0.833, pooled leave-one-out over all 16 train+val images 0.750 (12/16), LOO F1 0.750. Decision metric is pooled LOO accuracy (rule v3). Same hypothesis as WSP-005 iter004 (Wider spatial pooling in conv3 (3x3 box instead of 1-2-1 blur)).
- Hypothesis: A uniform 3x3 box in conv3 widens the receptive field (more position tolerance) and raises val by >= 1 image.
- Parameters: conv3.weight (9,216) + head refit (513) (1809 changed)
- Before: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0927, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.171, 'cv_acc': 0.9167, 'cv_margin': 1.1598}
- After: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 0.8692, 'train_acc': 0.9167, 'val_acc': 0.25, 'val_f1': 0.0, 'val_margin': 0.0305, 'cv_acc': 0.9167, 'cv_margin': 1.0526}
- Reason: no meaningful LOO improvement (loo_acc +0.0000, loo_f1 +0.0000)

## iter005: Pooled-cell head (shared weights over the 16 pooled cells) -> REJECTED

- Observed: WSP-006 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500, 5-fold CV 0.833, pooled leave-one-out over all 16 train+val images 0.750 (12/16), LOO F1 0.750. Decision metric is pooled LOO accuracy (rule v3). Same hypothesis as WSP-005 iter005 (Pooled-cell head (shared weights over the 16 pooled cells)).
- Hypothesis: Sharing one weight per channel over the 16 cells (32 weights) reduces variance and raises val by >= 1 image.
- Parameters: head.weight (512), head.bias (1) [closed-form head] (513 changed)
- Before: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0927, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.171, 'cv_acc': 0.9167, 'cv_margin': 1.1598}
- After: {'loo_acc': 0.625, 'loo_f1': 0.5714, 'loo_margin': 1.1764, 'train_acc': 0.9167, 'val_acc': 0.25, 'val_f1': 0.0, 'val_margin': -0.05, 'cv_acc': 0.9167, 'cv_margin': 1.4565}
- Reason: LOO F1 regression (loo_acc -0.1875, loo_f1 -0.2286)

## iter006: Down-weight border cells of the head (background suppression) -> REJECTED

- Observed: WSP-006 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500, 5-fold CV 0.833, pooled leave-one-out over all 16 train+val images 0.750 (12/16), LOO F1 0.750. Decision metric is pooled LOO accuracy (rule v3). Same hypothesis as WSP-005 iter006 (Down-weight border cells of the head (background suppression)).
- Hypothesis: Scaling border-cell head weights by 0.25 reduces background reliance and lifts val by >= 1 image.
- Parameters: head.weight border-cell entries (384 of 512), head.bias [closed-form head] (513 changed)
- Before: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0927, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.171, 'cv_acc': 0.9167, 'cv_margin': 1.1598}
- After: {'loo_acc': 0.8125, 'loo_f1': 0.7692, 'loo_margin': 1.2233, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.5537, 'cv_acc': 0.8333, 'cv_margin': 1.2396}
- Reason: LOO F1 regression (loo_acc +0.0000, loo_f1 -0.0308)

## iter007: Centroid head rule (no variance scaling) -> REJECTED

- Observed: WSP-006 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500, 5-fold CV 0.833, pooled leave-one-out over all 16 train+val images 0.750 (12/16), LOO F1 0.750. Decision metric is pooled LOO accuracy (rule v3). Same hypothesis as WSP-005 iter007 (Centroid head rule (no variance scaling)).
- Hypothesis: The plain class-mean difference transfers better and lifts val by >= 1 image.
- Parameters: head.weight (512), head.bias (1) [closed-form head] (513 changed)
- Before: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0927, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.171, 'cv_acc': 0.9167, 'cv_margin': 1.1598}
- After: {'loo_acc': 0.6875, 'loo_f1': 0.6154, 'loo_margin': 1.0874, 'train_acc': 0.8333, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.3193, 'cv_acc': 0.8333, 'cv_margin': 1.1309}
- Reason: LOO F1 regression (loo_acc -0.1250, loo_f1 -0.1846)

## iter008: Half-strength conv3 bias centring -> REJECTED

- Observed: WSP-006 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500, 5-fold CV 0.833, pooled leave-one-out over all 16 train+val images 0.750 (12/16), LOO F1 0.750. Decision metric is pooled LOO accuracy (rule v3). Same hypothesis as WSP-005 iter008 (Half-strength conv3 bias centring).
- Hypothesis: Half-strength centring gates only the weakest conv3 evidence without killing channels; val +>=1 image.
- Parameters: conv3.bias (32) + head refit (513) (545 changed)
- Before: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0927, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.171, 'cv_acc': 0.9167, 'cv_margin': 1.1598}
- After: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0866, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.19, 'cv_acc': 0.9167, 'cv_margin': 1.1624}
- Reason: no meaningful LOO improvement (loo_acc +0.0000, loo_f1 +0.0000)

## iter009: Full-covariance LDA head (ridge) -> REJECTED

- Observed: WSP-006 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500, 5-fold CV 0.833, pooled leave-one-out over all 16 train+val images 0.750 (12/16), LOO F1 0.750. Decision metric is pooled LOO accuracy (rule v3). Same hypothesis as WSP-005 iter009 (Full-covariance LDA head (ridge)).
- Hypothesis: Full-covariance LDA with ridge 0.05 x mean variance de-correlates them. With only 12 samples (rank <= 11) it may overfit; tested as a directional check.
- Parameters: head.weight (512), head.bias (1) [closed-form head] (513 changed)
- Before: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0927, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.171, 'cv_acc': 0.9167, 'cv_margin': 1.1598}
- After: {'loo_acc': 0.4375, 'loo_f1': 0.4, 'loo_margin': 0.6758, 'train_acc': 1.0, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': -0.0563, 'cv_acc': 0.3333, 'cv_margin': 0.5178}
- Reason: LOO F1 regression (loo_acc -0.3750, loo_f1 -0.4000)

## iter010: [exploratory] Centre/border pooled head (64 shared weights) -> REJECTED

- Observed: WSP-006 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500, 5-fold CV 0.833, pooled leave-one-out over all 16 train+val images 0.750 (12/16), LOO F1 0.750. Decision metric is pooled LOO accuracy (rule v3). Same hypothesis as WSP-005 iter010 ([exploratory] Centre/border pooled head (64 shared weights)).
- Hypothesis: Two shared weights per channel (centre cells, border cells) keep subject/background while cutting parameters; val +>=1 image.
- Parameters: head.weight (512), head.bias (1) [closed-form head] (513 changed)
- Before: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0927, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.171, 'cv_acc': 0.9167, 'cv_margin': 1.1598}
- After: {'loo_acc': 0.6875, 'loo_f1': 0.6667, 'loo_margin': 1.2915, 'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.3221, 'cv_acc': 0.9167, 'cv_margin': 1.4451}
- Reason: LOO F1 regression (loo_acc -0.1250, loo_f1 -0.1333)

## iter011: [exploratory] Remove colour-opponent filters from conv1 -> REJECTED

- Observed: WSP-006 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500, 5-fold CV 0.833, pooled leave-one-out over all 16 train+val images 0.750 (12/16), LOO F1 0.750. Decision metric is pooled LOO accuracy (rule v3). Same hypothesis as WSP-005 iter011 ([exploratory] Remove colour-opponent filters from conv1).
- Hypothesis: Zeroing conv1 filters 12-15 removes colour evidence from the pooled features; val +>=1 image if colour was noise.
- Parameters: conv1.weight[12:16] (300) + head refit (513) (788 changed)
- Before: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0927, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.171, 'cv_acc': 0.9167, 'cv_margin': 1.1598}
- After: {'loo_acc': 0.75, 'loo_f1': 0.75, 'loo_margin': 0.8846, 'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': -0.133, 'cv_acc': 0.8333, 'cv_margin': 1.0472}
- Reason: LOO F1 regression (loo_acc -0.0625, loo_f1 -0.0500)

## iter012: [exploratory] Remove centre-surround blob filters from conv1 -> REJECTED

- Observed: WSP-006 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500, 5-fold CV 0.833, pooled leave-one-out over all 16 train+val images 0.750 (12/16), LOO F1 0.750. Decision metric is pooled LOO accuracy (rule v3). Same hypothesis as WSP-005 iter012 ([exploratory] Remove centre-surround blob filters from conv1).
- Hypothesis: Removing them reduces noise features; val +>=1 image.
- Parameters: conv1.weight[8:12] (300) + head refit (513) (813 changed)
- Before: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0927, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.171, 'cv_acc': 0.9167, 'cv_margin': 1.1598}
- After: {'loo_acc': 0.8125, 'loo_f1': 0.7692, 'loo_margin': 1.0759, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.2737, 'cv_acc': 0.9167, 'cv_margin': 1.1456}
- Reason: LOO F1 regression (loo_acc +0.0000, loo_f1 -0.0308)

## iter013: [exploratory] Coarse odd-phase Gabors replace the blob filters -> REJECTED

- Observed: WSP-006 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500, 5-fold CV 0.833, pooled leave-one-out over all 16 train+val images 0.750 (12/16), LOO F1 0.750. Decision metric is pooled LOO accuracy (rule v3). Same hypothesis as WSP-005 iter013 ([exploratory] Coarse odd-phase Gabors replace the blob filters).
- Hypothesis: Odd-phase Gabors at 4 orientations replace blobs 8-11 and add silhouette-edge information; val +>=1 image.
- Parameters: conv1.weight[8:12] (300) + head refit (513) (813 changed)
- Before: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0927, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.171, 'cv_acc': 0.9167, 'cv_margin': 1.1598}
- After: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.1018, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.1616, 'cv_acc': 0.9167, 'cv_margin': 1.1654}
- Reason: no meaningful LOO improvement (loo_acc +0.0000, loo_f1 +0.0000)

## iter014: [exploratory] Remove conv2 parallel-pair detectors -> REJECTED

- Observed: WSP-006 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500, 5-fold CV 0.833, pooled leave-one-out over all 16 train+val images 0.750 (12/16), LOO F1 0.750. Decision metric is pooled LOO accuracy (rule v3). Same hypothesis as WSP-005 iter014 ([exploratory] Remove conv2 parallel-pair detectors).
- Hypothesis: Zeroing them lowers texture noise; val +>=1 image.
- Parameters: conv2.weight[16:24] (1,152) + head refit (513) (537 changed)
- Before: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0927, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.171, 'cv_acc': 0.9167, 'cv_margin': 1.1598}
- After: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.1014, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.2065, 'cv_acc': 0.9167, 'cv_margin': 1.1548}
- Reason: no meaningful LOO improvement (loo_acc +0.0000, loo_f1 +0.0000)

## iter015: [exploratory] Remove conv3 isotropic-energy channels -> REJECTED

- Observed: WSP-006 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500, 5-fold CV 0.833, pooled leave-one-out over all 16 train+val images 0.750 (12/16), LOO F1 0.750. Decision metric is pooled LOO accuracy (rule v3). Same hypothesis as WSP-005 iter015 ([exploratory] Remove conv3 isotropic-energy channels).
- Hypothesis: Zeroing conv3 24-31 removes redundant energy features; val +>=1 image.
- Parameters: conv3.weight[24:32] (2,304) + head refit (513) (1089 changed)
- Before: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0927, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.171, 'cv_acc': 0.9167, 'cv_margin': 1.1598}
- After: {'loo_acc': 0.75, 'loo_f1': 0.75, 'loo_margin': 0.9533, 'train_acc': 0.9167, 'val_acc': 0.25, 'val_f1': 0.0, 'val_margin': -0.0242, 'cv_acc': 0.9167, 'cv_margin': 1.1018}
- Reason: LOO F1 regression (loo_acc -0.0625, loo_f1 -0.0500)

## iter016: [exploratory] Centre-only head (border weight 0) -> REJECTED

- Observed: WSP-006 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500, 5-fold CV 0.833, pooled leave-one-out over all 16 train+val images 0.750 (12/16), LOO F1 0.750. Decision metric is pooled LOO accuracy (rule v3). Same hypothesis as WSP-005 iter016 ([exploratory] Centre-only head (border weight 0)).
- Hypothesis: Setting border weights to exactly 0 removes background entirely; val +>=1 image if background is a spurious cue.
- Parameters: head.weight (512 incl. 384 zeroed), head.bias (513 changed)
- Before: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0927, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.171, 'cv_acc': 0.9167, 'cv_margin': 1.1598}
- After: {'loo_acc': 0.75, 'loo_f1': 0.75, 'loo_margin': 1.2978, 'train_acc': 0.75, 'val_acc': 1.0, 'val_f1': 1.0, 'val_margin': 0.8817, 'cv_acc': 0.75, 'cv_margin': 1.286}
- Reason: LOO F1 regression (loo_acc -0.0625, loo_f1 -0.0500)

## iter017: [exploratory] Top-half emphasis in the head -> REJECTED

- Observed: WSP-006 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500, 5-fold CV 0.833, pooled leave-one-out over all 16 train+val images 0.750 (12/16), LOO F1 0.750. Decision metric is pooled LOO accuracy (rule v3). Same hypothesis as WSP-005 iter017 ([exploratory] Top-half emphasis in the head).
- Hypothesis: Weighting the top two pooled rows x1 and the bottom two x0.25 sharpens face evidence; val +>=1 image.
- Parameters: head.weight (512), head.bias [closed-form head] (513 changed)
- Before: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0927, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.171, 'cv_acc': 0.9167, 'cv_margin': 1.1598}
- After: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0899, 'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.1889, 'cv_acc': 0.8333, 'cv_margin': 1.2379}
- Reason: no meaningful LOO improvement (loo_acc +0.0000, loo_f1 +0.0000)

## iter018: [exploratory] Break conv3 filter symmetry (replication of WSP-001 iter006) -> REJECTED

- Observed: WSP-006 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500, 5-fold CV 0.833, pooled leave-one-out over all 16 train+val images 0.750 (12/16), LOO F1 0.750. Decision metric is pooled LOO accuracy (rule v3). Same hypothesis as WSP-005 iter018 ([exploratory] Break conv3 filter symmetry (replication of WSP-001 iter006)).
- Hypothesis: Eight distinct spatial kernels (vertical, horizontal, diagonals, left, right, top, bottom) give distinct layout features; val +>=1 image.
- Parameters: conv3.weight[24:32, 24:32] (576) + head refit (513) (1089 changed)
- Before: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0927, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.171, 'cv_acc': 0.9167, 'cv_margin': 1.1598}
- After: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0859, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.1644, 'cv_acc': 0.9167, 'cv_margin': 1.1523}
- Reason: no meaningful LOO improvement (loo_acc +0.0000, loo_f1 +0.0000)

## iter019: [exploratory] Control: pooled head with explicit uniform cell mask -> REJECTED

- Observed: WSP-006 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500, 5-fold CV 0.833, pooled leave-one-out over all 16 train+val images 0.750 (12/16), LOO F1 0.750. Decision metric is pooled LOO accuracy (rule v3). Same hypothesis as WSP-005 iter019 ([exploratory] Control: pooled head with explicit uniform cell mask).
- Hypothesis: A pooled head built with an explicit all-ones mask reproduces iter005 exactly (train .833, CV .833); a difference would mean a code-path bug.
- Parameters: head.weight (512), head.bias [closed-form head] (513 changed)
- Before: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0927, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.171, 'cv_acc': 0.9167, 'cv_margin': 1.1598}
- After: {'loo_acc': 0.625, 'loo_f1': 0.5714, 'loo_margin': 1.1764, 'train_acc': 0.9167, 'val_acc': 0.25, 'val_f1': 0.0, 'val_margin': -0.05, 'cv_acc': 0.9167, 'cv_margin': 1.4565}
- Reason: LOO F1 regression (loo_acc -0.1875, loo_f1 -0.2286)
