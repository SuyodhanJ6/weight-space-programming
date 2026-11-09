# Decision log

## Pre-registered acceptance rule

Accept iff NO regression on val accuracy AND NO regression on LOO accuracy (train+val, 16 images, head refit per fold) AND at least one of: val accuracy up by >=1 image (0.25), LOO accuracy up by >=1 image (0.0625), or (both accuracies unchanged and) val margin up by >=0.25 with LOO margin not down by >0.10. Train accuracy is recorded but never used to accept. Test split is never read during iterations.

## iter000: baseline

Hand-programmed initial network. {'train_acc': 0.9167, 'val_acc': 0.75, 'val_margin': 0.011, 'loo_acc': 0.8125, 'loo_margin': 1.0206}

## iter001: De-mean head weights (remove global-energy shortcut) -> REJECTED

- Observed: Baseline head weights nearly all positive (mean +0.119, min -0.034); logit correlates r=+0.80 with image edge density vs only +0.34 for the label.
- Hypothesis: Subtracting the mean head weight removes the direction that responds to overall feature energy, so predictions rely on which features differ, not how many fire.
- Parameters: head.weight (512), head.bias (1) (513 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.75, 'val_margin': 0.011, 'loo_acc': 0.8125, 'loo_margin': 1.0206}
- After: {'train_acc': 0.6667, 'val_acc': 0.75, 'val_margin': 0.9379, 'loo_acc': 0.625, 'loo_margin': 0.5083}
- Reason: regression (val +0.0000, loo -0.1875)

## iter002: Negative conv2 biases to restore ReLU gating -> ACCEPTED

- Observed: conv2 ReLU active on 95% of units, conv3 on 100%; network is almost linear.
- Hypothesis: Setting each conv2 bias to minus the channel's mean pre-activation makes units fire only for above-average local evidence, adding selectivity.
- Parameters: conv2.bias (32); head refit (513) (545 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.75, 'val_margin': 0.011, 'loo_acc': 0.8125, 'loo_margin': 1.0206}
- After: {'train_acc': 0.9167, 'val_acc': 0.75, 'val_margin': 0.208, 'loo_acc': 0.875, 'loo_margin': 1.0007}
- Reason: improvement (val +0.0000, loo +0.0625)

## iter003: Negative conv3 biases to restore ReLU gating -> REJECTED

- Observed: conv3 ReLU is 100% active; every conv3 channel contributes ~equally to dog_03.
- Hypothesis: Centring conv3 pre-activations gates the final features so only distinctive regions contribute to the head.
- Parameters: conv3.bias (32); head refit (513) (545 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.75, 'val_margin': 0.208, 'loo_acc': 0.875, 'loo_margin': 1.0007}
- After: {'train_acc': 0.9167, 'val_acc': 0.75, 'val_margin': 0.4272, 'loo_acc': 0.8125, 'loo_margin': 1.0768}
- Reason: regression (val +0.0000, loo -0.0625)

## iter004: Wire the dead colour pathway into conv2 -> ACCEPTED

- Observed: conv2 weight magnitude from colour channels 12-15 is exactly 0.0 (vs 28.0 from edge channels); dogs are warmer on average (R-B 27.5 vs 13.4).
- Hypothesis: Connecting the 4 colour-opponent channels to conv2 filters 24-31 lets warm/cool regional evidence reach the head.
- Parameters: conv2.weight[24:32, 12:16] (72 values); head refit (513) (585 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 0.75, 'val_margin': 0.208, 'loo_acc': 0.875, 'loo_margin': 1.0007}
- After: {'train_acc': 0.9167, 'val_acc': 1.0, 'val_margin': 0.6608, 'loo_acc': 0.875, 'loo_margin': 1.1485}
- Reason: improvement (val +0.2500, loo +0.0000)

## iter005: Down-weight border cells of the head (background suppression) -> REJECTED

- Observed: Mean |logit contribution| is 0.56 from the 4 centre cells vs 0.28 from the 12 border cells; cages/fences/grass dominate borders.
- Hypothesis: Scaling border-cell head weights by 0.25 reduces reliance on background context.
- Parameters: head.weight border-cell entries (384 of 512); head.bias (513 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 1.0, 'val_margin': 0.6608, 'loo_acc': 0.875, 'loo_margin': 1.1485}
- After: {'train_acc': 0.9167, 'val_acc': 1.0, 'val_margin': 0.948, 'loo_acc': 0.8125, 'loo_margin': 1.2275}
- Reason: regression (val +0.0000, loo -0.0625)

## iter006: Break conv3 filter symmetry (filters 24-31) -> REJECTED

- Observed: conv3 filters 24-31 produce identical outputs (max difference < 1e-6): 8 redundant channels, the head sums the same value 8 times.
- Hypothesis: Giving each filter a different spatial kernel (vertical, horizontal, diagonals, left, right, top, bottom) makes them carry distinct spatial layout information.
- Parameters: conv3.weight[24:32, 24:32] (576 values); head refit (513) (1089 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 1.0, 'val_margin': 0.6608, 'loo_acc': 0.875, 'loo_margin': 1.1485}
- After: {'train_acc': 0.9167, 'val_acc': 1.0, 'val_margin': 0.6544, 'loo_acc': 0.875, 'loo_margin': 1.1428}
- Reason: no meaningful improvement (val +0.0000, loo +0.0000, margins -0.006/-0.006)

## iter007: Amplify colour pathway x3 -> REJECTED

- Observed: iter004 fixed dog_03 (warm dog) but colour weights (1/9 each) are small next to the edge weights; LOO accuracy did not move.
- Hypothesis: Scaling the 72 colour weights by 3 raises colour influence; if colour truly separates classes, LOO and val should improve, otherwise it overfits.
- Parameters: conv2.weight[24:32,12:16] (72); head refit (513) (585 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 1.0, 'val_margin': 0.6608, 'loo_acc': 0.875, 'loo_margin': 1.1485}
- After: {'train_acc': 0.9167, 'val_acc': 1.0, 'val_margin': 0.7824, 'loo_acc': 0.8125, 'loo_margin': 1.1261}
- Reason: regression (val +0.0000, loo -0.0625)

## iter008: Half-strength conv3 bias centring -> REJECTED

- Observed: iter003 (full-strength) reduced LOO accuracy 0.875 -> 0.8125 on a model without the colour pathway.
- Hypothesis: Half-strength centring keeps some gating without killing weak features.
- Parameters: conv3.bias (32); head refit (513) (545 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 1.0, 'val_margin': 0.6608, 'loo_acc': 0.875, 'loo_margin': 1.1485}
- After: {'train_acc': 0.8333, 'val_acc': 1.0, 'val_margin': 0.7236, 'loo_acc': 0.875, 'loo_margin': 1.1304}
- Reason: no meaningful improvement (val +0.0000, loo +0.0000, margins +0.063/-0.018)

## iter009: Centroid head rule (no variance scaling) -> REJECTED

- Observed: LOO errors persist on dog_09 and dog_01 (dark/low-edge dogs).
- Hypothesis: Using the plain class-mean difference is more robust and transfers better under LOO.
- Parameters: head.weight (512), head.bias (1) (513 changed)
- Before: {'train_acc': 0.9167, 'val_acc': 1.0, 'val_margin': 0.6608, 'loo_acc': 0.875, 'loo_margin': 1.1485}
- After: {'train_acc': 0.8333, 'val_acc': 1.0, 'val_margin': 0.9686, 'loo_acc': 0.8125, 'loo_margin': 1.1167}
- Reason: regression (val +0.0000, loo -0.0625)
