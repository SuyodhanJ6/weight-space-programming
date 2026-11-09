# Decision log

## Pre-registered acceptance rule

wsp-004, pre-registered before any edit. Accept iff (a) validation accuracy rises by >= 0.010 (1 percentage point) over the accepted checkpoint AND (b) 5-fold CV accuracy on TRAIN does not fall by more than 0.010 AND (c) validation F1 does not fall by more than 0.010. Train accuracy is recorded but never used to accept. The sealed test split is never read during iterations.

## iter000: baseline

Hand-programmed initial network. {'train_acc': 0.795, 'val_acc': 0.7843, 'val_f1': 0.7774, 'val_margin': 1.185, 'cv_acc': 0.7765, 'cv_margin': 1.1733}

## iter001: Local-peakiness channels replace the coarse box copies (48 channels) -> REJECTED

- Observed: WSP-004 baseline (hand-written 48-filter conv1, fine+coarse passthrough, full-covariance head): train 0.795, val 0.784, CV 0.777. Ablation: fine-only 0.781, coarse-only 0.778 (coarse copies add ~0.3 pt); odd Gabors most valuable (drop -> 0.768); coarse-freq Gabors, DoG, colour, hue each add <=0.4 pt.
- Hypothesis: ReLU(f - 3x3 box mean of f) per conv1 map measures how much a feature stands out from its surroundings (nonlinear, not available linearly) and raises val accuracy by >=1 point over 0.7843. (WSP-003: peaks gave +0.7 on the 16-map net.)
- Parameters: conv2.weight[48:96] (48x48x3x3 = 20,736) + head refit (1,537, lda_full) (22273 changed)
- Before: {'train_acc': 0.795, 'val_acc': 0.7843, 'val_f1': 0.7774, 'val_margin': 1.185, 'cv_acc': 0.7765, 'cv_margin': 1.1733}
- After: {'train_acc': 0.791, 'val_acc': 0.7803, 'val_f1': 0.7716, 'val_margin': 1.1704, 'cv_acc': 0.7712, 'cv_margin': 1.157}
- Reason: no meaningful val improvement (val -0.0040, cv -0.0053, val_f1 -0.0058)

## iter002: Soft-AND conjunction channels replace the coarse box copies (48 channels) -> REJECTED

- Observed: WSP-004 baseline (hand-written 48-filter conv1, fine+coarse passthrough, full-covariance head): train 0.795, val 0.784, CV 0.777. Ablation: fine-only 0.781, coarse-only 0.778 (coarse copies add ~0.3 pt); odd Gabors most valuable (drop -> 0.768); coarse-freq Gabors, DoG, colour, hue each add <=0.4 pt. WSP-003: colour x orientation conjunctions were accepted under the diagonal head.
- Hypothesis: 48 conjunction channels (4 colour axes x 8 even orientations, 8 corner pairs k/k+45deg, 8 odd-phase junction pairs k/k+90deg) add nonlinear information and raise val accuracy by >=1 point.
- Parameters: conv2.weight+bias[48:96] (20,736+48) + head refit (1,537, lda_full) (2105 changed)
- Before: {'train_acc': 0.795, 'val_acc': 0.7843, 'val_f1': 0.7774, 'val_margin': 1.185, 'cv_acc': 0.7765, 'cv_margin': 1.1733}
- After: {'train_acc': 0.7774, 'val_acc': 0.77, 'val_f1': 0.7607, 'val_margin': 1.113, 'cv_acc': 0.7646, 'cv_margin': 1.1155}
- Reason: diagnostic regression (cv -0.0119, val_f1 -0.0167)

## iter003: Coarse-frequency ODD Gabors replace the weakest group (coarse even Gabors) -> REJECTED

- Observed: WSP-004 baseline (hand-written 48-filter conv1, fine+coarse passthrough, full-covariance head): train 0.795, val 0.784, CV 0.777. Ablation: fine-only 0.781, coarse-only 0.778 (coarse copies add ~0.3 pt); odd Gabors most valuable (drop -> 0.768); coarse-freq Gabors, DoG, colour, hue each add <=0.4 pt. iter001/002 (nonlinear replacements of the coarse copies) both regressed.
- Hypothesis: Replacing those 4 slots with coarse ODD-phase Gabors (orientations 0,2,4,6, wavelength 6) adds edge information and raises val accuracy by >=1 point over 0.7843.
- Parameters: conv1.weight[44:48] (4x3x5x5 = 300) + head refit (1,537, lda_full) (1837 changed)
- Before: {'train_acc': 0.795, 'val_acc': 0.7843, 'val_f1': 0.7774, 'val_margin': 1.185, 'cv_acc': 0.7765, 'cv_margin': 1.1733}
- After: {'train_acc': 0.7985, 'val_acc': 0.7863, 'val_f1': 0.7806, 'val_margin': 1.1915, 'cv_acc': 0.7771, 'cv_margin': 1.1756}
- Reason: no meaningful val improvement (val +0.0020, cv +0.0006, val_f1 +0.0032)
