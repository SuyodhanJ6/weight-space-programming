# WSP progress

- Updated: 2026-11-09T22:04:05
- Status: complete
- Last completed: iter019
- Accepted checkpoint: `checkpoints/wsp-006/iter003.pt`
- Current metrics: {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0927, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.171, 'cv_acc': 0.9167, 'cv_margin': 1.1598}
- Baseline metrics: {'loo_acc': 0.75, 'loo_f1': 0.75, 'loo_margin': 0.8846, 'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': -0.133, 'cv_acc': 0.8333, 'cv_margin': 1.0472}
- Accepted: ['iter000', 'iter003']
- Rejected: ['iter001', 'iter002', 'iter004', 'iter005', 'iter006', 'iter007', 'iter008', 'iter009', 'iter010', 'iter011', 'iter012', 'iter013', 'iter014', 'iter015', 'iter016', 'iter017', 'iter018', 'iter019']
- Next action: write next iterations in src/iterations.py or run final

| id | title | decision | metrics |
|---|---|---|---|
| iter000 | hand-programmed baseline | baseline | {'loo_acc': 0.75, 'loo_f1': 0.75, 'loo_margin': 0.8846, 'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': -0.133, 'cv_acc': 0.8333, 'cv_margin': 1.0472} |
| iter001 | De-mean the head (replication of WSP-001 iter001 on a new split) | rejected | {'loo_acc': 0.6875, 'loo_f1': 0.7059, 'loo_margin': 0.722, 'train_acc': 0.5833, 'val_acc': 1.0, 'val_f1': 1.0, 'val_margin': 1.1126, 'cv_acc': 0.5833, 'cv_margin': 0.0824} |
| iter002 | Negative conv2 biases to restore ReLU gating | rejected | {'loo_acc': 0.75, 'loo_f1': 0.7143, 'loo_margin': 0.8817, 'train_acc': 0.9167, 'val_acc': 0.25, 'val_f1': 0.0, 'val_margin': 0.0039, 'cv_acc': 0.9167, 'cv_margin': 0.9901} |
| iter003 | Wire the dead colour pathway into conv2 | accepted | {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0927, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.171, 'cv_acc': 0.9167, 'cv_margin': 1.1598} |
| iter004 | Wider spatial pooling in conv3 (3x3 box instead of 1-2-1 blur) | rejected | {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 0.8692, 'train_acc': 0.9167, 'val_acc': 0.25, 'val_f1': 0.0, 'val_margin': 0.0305, 'cv_acc': 0.9167, 'cv_margin': 1.0526} |
| iter005 | Pooled-cell head (shared weights over the 16 pooled cells) | rejected | {'loo_acc': 0.625, 'loo_f1': 0.5714, 'loo_margin': 1.1764, 'train_acc': 0.9167, 'val_acc': 0.25, 'val_f1': 0.0, 'val_margin': -0.05, 'cv_acc': 0.9167, 'cv_margin': 1.4565} |
| iter006 | Down-weight border cells of the head (background suppression) | rejected | {'loo_acc': 0.8125, 'loo_f1': 0.7692, 'loo_margin': 1.2233, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.5537, 'cv_acc': 0.8333, 'cv_margin': 1.2396} |
| iter007 | Centroid head rule (no variance scaling) | rejected | {'loo_acc': 0.6875, 'loo_f1': 0.6154, 'loo_margin': 1.0874, 'train_acc': 0.8333, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.3193, 'cv_acc': 0.8333, 'cv_margin': 1.1309} |
| iter008 | Half-strength conv3 bias centring | rejected | {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0866, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.19, 'cv_acc': 0.9167, 'cv_margin': 1.1624} |
| iter009 | Full-covariance LDA head (ridge) | rejected | {'loo_acc': 0.4375, 'loo_f1': 0.4, 'loo_margin': 0.6758, 'train_acc': 1.0, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': -0.0563, 'cv_acc': 0.3333, 'cv_margin': 0.5178} |
| iter010 | [exploratory] Centre/border pooled head (64 shared weights) | rejected | {'loo_acc': 0.6875, 'loo_f1': 0.6667, 'loo_margin': 1.2915, 'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.3221, 'cv_acc': 0.9167, 'cv_margin': 1.4451} |
| iter011 | [exploratory] Remove colour-opponent filters from conv1 | rejected | {'loo_acc': 0.75, 'loo_f1': 0.75, 'loo_margin': 0.8846, 'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': -0.133, 'cv_acc': 0.8333, 'cv_margin': 1.0472} |
| iter012 | [exploratory] Remove centre-surround blob filters from conv1 | rejected | {'loo_acc': 0.8125, 'loo_f1': 0.7692, 'loo_margin': 1.0759, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.2737, 'cv_acc': 0.9167, 'cv_margin': 1.1456} |
| iter013 | [exploratory] Coarse odd-phase Gabors replace the blob filters | rejected | {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.1018, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.1616, 'cv_acc': 0.9167, 'cv_margin': 1.1654} |
| iter014 | [exploratory] Remove conv2 parallel-pair detectors | rejected | {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.1014, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.2065, 'cv_acc': 0.9167, 'cv_margin': 1.1548} |
| iter015 | [exploratory] Remove conv3 isotropic-energy channels | rejected | {'loo_acc': 0.75, 'loo_f1': 0.75, 'loo_margin': 0.9533, 'train_acc': 0.9167, 'val_acc': 0.25, 'val_f1': 0.0, 'val_margin': -0.0242, 'cv_acc': 0.9167, 'cv_margin': 1.1018} |
| iter016 | [exploratory] Centre-only head (border weight 0) | rejected | {'loo_acc': 0.75, 'loo_f1': 0.75, 'loo_margin': 1.2978, 'train_acc': 0.75, 'val_acc': 1.0, 'val_f1': 1.0, 'val_margin': 0.8817, 'cv_acc': 0.75, 'cv_margin': 1.286} |
| iter017 | [exploratory] Top-half emphasis in the head | rejected | {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0899, 'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.1889, 'cv_acc': 0.8333, 'cv_margin': 1.2379} |
| iter018 | [exploratory] Break conv3 filter symmetry (replication of WSP-001 iter006) | rejected | {'loo_acc': 0.8125, 'loo_f1': 0.8, 'loo_margin': 1.0859, 'train_acc': 0.9167, 'val_acc': 0.75, 'val_f1': 0.6667, 'val_margin': 0.1644, 'cv_acc': 0.9167, 'cv_margin': 1.1523} |
| iter019 | [exploratory] Control: pooled head with explicit uniform cell mask | rejected | {'loo_acc': 0.625, 'loo_f1': 0.5714, 'loo_margin': 1.1764, 'train_acc': 0.9167, 'val_acc': 0.25, 'val_f1': 0.0, 'val_margin': -0.05, 'cv_acc': 0.9167, 'cv_margin': 1.4565} |
