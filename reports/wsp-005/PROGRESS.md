# WSP progress

- Updated: 2026-11-09T21:32:43
- Status: complete
- Last completed: iter019
- Accepted checkpoint: `checkpoints/wsp-005/iter000_initial.pt`
- Current metrics: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- Baseline metrics: {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011}
- Accepted: ['iter000']
- Rejected: ['iter001', 'iter002', 'iter003', 'iter004', 'iter005', 'iter006', 'iter007', 'iter008', 'iter009', 'iter010', 'iter011', 'iter012', 'iter013', 'iter014', 'iter015', 'iter016', 'iter017', 'iter018', 'iter019']
- Next action: write next iterations in src/iterations.py or run final

| id | title | decision | metrics |
|---|---|---|---|
| iter000 | hand-programmed baseline | baseline | {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011} |
| iter001 | De-mean the head (replication of WSP-001 iter001 on a new split) | rejected | {'train_acc': 0.6667, 'val_acc': 0.75, 'val_f1': 0.8, 'val_margin': 0.4591, 'cv_acc': 0.5, 'cv_margin': 0.4311} |
| iter002 | Negative conv2 biases to restore ReLU gating | rejected | {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.1335, 'cv_acc': 0.9167, 'cv_margin': 1.2856} |
| iter003 | Wire the dead colour pathway into conv2 | rejected | {'train_acc': 1.0, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 1.0357, 'cv_acc': 0.9167, 'cv_margin': 1.278} |
| iter004 | Wider spatial pooling in conv3 (3x3 box instead of 1-2-1 blur) | rejected | {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2634, 'cv_acc': 0.9167, 'cv_margin': 1.2911} |
| iter005 | Pooled-cell head (shared weights over the 16 pooled cells) | rejected | {'train_acc': 0.8333, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.1352, 'cv_acc': 0.8333, 'cv_margin': 1.3793} |
| iter006 | Down-weight border cells of the head (background suppression) | rejected | {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.3838, 'cv_acc': 0.9167, 'cv_margin': 1.3648} |
| iter007 | Centroid head rule (no variance scaling) | rejected | {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2717, 'cv_acc': 0.9167, 'cv_margin': 1.1265} |
| iter008 | Half-strength conv3 bias centring | rejected | {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.1843, 'cv_acc': 0.9167, 'cv_margin': 1.2761} |
| iter009 | Full-covariance LDA head (ridge) | rejected | {'train_acc': 1.0, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.4502, 'cv_acc': 0.6667, 'cv_margin': 1.3113} |
| iter010 | [exploratory] Centre/border pooled head (64 shared weights) | rejected | {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2436, 'cv_acc': 0.8333, 'cv_margin': 1.416} |
| iter011 | [exploratory] Remove colour-opponent filters from conv1 | rejected | {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2796, 'cv_acc': 0.9167, 'cv_margin': 1.3011} |
| iter012 | [exploratory] Remove centre-surround blob filters from conv1 | rejected | {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2492, 'cv_acc': 0.9167, 'cv_margin': 1.2856} |
| iter013 | [exploratory] Coarse odd-phase Gabors replace the blob filters | rejected | {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.293, 'cv_acc': 0.9167, 'cv_margin': 1.2472} |
| iter014 | [exploratory] Remove conv2 parallel-pair detectors | rejected | {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2913, 'cv_acc': 0.9167, 'cv_margin': 1.2839} |
| iter015 | [exploratory] Remove conv3 isotropic-energy channels | rejected | {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2599, 'cv_acc': 0.9167, 'cv_margin': 1.3025} |
| iter016 | [exploratory] Centre-only head (border weight 0) | rejected | {'train_acc': 0.8333, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.4453, 'cv_acc': 0.8333, 'cv_margin': 1.3979} |
| iter017 | [exploratory] Top-half emphasis in the head | rejected | {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.269, 'cv_acc': 0.8333, 'cv_margin': 1.19} |
| iter018 | [exploratory] Break conv3 filter symmetry (replication of WSP-001 iter006) | rejected | {'train_acc': 0.9167, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.2754, 'cv_acc': 0.9167, 'cv_margin': 1.3019} |
| iter019 | [exploratory] Control: pooled head with explicit uniform cell mask | rejected | {'train_acc': 0.8333, 'val_acc': 0.5, 'val_f1': 0.5, 'val_margin': 0.1352, 'cv_acc': 0.8333, 'cv_margin': 1.3793} |
