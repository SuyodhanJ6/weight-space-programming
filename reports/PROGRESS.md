# WSP progress

- Updated: 2026-11-08T19:02:43
- Status: complete
- Last completed: iter009
- Accepted checkpoint: `checkpoints/iter004.pt`
- Current metrics: {'train_acc': 0.9167, 'val_acc': 1.0, 'val_margin': 0.6608, 'loo_acc': 0.875, 'loo_margin': 1.1485}
- Baseline metrics: {'train_acc': 0.9167, 'val_acc': 0.75, 'val_margin': 0.011, 'loo_acc': 0.8125, 'loo_margin': 1.0206}
- Accepted: ['iter000', 'iter002', 'iter004']
- Rejected: ['iter001', 'iter003', 'iter005', 'iter006', 'iter007', 'iter008', 'iter009']
- Next action: write next iterations in src/iterations.py or run final

| id | title | decision | metrics |
|---|---|---|---|
| iter000 | hand-programmed baseline | baseline | {'train_acc': 0.9167, 'val_acc': 0.75, 'val_margin': 0.011, 'loo_acc': 0.8125, 'loo_margin': 1.0206} |
| iter001 | De-mean head weights (remove global-energy shortcut) | rejected | {'train_acc': 0.6667, 'val_acc': 0.75, 'val_margin': 0.9379, 'loo_acc': 0.625, 'loo_margin': 0.5083} |
| iter002 | Negative conv2 biases to restore ReLU gating | accepted | {'train_acc': 0.9167, 'val_acc': 0.75, 'val_margin': 0.208, 'loo_acc': 0.875, 'loo_margin': 1.0007} |
| iter003 | Negative conv3 biases to restore ReLU gating | rejected | {'train_acc': 0.9167, 'val_acc': 0.75, 'val_margin': 0.4272, 'loo_acc': 0.8125, 'loo_margin': 1.0768} |
| iter004 | Wire the dead colour pathway into conv2 | accepted | {'train_acc': 0.9167, 'val_acc': 1.0, 'val_margin': 0.6608, 'loo_acc': 0.875, 'loo_margin': 1.1485} |
| iter005 | Down-weight border cells of the head (background suppression) | rejected | {'train_acc': 0.9167, 'val_acc': 1.0, 'val_margin': 0.948, 'loo_acc': 0.8125, 'loo_margin': 1.2275} |
| iter006 | Break conv3 filter symmetry (filters 24-31) | rejected | {'train_acc': 0.9167, 'val_acc': 1.0, 'val_margin': 0.6544, 'loo_acc': 0.875, 'loo_margin': 1.1428} |
| iter007 | Amplify colour pathway x3 | rejected | {'train_acc': 0.9167, 'val_acc': 1.0, 'val_margin': 0.7824, 'loo_acc': 0.8125, 'loo_margin': 1.1261} |
| iter008 | Half-strength conv3 bias centring | rejected | {'train_acc': 0.8333, 'val_acc': 1.0, 'val_margin': 0.7236, 'loo_acc': 0.875, 'loo_margin': 1.1304} |
| iter009 | Centroid head rule (no variance scaling) | rejected | {'train_acc': 0.8333, 'val_acc': 1.0, 'val_margin': 0.9686, 'loo_acc': 0.8125, 'loo_margin': 1.1167} |
