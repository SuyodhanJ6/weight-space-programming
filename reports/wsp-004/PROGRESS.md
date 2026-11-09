# WSP progress

- Updated: 2026-11-08T20:10:47
- Status: complete
- Last completed: iter003
- Accepted checkpoint: `checkpoints/wsp-004/iter000_initial.pt`
- Current metrics: {'train_acc': 0.795, 'val_acc': 0.7843, 'val_f1': 0.7774, 'val_margin': 1.185, 'cv_acc': 0.7765, 'cv_margin': 1.1733}
- Baseline metrics: {'train_acc': 0.795, 'val_acc': 0.7843, 'val_f1': 0.7774, 'val_margin': 1.185, 'cv_acc': 0.7765, 'cv_margin': 1.1733}
- Accepted: ['iter000']
- Rejected: ['iter001', 'iter002', 'iter003']
- Next action: write next iterations in src/iterations.py or run final

| id | title | decision | metrics |
|---|---|---|---|
| iter000 | hand-programmed baseline | baseline | {'train_acc': 0.795, 'val_acc': 0.7843, 'val_f1': 0.7774, 'val_margin': 1.185, 'cv_acc': 0.7765, 'cv_margin': 1.1733} |
| iter001 | Local-peakiness channels replace the coarse box copies (48 channels) | rejected | {'train_acc': 0.791, 'val_acc': 0.7803, 'val_f1': 0.7716, 'val_margin': 1.1704, 'cv_acc': 0.7712, 'cv_margin': 1.157} |
| iter002 | Soft-AND conjunction channels replace the coarse box copies (48 channels) | rejected | {'train_acc': 0.7774, 'val_acc': 0.77, 'val_f1': 0.7607, 'val_margin': 1.113, 'cv_acc': 0.7646, 'cv_margin': 1.1155} |
| iter003 | Coarse-frequency ODD Gabors replace the weakest group (coarse even Gabors) | rejected | {'train_acc': 0.7985, 'val_acc': 0.7863, 'val_f1': 0.7806, 'val_margin': 1.1915, 'cv_acc': 0.7771, 'cv_margin': 1.1756} |
