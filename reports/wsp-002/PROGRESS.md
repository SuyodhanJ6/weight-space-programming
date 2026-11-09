# WSP progress

- Updated: 2026-11-08T19:30:47
- Status: complete
- Last completed: iter027
- Accepted checkpoint: `checkpoints/wsp-002/iter004.pt`
- Current metrics: {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954}
- Baseline metrics: {'train_acc': 0.622, 'val_acc': 0.5717, 'val_f1': 0.5403, 'val_margin': 0.398, 'cv_acc': 0.618, 'cv_margin': 0.5081}
- Accepted: ['iter000', 'iter004']
- Rejected: ['iter001', 'iter002', 'iter003', 'iter005', 'iter006', 'iter007', 'iter008', 'iter009', 'iter010', 'iter011', 'iter012', 'iter013', 'iter014', 'iter015', 'iter016', 'iter017', 'iter018', 'iter019', 'iter020', 'iter021', 'iter022', 'iter023', 'iter024', 'iter025', 'iter026', 'iter027']
- Next action: write next iterations in src/iterations.py or run final

| id | title | decision | metrics |
|---|---|---|---|
| iter000 | hand-programmed baseline | baseline | {'train_acc': 0.622, 'val_acc': 0.5717, 'val_f1': 0.5403, 'val_margin': 0.398, 'cv_acc': 0.618, 'cv_margin': 0.5081} |
| iter001 | Remove global-energy direction from the head (replication at scale) | rejected | {'train_acc': 0.593, 'val_acc': 0.5633, 'val_f1': 0.5619, 'val_margin': 0.3687, 'cv_acc': 0.586, 'cv_margin': 0.3878} |
| iter002 | Centre conv2 pre-activations with negative biases (replication) | rejected | {'train_acc': 0.613, 'val_acc': 0.5783, 'val_f1': 0.5358, 'val_margin': 0.3726, 'cv_acc': 0.613, 'cv_margin': 0.4708} |
| iter003 | Centre conv3 pre-activations with negative biases (replication) | rejected | {'train_acc': 0.594, 'val_acc': 0.575, 'val_f1': 0.5216, 'val_margin': 0.3504, 'cv_acc': 0.594, 'cv_margin': 0.4193} |
| iter004 | Wire the dead colour pathway into conv2 (replication) | accepted | {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3894, 'cv_acc': 0.632, 'cv_margin': 0.5954} |
| iter005 | Suppress the bottom pooled row in the head | rejected | {'train_acc': 0.636, 'val_acc': 0.6, 'val_f1': 0.5862, 'val_margin': 0.4039, 'cv_acc': 0.633, 'cv_margin': 0.6116} |
| iter006 | Identity routing for conv3 filters 24-31 (de-duplicate colour/blob features) | rejected | {'train_acc': 0.631, 'val_acc': 0.605, 'val_f1': 0.5864, 'val_margin': 0.4315, 'cv_acc': 0.632, 'cv_margin': 0.5995} |
| iter007 | Amplify colour pathway x3 (replication of WSP-001 iter007) | rejected | {'train_acc': 0.637, 'val_acc': 0.6, 'val_f1': 0.5789, 'val_margin': 0.3829, 'cv_acc': 0.633, 'cv_margin': 0.5889} |
| iter008 | Wire colour into conv2 corner filters 8-15 | rejected | {'train_acc': 0.628, 'val_acc': 0.5917, 'val_f1': 0.5601, 'val_margin': 0.3527, 'cv_acc': 0.624, 'cv_margin': 0.5682} |
| iter009 | Coarser oriented edge filters in conv1 (wavelength 4 -> 6 px) | rejected | {'train_acc': 0.635, 'val_acc': 0.5983, 'val_f1': 0.5852, 'val_margin': 0.3733, 'cv_acc': 0.631, 'cv_margin': 0.5916} |
| iter010 | Centroid head rule (no per-feature variance scaling) | rejected | {'train_acc': 0.61, 'val_acc': 0.5867, 'val_f1': 0.554, 'val_margin': 0.2861, 'cv_acc': 0.612, 'cv_margin': 0.5203} |
| iter011 | Scale border-cell head weights by 0.25 | rejected | {'train_acc': 0.634, 'val_acc': 0.5883, 'val_f1': 0.5704, 'val_margin': 0.3919, 'cv_acc': 0.636, 'cv_margin': 0.6162} |
| iter012 | Prune the 8 weakest conv3 channels | rejected | {'train_acc': 0.636, 'val_acc': 0.5967, 'val_f1': 0.5769, 'val_margin': 0.3683, 'cv_acc': 0.634, 'cv_margin': 0.5927} |
| iter013 | Average colour over a larger area (Gaussian sigma 1.5 -> 5x5 box) | rejected | {'train_acc': 0.641, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.3864, 'cv_acc': 0.632, 'cv_margin': 0.5931} |
| iter014 | [round 2] Wire colour into conv2 parallel-pair filters 16-23 | rejected | {'train_acc': 0.625, 'val_acc': 0.5933, 'val_f1': 0.5627, 'val_margin': 0.3516, 'cv_acc': 0.623, 'cv_margin': 0.5668} |
| iter015 | [round 2] Half-strength conv2 bias gating on the accepted model | rejected | {'train_acc': 0.638, 'val_acc': 0.6033, 'val_f1': 0.5825, 'val_margin': 0.3841, 'cv_acc': 0.632, 'cv_margin': 0.5864} |
| iter016 | [round 2] Remove oriented-edge inputs from the colour/blob filters | rejected | {'train_acc': 0.636, 'val_acc': 0.6033, 'val_f1': 0.5868, 'val_margin': 0.3867, 'cv_acc': 0.633, 'cv_margin': 0.5936} |
| iter017 | [round 3] Bright blob-pair detectors (eyes) in conv3 filters 25-28 | rejected | {'train_acc': 0.636, 'val_acc': 0.6017, 'val_f1': 0.5872, 'val_margin': 0.3834, 'cv_acc': 0.631, 'cv_margin': 0.6009} |
| iter018 | [round 3] Dark blob-pair detectors (nostril/dark eyes) in conv3 filters 29-31 | rejected | {'train_acc': 0.633, 'val_acc': 0.6133, 'val_f1': 0.5958, 'val_margin': 0.4223, 'cv_acc': 0.631, 'cv_margin': 0.6017} |
| iter019 | [round 3] Larger centre-surround blobs in conv1 (sigma 0.8 -> 1.5) | rejected | {'train_acc': 0.64, 'val_acc': 0.6067, 'val_f1': 0.5889, 'val_margin': 0.388, 'cv_acc': 0.634, 'cv_margin': 0.5955} |
| iter020 | [round 4] Odd-phase (sine) Gabors in conv1: step-edge detectors | rejected | {'train_acc': 0.627, 'val_acc': 0.6, 'val_f1': 0.5918, 'val_margin': 0.3797, 'cv_acc': 0.624, 'cv_margin': 0.5835} |
| iter021 | [round 4] Sharper orientation tuning in conv2 line filters | rejected | {'train_acc': 0.635, 'val_acc': 0.6017, 'val_f1': 0.5829, 'val_margin': 0.3827, 'cv_acc': 0.634, 'cv_margin': 0.5928} |
| iter022 | [round 4] Half-strength conv3 bias gating on the accepted model | rejected | {'train_acc': 0.634, 'val_acc': 0.5933, 'val_f1': 0.5704, 'val_margin': 0.3801, 'cv_acc': 0.628, 'cv_margin': 0.5742} |
| iter023 | [round 4] Hierarchy ablation: pass conv1 features straight through conv2/conv3 | rejected | {'train_acc': 0.645, 'val_acc': 0.5983, 'val_f1': 0.5689, 'val_margin': 0.4593, 'cv_acc': 0.638, 'cv_margin': 0.6015} |
| iter024 | [round 5] Multi-scale passthrough: 16 fine + 16 coarse conv1 feature maps | rejected | {'train_acc': 0.647, 'val_acc': 0.605, 'val_f1': 0.5775, 'val_margin': 0.4562, 'cv_acc': 0.64, 'cv_margin': 0.6075} |
| iter025 | [round 6] Spatially shared head (32 effective weights) | rejected | {'train_acc': 0.625, 'val_acc': 0.5917, 'val_f1': 0.5739, 'val_margin': 0.3455, 'cv_acc': 0.629, 'cv_margin': 0.5521} |
| iter026 | [round 6] Centre/border shared head (64 effective weights) | rejected | {'train_acc': 0.631, 'val_acc': 0.5933, 'val_f1': 0.5779, 'val_margin': 0.3785, 'cv_acc': 0.632, 'cv_margin': 0.6021} |
| iter027 | [round 6] Luminance-distribution (soft histogram) ramps replace the DoG blobs | rejected | {'train_acc': 0.608, 'val_acc': 0.5917, 'val_f1': 0.5724, 'val_margin': 0.3653, 'cv_acc': 0.606, 'cv_margin': 0.5589} |
