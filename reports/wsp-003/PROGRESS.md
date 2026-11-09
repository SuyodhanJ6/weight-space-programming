# WSP progress

- Updated: 2026-11-08T19:52:56
- Status: complete
- Last completed: iter021
- Accepted checkpoint: `checkpoints/wsp-003/iter013.pt`
- Current metrics: {'train_acc': 0.7674, 'val_acc': 0.7663, 'val_f1': 0.7587, 'val_margin': 1.1243, 'cv_acc': 0.7564, 'cv_margin': 1.1082}
- Baseline metrics: {'train_acc': 0.6021, 'val_acc': 0.6013, 'val_f1': 0.5862, 'val_margin': 0.473, 'cv_acc': 0.6014, 'cv_margin': 0.462}
- Accepted: ['iter000', 'iter001', 'iter005', 'iter009', 'iter013']
- Rejected: ['iter002', 'iter003', 'iter004', 'iter006', 'iter007', 'iter008', 'iter010', 'iter011', 'iter012', 'iter014', 'iter015', 'iter016', 'iter017', 'iter018', 'iter019', 'iter020', 'iter021']
- Next action: write next iterations in src/iterations.py or run final

| id | title | decision | metrics |
|---|---|---|---|
| iter000 | hand-programmed baseline | baseline | {'train_acc': 0.6021, 'val_acc': 0.6013, 'val_f1': 0.5862, 'val_margin': 0.473, 'cv_acc': 0.6014, 'cv_margin': 0.462} |
| iter001 | Wire the colour pathway into conv2 (replication at 8x data) | accepted | {'train_acc': 0.6204, 'val_acc': 0.6243, 'val_f1': 0.6094, 'val_margin': 0.5641, 'cv_acc': 0.6192, 'cv_margin': 0.5373} |
| iter002 | Multi-scale passthrough: 16 fine + 16 coarse conv1 maps (replication at 8x data) | rejected | {'train_acc': 0.6285, 'val_acc': 0.6263, 'val_f1': 0.6141, 'val_margin': 0.5853, 'cv_acc': 0.6286, 'cv_margin': 0.5697} |
| iter003 | Three-scale pyramid: fine blur + mid + coarse box | rejected | {'train_acc': 0.6288, 'val_acc': 0.626, 'val_f1': 0.6134, 'val_margin': 0.5845, 'cv_acc': 0.6281, 'cv_margin': 0.5689} |
| iter004 | Colour x texture conjunction detectors (thresholded soft-AND) | rejected | {'train_acc': 0.6305, 'val_acc': 0.6307, 'val_f1': 0.6112, 'val_margin': 0.6053, 'cv_acc': 0.6292, 'cv_margin': 0.586} |
| iter005 | Fine conjunctions: every colour x every single orientation (32 channels) | accepted | {'train_acc': 0.6288, 'val_acc': 0.6357, 'val_f1': 0.6064, 'val_margin': 0.5789, 'cv_acc': 0.6279, 'cv_margin': 0.5619} |
| iter006 | Hue-bin and colour x blob conjunctions replace the weakest-orientation slots | rejected | {'train_acc': 0.6211, 'val_acc': 0.6327, 'val_f1': 0.6001, 'val_margin': 0.5574, 'cv_acc': 0.6221, 'cv_margin': 0.5418} |
| iter007 | Neighbourhood-tolerant conjunctions (colour at centre AND orientation over 3x3) | rejected | {'train_acc': 0.6189, 'val_acc': 0.6233, 'val_f1': 0.5796, 'val_margin': 0.5086, 'cv_acc': 0.6171, 'cv_margin': 0.4964} |
| iter008 | Higher AND threshold (-2.0 -> -3.0): rarer, stronger co-occurrences | rejected | {'train_acc': 0.6308, 'val_acc': 0.6333, 'val_f1': 0.5974, 'val_margin': 0.5458, 'cv_acc': 0.6288, 'cv_margin': 0.5435} |
| iter009 | Full-covariance LDA head (closed form) [data-fitted head, not a hand edit] | accepted | {'train_acc': 0.7515, 'val_acc': 0.7443, 'val_f1': 0.7338, 'val_margin': 1.039, 'cv_acc': 0.7418, 'cv_margin': 1.0472} |
| iter010 | [full-LDA head] 16 marginal passthrough + 16 grouped conjunction channels | rejected | {'train_acc': 0.7272, 'val_acc': 0.7177, 'val_f1': 0.7082, 'val_margin': 0.9427, 'cv_acc': 0.7177, 'cv_margin': 0.9528} |
| iter011 | [full-LDA head] Multi-scale passthrough (16 fine + 16 coarse conv1 maps) | rejected | {'train_acc': 0.5152, 'val_acc': 0.5043, 'val_f1': 0.6601, 'val_margin': 0.1565, 'cv_acc': 0.5069, 'cv_margin': 0.0724} |
| iter012 | [full-LDA head] Three-scale pyramid with a coarse box scale | rejected | {'train_acc': 0.5151, 'val_acc': 0.5043, 'val_f1': 0.6603, 'val_margin': 0.1553, 'cv_acc': 0.5072, 'cv_margin': 0.0755} |
| iter013 | [full-LDA head] Multi-scale passthrough with conv2.bias reset (valid re-run of iter011) | accepted | {'train_acc': 0.7674, 'val_acc': 0.7663, 'val_f1': 0.7587, 'val_margin': 1.1243, 'cv_acc': 0.7564, 'cv_margin': 1.1082} |
| iter014 | [full-LDA head] Three-scale pyramid with conv2.bias reset (valid re-run of iter012) | rejected | {'train_acc': 0.7664, 'val_acc': 0.7677, 'val_f1': 0.7604, 'val_margin': 1.125, 'cv_acc': 0.7566, 'cv_margin': 1.1084} |
| iter015 | [full-LDA head] Two spatial-frequency Gabor bands (odd orientations -> wavelength 6) | rejected | {'train_acc': 0.766, 'val_acc': 0.7673, 'val_f1': 0.7603, 'val_margin': 1.1215, 'cv_acc': 0.758, 'cv_margin': 1.1081} |
| iter016 | [full-LDA head] Local-peakiness channels replace the coarse marginal copies | rejected | {'train_acc': 0.7738, 'val_acc': 0.7733, 'val_f1': 0.7645, 'val_margin': 1.1485, 'cv_acc': 0.7641, 'cv_margin': 1.1209} |
| iter017 | [full-LDA head] Diagonal hue sectors replace the DoG blob filters | rejected | {'train_acc': 0.7624, 'val_acc': 0.7597, 'val_f1': 0.7508, 'val_margin': 1.0902, 'cv_acc': 0.751, 'cv_margin': 1.0811} |
| iter018 | [full-LDA head] Odd-phase Gabors replace the DoG blob filters | rejected | {'train_acc': 0.7724, 'val_acc': 0.7733, 'val_f1': 0.7662, 'val_margin': 1.139, 'cv_acc': 0.7622, 'cv_margin': 1.1192} |
| iter019 | [full-LDA head] Combine local-peakiness channels (iter016) and odd-phase Gabors (iter018) | rejected | {'train_acc': 0.7772, 'val_acc': 0.776, 'val_f1': 0.7689, 'val_margin': 1.1615, 'cv_acc': 0.7634, 'cv_margin': 1.1337} |
| iter020 | [round 2] Dark-bar polarity: negated Gabors replace the DoG blob filters | rejected | {'train_acc': 0.7676, 'val_acc': 0.763, 'val_f1': 0.7549, 'val_margin': 1.1137, 'cv_acc': 0.758, 'cv_margin': 1.0967} |
| iter021 | [round 2] Spend all channels on the 10 best conv1 maps x {fine, coarse, peak} | rejected | {'train_acc': 0.7672, 'val_acc': 0.7657, 'val_f1': 0.756, 'val_margin': 1.1201, 'cv_acc': 0.7578, 'cv_margin': 1.1048} |
