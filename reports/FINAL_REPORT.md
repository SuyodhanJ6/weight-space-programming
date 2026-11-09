# Weight-Space Programming: Final Report

## 1. Research objective
Test whether an LLM agent can build and improve a cat-vs-dog CNN by reasoning about and directly
editing its weights, with no gradient-based training. The result is measured, not assumed.

## 2. Dataset
20 images (10 cat, 10 dog) from Microsoft Cats vs. Dogs, all RGB JPEG, 0 corrupted, 0 exact or near
duplicates (`reports/DATASET_INSPECTION.md`). Fixed split (seed 0): **train 12, val 4, test 4**.
No images were added. The full download was deleted at the user's request.

## 3. Architecture and rationale
`CatDogNet`, 15k parameters: conv5x5(3→16)-pool-conv3x3(16→32)-pool-conv3x3(32→32)-avgpool(4x4)-
linear(512→1). Small enough to inspect every weight group; convolution shares weights so one
hand-written filter acts everywhere; a 4x4 pooled grid keeps coarse layout. PyTorch is used only as
a tensor library: autograd disabled globally, parameters `requires_grad=False`.

## 4. Initial parameter strategy (iter000)
All weights zeroed, then programmed: conv1 = analytic oriented Gabor (8), centre-surround (4) and
colour-opponent (4) filters; conv2 = hand-composed line / corner / parallel-pair / blob+edge
detectors; conv3 = centre-weighted combinations of those groups; head = closed-form class-mean
linear discriminant from the 12 train images (one shot, no iteration). All biases except the head = 0.

## 5. Baseline
train 0.917, val 0.75, LOO (16 imgs) 0.8125.

## 6. Methodology
Observe (inspect weights/activations/errors) → hypothesis → explicit edit via `WeightProgrammer`
→ validate (shapes, NaN/inf, no autograd) → evaluate → accept or roll back by the pre-registered
rule (`experiments/decision_log.md`). Metrics: accuracy, precision, recall, F1, confusion, BCE
(metric only), margins, and a leave-one-out (LOO) diagnostic over train+val (head refit on 15,
predict 1). Train accuracy never drives acceptance. Resumable state in `experiments/`.

## 7. Iterations performed: 9 (2 accepted, 7 rejected)
| id | hypothesis (short) | params changed | val | LOO | decision | reason |
|---|---|---|---|---|---|---|
| iter001 | De-mean head weights (remove global-energy shortcut) | 513 | 0.75→0.75 | 0.8125→0.6250 | REJECTED | regression (val +0.0000, loo -0.1875) |
| iter002 | Negative conv2 biases to restore ReLU gating | 545 | 0.75→0.75 | 0.8125→0.8750 | ACCEPTED | improvement (val +0.0000, loo +0.0625) |
| iter003 | Negative conv3 biases to restore ReLU gating | 545 | 0.75→0.75 | 0.8750→0.8125 | REJECTED | regression (val +0.0000, loo -0.0625) |
| iter004 | Wire the dead colour pathway into conv2 | 585 | 0.75→1.00 | 0.8750→0.8750 | ACCEPTED | improvement (val +0.2500, loo +0.0000) |
| iter005 | Down-weight border cells of the head (background suppression) | 513 | 1.00→1.00 | 0.8750→0.8125 | REJECTED | regression (val +0.0000, loo -0.0625) |
| iter006 | Break conv3 filter symmetry (filters 24-31) | 1089 | 1.00→1.00 | 0.8750→0.8750 | REJECTED | no meaningful improvement (val +0.0000, loo +0.0000, margins -0.006/-0.006) |
| iter007 | Amplify colour pathway x3 | 585 | 1.00→1.00 | 0.8750→0.8125 | REJECTED | regression (val +0.0000, loo -0.0625) |
| iter008 | Half-strength conv3 bias centring | 545 | 1.00→1.00 | 0.8750→0.8750 | REJECTED | no meaningful improvement (val +0.0000, loo +0.0000, margins +0.063/-0.018) |
| iter009 | Centroid head rule (no variance scaling) | 513 | 1.00→1.00 | 0.8750→0.8125 | REJECTED | regression (val +0.0000, loo -0.0625) |

## 8. Key hypotheses and findings
- The baseline logit correlated r=+0.80 with image edge density (label vs edge density only +0.34):
  it was largely an edge-density meter.
- conv2 received exactly 0.0 weight from the 4 colour channels (28.0 from edge channels): the
  network was colour-blind. conv3 filters 24-31 were identical (8 redundant channels). ReLUs were
  almost never gating (conv2 95%, conv3 100% active).

## 9. Examples of direct weight modifications
- iter002: `conv2.bias[0:32] = -mean(pre-activation over train images)` (32 values) + head refit.
- iter004: `conv2.weight[24+k, 12+k%4, :, :] = 1/9` for k=0..7 (72 values) wiring colour into conv2.
- iter007: `conv2.weight[24:32, 12:16] *= 3` (transform). Full old/new values:
  `experiments/parameter_changes.json`, `experiments/param_changes/iterNNN.pt`.

## 10. Before/after (accepted path)
| | train | val | LOO | val margin |
|---|---|---|---|---|
| iter000 baseline | 0.917 | 0.75 | 0.8125 | 0.011 |
| iter004 final | 0.917 | 1.00 | 0.8750 | 0.661 |

## 11. Accepted / rejected
Accepted: iter002 (conv2 bias gating; LOO 13/16→14/16), iter004 (colour pathway; val 3/4→4/4 via one
image, dog_03). Rejected: iter001 (de-mean head), 003 (conv3 bias), 005 (centre mask), 006
(break conv3 symmetry), 007 (x3 colour), 008 (half conv3 bias), 009 (centroid head). Several
regressed LOO by one image; none was kept because of train accuracy.

## 12. Final model (`checkpoints/iter004.pt`) on the locked test split (read once)
| | accuracy | precision | recall | F1 | confusion (tp/fp/tn/fn) |
|---|---|---|---|---|---|
| initial (iter000) | 0.50 | 0.50 | 0.50 | 0.50 | 1/1/1/1 |
| final (iter004) | 0.75 | 0.67 | 1.00 | 0.80 | 2/1/1/0 |

Test n=4: 2/4 → 3/4 is a one-image difference (dog_06). `cat_07` is wrong in both (initial logit +1.73, final +0.36).
This is not statistically meaningful (any result on 4 images has a 95% interval of roughly 20-99%).

## 13. Audit
Own audit (`src/weight_audit.py`, passed; also caught deliberately planted violations): AST scan found no
optimizer/backward/autograd/network/pretrained use; runtime: grad mode off, no optimizer objects,
no param/activation in a graph; scoring read only train+val; every iteration reproduces exactly from
parent checkpoint + recorded diff; baseline rebuilds bit-exact from code; rejected experiments kept.
Independent sub-agent audit: no violations; concerns listed in §14.

## 14. Limitations (honest)
- **Not "pure reasoning".** Biases and every head refit are closed-form statistics of the 12 train
  images (LDA/centroid), and my edits were chosen by looking at per-image errors. This is "hand edits
  plus closed-form data fits, no gradient descent", not weights set from reasoning alone.
- **Selection leakage / tiny sample.** 9 decisions were made against the same 4 val and 16 LOO images,
  so val/LOO are optimistically biased and 1-image granular. The test set has 4 images: low power.
- **iter004's evidence is one image.** LOO did not change for it.
- Statistics from all 20 images (including the 4 test images) were summarised in the dataset report
  and cited in iter004's rationale; dev-only numbers are identical, and no test-model output was used.
- No generalization claim is supported. The 20 images were all seen visually before design.
- Runtime network block is a monkeypatch, not a sandbox. Internet was used earlier, with approval, only
  to install packages and download the dataset (before the offline experiment phase).

## 15. Does the experiment support the WSP hypothesis?
**Weakly and only partially.** Supported: the agent found real, verifiable structural defects by
inspection (edge-density shortcut, dead colour pathway, duplicate filters, non-gating ReLUs), and
made documented, reproducible edits. Controls (`experiments/controls.json`): 40 random-conv-feature
nets with the same head rule scored LOO 0.64 (max 0.75); none matched the hand-programmed
baseline (0%) or final (0%); a label-shuffle null reached the final LOO in
0.5% of runs. So the hand-built extractor carries some signal on the dev images beyond chance
and random features. Not supported: any claim of real cat-vs-dog generalization, or that the
improvement came from the reasoning edits rather than from fitting and selection on tiny data. Honest summary:
WSP produced an auditable, modest, statistically unproven improvement.

## 16. Recommendations
1. Use a larger held-out evaluation set (hundreds of images) to measure generalization and cut selection bias.
2. Separate "hand edits" from "closed-form fits" in the accounting, and test hand-only heads.
3. Pre-register a fixed number of iterations and a single final test.
4. Add richer hand-built features (muzzle/ear shape cues) and compare to random-feature baselines with confidence intervals.

## Reproducibility
`README.md` lists commands. Resume: `uv run python src/experiment_runner.py run` (reads
`experiments/experiment_state.json`; rolls back interrupted work; skips completed iterations).
Artifacts: `checkpoints/iter000_initial.pt` (initial), `checkpoints/iter004.pt` (final),
`experiments/experiment_history.json`, `parameter_changes.json`, `decision_log.md`,
`final_test_results.json`, `controls.json`, `audit_report.json`, `logs/execution.log`.
