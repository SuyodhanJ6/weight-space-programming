# WSP-002 Final Report

## 1. Result in one paragraph
A hand-programmed CNN with a closed-form head reaches about **62% on a sealed 2,000-image test set** (95% CI
60-64%), clearly above chance but far from 100%. Across 13 reasoned weight edits only one was accepted
(wiring colour into conv2). Its +3.5-point validation gain **did not replicate on the sealed test set**
(62.2% → 61.6%, paired McNemar p=0.59). The experiment therefore shows that direct weight
programming can make a network that beats chance, but gives **no evidence that the LLM's edits improved
it**, and the hand-built features were not better than random convolution features on validation.

## 2. Preservation of WSP-001
WSP-001 artifacts (experiments/, checkpoints/*.pt, reports/*.md, logs/, splits) were hashed and made
read-only before any WSP-002 work; the audit re-verifies them (`wsp001_artifacts_changed = []`).
Code was generalised through `src/paths.py` (WSP_EXPERIMENT env var, default wsp-001); wsp-001's own audit still passes.

## 3. Data (offline except one approved download)
**Exception:** `CLAUDE.md` forbids downloads; the user explicitly approved ONE re-download of the official
Microsoft Cats vs. Dogs zip (sha256 `b7974bd00a84a999…`) because no local dataset existed (the earlier copy had been deleted
at the user's request). Everything afterwards ran offline; the zip and extracted tree were deleted.
- Validation per image: JPEG, decodable, ≥100 px, ≥1 KB. Rejects (cat/dog): {'not_jpeg': 94, 'too_small_image': 78, 'exact_duplicate_or_wsp001': 27, 'too_small_file': 3} / {'exact_duplicate_or_wsp001': 19, 'too_small_image': 80, 'not_jpeg': 135, 'too_small_file': 2}.
  Near-duplicates removed (dHash Hamming ≤4): {'cat': 35, 'dog': 27}. The 20 WSP-001 images were excluded.
- Fixed seed 2, content-independent sampling, class balanced: **train 1,000 (500+500), validation 600 (300+300),
  sealed test 2,000 (1,000+1,000)**. Originals never modified; train/val files read-only, hashes verified by the audit.
- Content statistics were computed for train/val only. Both classes are diverse (sizes 105-500 px, brightness
  range ~26-221). Cats vs dogs: edge density 10.2 vs 11.5, warmth (R-B) 19.7 vs 17.6, brightness 116 vs 121: only small differences.
- **Sealing:** test images in `dataset/wsp-002/sealed_test/` (mode 000), no cache, loader raises PermissionError until
  `final` unlocks it. No test image or prediction was inspected during editing.

## 4. Architecture and initialisation
`CatDogNet` (unchanged from WSP-001): conv5x5(3→16)-pool-conv3x3(16→32)-pool-conv3x3(32→32)-avgpool 4x4-linear(512→1),
~15k parameters, autograd disabled, no pretrained weights. Initial weights: analytic Gabor/centre-surround/colour-opponent
conv1; hand-composed line/corner/parallel/blob conv2; conv3 combinations; head fitted in closed form on the train split.

## 5. Training results (train split, 1,000 images; used for reasoning and head fitting; NOT evidence of generalisation)
Baseline train accuracy 0.622; final 0.641. 5-fold CV on train (head refit per fold): 0.618 → 0.632.

## 6. Validation results (600 images; used to accept/reject; biased by selection)
| | accuracy | 95% CI | F1 |
|---|---|---|---|
| baseline | 0.5717 | [0.5317, 0.6107] | 0.540 |
| final (iter004) | 0.6067 | [0.567, 0.645] | 0.589 |

Paired: 73 images fixed vs 52 broken, McNemar p=0.073 (not significant, and optimistic due to 13 looks at the same validation set).

## 7. One-time sealed test results (2,000 images, evaluated once)
| model | accuracy | 95% CI | precision | recall | F1 | confusion tp/fp/tn/fn |
|---|---|---|---|---|---|---|
| initial (iter000) | 0.6220 | [0.6005, 0.643] | 0.630 | 0.592 | 0.610 | 592/348/652/408 |
| final (iter004) | 0.6160 | [0.5945, 0.6371] | 0.618 | 0.606 | 0.612 | 606/374/626/394 |

Paired: final right/initial wrong = 200, initial right/final wrong = 212, McNemar p = 0.588.
**The accepted edit did not improve unseen-image accuracy.** The initial model scored higher on test (62.2%) than on validation (57.2%): with
n=600 the validation estimate itself carried about ±4 points of noise. Both models are statistically indistinguishable from each other.
The goal of 100% was not reached and is not reachable with this method and data; iteration continued through the full planned set regardless.

## 8. Edits: accepted vs rejected (13 iterations; 1 accepted, 12 rejected)
Acceptance rule (pre-registered): accept iff val accuracy +≥1 pt AND train 5-fold CV not down >1 pt AND val F1 not down >1 pt.
Columns count changed parameters by origin: **hand** = explicit wiring/kernel values written by the agent; **derived** = constants computed
from train statistics (bias calibration, channel pruning choice); **fit** = head weights re-derived by the one-shot closed-form discriminant on train.
| id | hypothesis | hand | derived | fit | val acc | CV acc | val F1 | decision |
|---|---|---|---|---|---|---|---|---|
| iter001 | Remove global-energy direction from the head (replication at scale) | 0 | 0 | 513 | 0.5717→0.5633 | 0.618→0.586 | 0.540→0.562 | **REJECTED** |
| iter002 | Centre conv2 pre-activations with negative biases (replication) | 0 | 32 | 513 | 0.5717→0.5783 | 0.618→0.613 | 0.540→0.536 | **REJECTED** |
| iter003 | Centre conv3 pre-activations with negative biases (replication) | 0 | 32 | 513 | 0.5717→0.5750 | 0.618→0.594 | 0.540→0.522 | **REJECTED** |
| iter004 | Wire the dead colour pathway into conv2 (replication) | 72 | 0 | 513 | 0.5717→0.6067 | 0.618→0.632 | 0.540→0.589 | **ACCEPTED** |
| iter005 | Suppress the bottom pooled row in the head | 0 | 0 | 513 | 0.6067→0.6000 | 0.632→0.633 | 0.589→0.586 | **REJECTED** |
| iter006 | Identity routing for conv3 filters 24-31 (de-duplicate colour/blob features) | 576 | 0 | 513 | 0.6067→0.6050 | 0.632→0.632 | 0.589→0.586 | **REJECTED** |
| iter007 | Amplify colour pathway x3 (replication of WSP-001 iter007) | 72 | 0 | 513 | 0.6067→0.6000 | 0.632→0.633 | 0.589→0.579 | **REJECTED** |
| iter008 | Wire colour into conv2 corner filters 8-15 | 72 | 0 | 513 | 0.6067→0.5917 | 0.632→0.624 | 0.589→0.560 | **REJECTED** |
| iter009 | Coarser oriented edge filters in conv1 (wavelength 4 -> 6 px) | 600 | 0 | 513 | 0.6067→0.5983 | 0.632→0.631 | 0.589→0.585 | **REJECTED** |
| iter010 | Centroid head rule (no per-feature variance scaling) | 0 | 0 | 513 | 0.6067→0.5867 | 0.632→0.612 | 0.589→0.554 | **REJECTED** |
| iter011 | Scale border-cell head weights by 0.25 | 0 | 0 | 513 | 0.6067→0.5883 | 0.632→0.636 | 0.589→0.570 | **REJECTED** |
| iter012 | Prune the 8 weakest conv3 channels | 0 | 216 | 513 | 0.6067→0.5967 | 0.632→0.634 | 0.589→0.577 | **REJECTED** |
| iter013 | Average colour over a larger area (Gaussian sigma 1.5 -> 5x5 box) | 250 | 0 | 513 | 0.6067→0.6067 | 0.632→0.632 | 0.589→0.589 | **REJECTED** |

Notes: iter001-004 re-tested WSP-001 ideas at scale (reason recorded in each spec). iter004 was predicted to *fail* (warmth barely relates to label at this scale)
and instead raised validation accuracy: the prediction was wrong. A diagnostic on train/val showed the colour signal is present in both
centre and border cells (~57-58% val each), so it is not purely a scene cue. iter013 changed 763 parameters but moved no prediction (236/600 wrong before and after).

## 9. Hand-programmed edits vs closed-form data fitting
Every iteration that changes the extractor also **refits the 513 head parameters** from train class statistics (closed-form LDA). That is data fitting,
not hand programming, and it is where most of the model's accuracy comes from. iter004: 72 hand-written conv weights + 513 fitted head weights; the
gain (if real) cannot be attributed to the 72 hand weights alone. No gradient descent, search, or iterative optimisation was used.

## 10. Controls (validation split)
- Random conv weights + the same closed-form head (20 seeds): mean val accuracy 0.577 (max 0.625).
  60% of random nets matched the hand-programmed baseline (0.572); 10% matched the final model.
  **Hand-programmed features were not better than random convolution features.** (Not repeated on test: the test set was read once.)
- Label-shuffled head on the final extractor: mean 0.511; 1.5% reached the final val accuracy (so ~61% is above chance).

## 11. Audit
Own audit (`src/weight_audit.py`): **PASSED** - static AST scan clean, grad mode off, no optimizer objects, no param/activation in a graph,
scoring read only train/val, 13 iterations replay bit-exactly from parent checkpoint + recorded diff, baseline rebuilds from code,
test sealed until `final`, WSP-001 artifacts unchanged. An independent sub-agent audit found no violations and raised: selection bias on the 600-image validation set
(confirmed by the test result), head refit being data fitting (§9), and a question about iter013 (checked: it changed parameters but had no effect).
Audit-time change: `cmd_final` was extended to record paired per-image outcomes for the test comparison, before it was run; the audit passed again afterwards.

## 12. Limitations and uncertainty
- Absolute accuracy ~62% is weak; a trained small CNN would do far better. 13 iterations is a small search of ideas, not exhaustive.
- Validation (n=600) is too small to resolve 1-3 point effects and was reused 13 times; the one accepted edit did not hold up.
- Closed-form head fitting blurs the "weights programmed purely by reasoning" claim; the agent's edits also followed per-feature statistics on train.
- One dataset (Microsoft Cats vs. Dogs), 64x64 inputs, single seed for the split; web-image quirks (backgrounds, framing) may be exploited.
- The test CI half-width is ~2 points; differences below that cannot be claimed.

## 13. Conclusion (cautious)
Direct, auditable weight programming plus a closed-form head produced a classifier at ~62% (above chance, p≪0.001 vs 50%) with no gradient training.
The LLM-authored edits themselves produced **no verified improvement** on unseen data, and hand-built features were no better than random ones on
validation. The WSP hypothesis, as "an LLM can improve a network by reasoning about its weights", is **not supported** by this experiment; the
infrastructure (transactional edits, rollback, audit, sealed test) works and makes a better-powered follow-up straightforward.

## 14. Recommendations
1. Use a validation set of several thousand images, or a separate "selection" set per iteration, to avoid adaptive overfitting.
2. Pre-register a small number of edits; require replication on a fresh split before acceptance.
3. Measure the hand-programmed extractor against random-feature baselines on the *same* test protocol.
4. Test richer hand-built features (multi-scale, explicit parts/shape) and a larger input size.

## 15. Reproduce
`export WSP_EXPERIMENT=wsp-002`; `python src/dataset_wsp002.py` (needs the zip; already built here); `python src/experiment_runner.py run` (resumable);
`python src/weight_audit.py`; `python src/controls_wsp002.py`; `python src/experiment_runner.py final` (once). Artifacts: `experiments/wsp-002/`,
`checkpoints/wsp-002/` (initial `iter000_initial.pt`, final `iter004.pt`), `logs/wsp-002/execution.log`, `reports/wsp-002/`.


---

## Addendum: Round 2 (post-test exploratory, iterations 14-16)

Requested continuation: keep working while accuracy is below 100%. Because the sealed test set had already been read once (§7), Round 2 is
**exploratory and cannot be validated on test**: edits were judged on train/validation only under the same pre-registered acceptance rule, and the test set was
not read again. The test aggregate was known to the agent when these hypotheses were written (so they are not independent of it), though no per-image test output was seen.

| id | hypothesis | params changed | val acc | CV acc | decision | reason |
|---|---|---|---|---|---|---|
| iter014 | Wire colour into conv2 parallel-pair filters 16-23 | 585 | 0.6067→0.5933 | 0.632→0.623 | **REJECTED** | diagnostic regression (cv -0.0090, val_f1 -0.0262) |
| iter015 | Half-strength conv2 bias gating on the accepted model | 545 | 0.6067→0.6033 | 0.632→0.632 | **REJECTED** | no meaningful val improvement (val -0.0034, cv +0.0000, val_f1 -0.0064) |
| iter016 | Remove oriented-edge inputs from the colour/blob filters | 585 | 0.6067→0.6033 | 0.632→0.633 | **REJECTED** | no meaningful val improvement (val -0.0034, cv +0.0010, val_f1 -0.0021) |

Outcome: 0 of 3 accepted. Totals for WSP-002: **16 iterations, 1 accepted (iter004), 15 rejected.** The accepted checkpoint is unchanged
(`checkpoints/wsp-002/iter004.pt`: val 0.6067, CV 0.632), so the one-time test result in §7 still describes the final model.
Post-round audit: **PASSED** (16 iterations replayed bit-exactly; test split untouched since the single `final` run; WSP-001 artifacts unchanged).
Stopping reason: no remaining hypothesis supported by the evidence gathered; every colour-routing, gating, masking, pruning, scale and head-rule variant tried either regressed a diagnostic or
moved validation by under 1 point. Reaching 100% is not a realistic target for this fixed ~15k-parameter architecture with hand-set features and a linear head on 64x64 inputs.


## Addendum: Round 3 (post-test exploratory, iterations 17-19)

The flow continued with a different kind of hypothesis: explicit structural face priors instead of re-weighting generic features (blob-pair "eye/nostril" detectors in the
redundant conv3 filters, and part-scale blob filters). Same rule, train/validation only, test not read again.

| id | hypothesis | params changed | val acc | CV acc | decision |
|---|---|---|---|---|---|
| iter017 | Bright blob-pair detectors (eyes) in conv3 filters 25-28 | 801 | 0.6067→0.6017 | 0.632→0.631 | **REJECTED** |
| iter018 | Dark blob-pair detectors (nostril/dark eyes) in conv3 filters 29-31 | 729 | 0.6067→0.6133 | 0.632→0.631 | **REJECTED** |
| iter019 | Larger centre-surround blobs in conv1 (sigma 0.8 -> 1.5) | 663 | 0.6067→0.6067 | 0.632→0.634 | **REJECTED** |

Outcome: 0 of 3 accepted (iter018 raised val by +0.7 pt, below the 1-pt rule and with CV -0.1 pt: indistinguishable from noise at n=600).
**Final totals: 19 iterations, 1 accepted (iter004), 18 rejected.** Accepted checkpoint unchanged (`checkpoints/wsp-002/iter004.pt`); the one-time test result (§7) still applies.
Audit after round 3: **PASSED**.
What would change the picture is not more edits at n=600 but a larger validation set (several thousand images) so that 1-pt effects are resolvable and selection bias is smaller.


## Addendum: Round 4 (user-directed continuation at n=600, iterations 20-23)

When asked how to proceed, the user chose to keep iterating on the same 600 validation images without new data. Gains at this sample size cannot be verified (the sealed test
set was read once and is spent); the rule and the no-test-reads policy are unchanged.

| id | hypothesis | params changed | val acc | CV acc | decision |
|---|---|---|---|---|---|
| iter020 | Odd-phase (sine) Gabors in conv1: step-edge detectors | 1113 | 0.6067→0.6000 | 0.632→0.624 | **REJECTED** |
| iter021 | Sharper orientation tuning in conv2 line filters | 529 | 0.6067→0.6017 | 0.632→0.634 | **REJECTED** |
| iter022 | Half-strength conv3 bias gating on the accepted model | 545 | 0.6067→0.5933 | 0.632→0.628 | **REJECTED** |
| iter023 | Hierarchy ablation: pass conv1 features straight through conv2/conv3 | 2105 | 0.6067→0.5983 | 0.632→0.638 | **REJECTED** |

Key finding (iter023, hierarchy ablation): replacing the entire hand-composed conv2/conv3 hierarchy with plain passthrough of the 16 conv1 features gave val 0.598 / CV 0.638 /
train 0.645 versus 0.607 / 0.632 / 0.641 for the accepted model: **no measurable difference**. It was rejected only because val F1 fell more than 1 pt.
So the hand-programmed deeper layers contribute nothing detectable; the closed-form head over low-level features carries the accuracy. This is consistent with the random-feature control (§10)
and explains why 22 different edits of every kind left validation pinned at 60-61%.

**Final totals: 23 iterations, 1 accepted (iter004), 22 rejected.** Accepted checkpoint unchanged (`checkpoints/wsp-002/iter004.pt`); one-time test result (§7) still applies.
Audit after round 4: **PASSED**. 100% accuracy was never reached; with this fixed architecture, hand-set features and a linear head the plateau appears structural,
so a different lever (more data for resolving edits, or a different feature design) is needed rather than more edits of the same family.


## Addendum: Round 5 and stopping decision (iteration 24)

The user directed: work only on the existing data, continue until the best accuracy is reached, and **stop if many tries still do not get there**; and make sure the audit covers
every weight update made by the programming script.

**Round 5 (iter024, multi-scale passthrough):** 16 fine conv1 feature maps + 16 coarse (3x3-box) copies. Result: val 0.6067→0.6050,
CV 0.632→0.640, val F1 0.589→0.578: **REJECTED** (val F1 fell >1 pt, val did not rise).

**Stopping decision.** 24 iterations across edge scale/phase/orientation tuning, colour routing and scale, ReLU gating (conv2, conv3, full and half strength), head rules and masks, channel pruning,
conv3 symmetry breaking, blob-pair face priors, part-scale blobs, a hierarchy ablation and a multi-scale rebuild all landed at **val 59-61%, train CV 61-64%**. Structurally different extractors converge on the same ceiling, so the
limit is the information in pooled low-level features under a linear head, not a missing weight tweak. Per the user's rule, the run stops here. Best model by the pre-registered rule: `checkpoints/wsp-002/iter004.pt`
(val 0.6067, CV 0.632; sealed-test accuracy 61.6%, see §7). 100% was never close.

**Weight-update audit (added at the user's request).** `src/weight_audit.py` now additionally: (1) runtime tripwire in `wsp_engine.run_iteration` rejects any iteration that changes a tensor not covered by logged
`WeightProgrammer` operations; (2) static scan forbids in-place weight mutation (`copy_`, `.data`, direct `weight[...] =` ...) in the agent-authored iteration files, evaluator and all non-programming modules;
(3) replays every iteration's edit on its parent checkpoint and requires an identical checkpoint hash, full operation coverage, a stated hypothesis and a reason on every operation; (4) writes a human-readable
trail to `reports/wsp-002/WEIGHT_UPDATE_AUDIT.md`. The checks were validated by planting violations (all caught). Result after the final round: **PASSED**,
24 iterations replayed bit-exactly, 0 unlogged changes. WSP-001's audit also passes under the new checks and its artifacts are unchanged.

**Final totals: 24 iterations, 1 accepted, 23 rejected.**


## Addendum: Round 6 (iterations 25-27): testing the remaining levers

After round 5 the ceiling looked structural; two specific explanations were tested before stopping.

| id | hypothesis | params changed | val acc | CV acc | decision |
|---|---|---|---|---|---|
| iter025 | Spatially shared head (32 effective weights) | 513 | 0.6067→0.5917 | 0.632→0.629 | **REJECTED** |
| iter026 | Centre/border shared head (64 effective weights) | 513 | 0.6067→0.5933 | 0.632→0.632 | **REJECTED** |
| iter027 | Luminance-distribution (soft histogram) ramps replace the DoG blobs | 817 | 0.6067→0.5917 | 0.632→0.606 | **REJECTED** |

- **Variance is not the limit (iter025/026):** heads with only 32 or 64 effective weights (spatially shared) give CV 0.629 / 0.632 versus 0.632 for the full 512-weight head. Fewer parameters did not help, so the
  head is not over-fitting; the **features simply carry about 63% of the class information**.
- **Luminance histogram features are worse than the blobs they replaced (iter027):** CV 0.632 → 0.606 (-2.6 pts).

**Final totals for WSP-002: 27 iterations, 1 accepted (iter004), 26 rejected.** Accepted checkpoint unchanged (`checkpoints/wsp-002/iter004.pt`); sealed-test accuracy 61.6% (§7). The weight-update audit
(tripwire, mutation scan, bit-exact replay of all 27 iterations) is **PASSED**.

**Why the run ends here.** Every family tried (edges, colour, gating, head rules and sharing, pruning, structural face priors, hierarchy ablation, multi-scale, luminance histograms) converges on the same ~60-64% ceiling, and the
variance explanation has been ruled out. Improving further needs *more informative features learned from more data* (e.g. a training set far larger than 1,000 images, or learned filters), which this hand-programming-only,
existing-data-only protocol does not allow. 100% is not attainable here; the honest finding is that direct weight programming plus a closed-form head saturates near 62% on this task.
