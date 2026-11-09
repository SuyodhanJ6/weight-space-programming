# WSP-004 Final Report (wide network)

## 1. Result in one paragraph
A **hand-written wide network** (48 conv1 filters, 96/96 channels, 1,536 features) with a closed-form full-covariance head scores **78.0%** (95% CI 76.7%-79.3%)
on **3,884 brand-new unseen test images**, versus **76.7%** for WSP-003's final model scored on the same images
(paired McNemar p = 0.005; +1.4 pt). The three WSP-004 edits were all rejected, so the final model is the hand-written wide baseline itself: **the improvement over WSP-003 comes from widening
the hand-built feature bank, not from accepted edits.** About three quarters of the accuracy still comes from the data-fitted closed-form head (see section 8). WSP-003's model scoring 76.7% here replicates its earlier 77.1%
on a different test set, a useful out-of-sample confirmation. 100% was not reached; the family saturates near 78-79%.

## 2. Why this experiment
WSP-003 showed the 16-filter network was saturated at about 77%: train = val = CV (bias-limited), fine marginals alone gave 76.5% of the 77.3% total, and every re-allocation of the 16 conv1 filters moved validation by -0.7 to +1.0 pt.
Only more independent measurements could help, a documented architecture change approved by the user together with one more download for a fresh test set.

## 3. Data (one approved download; offline afterwards)
Train and validation are **identical to WSP-003** (8,000 / 3,000) for a like-for-like comparison. The **sealed test is every leftover unseen image: {'cat': 1942, 'dog': 1942}** (exact duplicates against all earlier experiments removed;
{'cat': 14, 'dog': 8} near-duplicates inside the new pool and {'cat': 18, 'dog': 14} near-duplicates of train/val images removed). Independent audit: zero SHA-256 overlap of the test with WSP-002, WSP-003, WSP-004 train/val or the pilot images.
The test directory was sealed (mode 000) and **every test image was re-hashed against the manifest immediately after unsealing and before evaluation** (new integrity gate); the split was read exactly once.

## 4. Architecture and initial weights
`CatDogNet(widths=(48,96,96))`. conv1: even Gabors x 8 orientations x 2 polarities, odd-phase Gabors x 8 x 2, 4 DoG, 4 colour axes, 4 diagonal hue sectors, 4 coarse-frequency Gabors (all hand-written);
conv2: fine passthrough + 3x3-box coarse copies (96); conv3: identity with blur (96); head: closed-form full-covariance LDA from train (ridge 0.05 x mean variance, fixed in advance).

## 5. Training / validation
| | train | val | 5-fold CV |
|---|---|---|---|
| baseline = final | 0.7950 | 0.7843 | 0.7765 |

(WSP-003 final: 0.7674 / 0.7663 / 0.7564.) The small train-CV gap (1.8 pt) shows the 1,536-feature head starting to carry variance.

## 6. Iterations (3, none accepted)
Pre-registered rule: accept iff val +>=1.0 pt AND CV not down >1 pt AND val F1 not down >1 pt.
| id | hypothesis | params | val | CV | decision | reason |
|---|---|---|---|---|---|---|
| iter001 | Local-peakiness channels replace the coarse box copies (48 channels) | 22273 | 0.7843→0.7803 | 0.7765→0.7712 | **REJECTED** | no meaningful val improvement (val -0.0040, cv -0.0053, val_f1 -0.0058) |
| iter002 | Soft-AND conjunction channels replace the coarse box copies (48 channels) | 2105 | 0.7843→0.7700 | 0.7765→0.7646 | **REJECTED** | diagnostic regression (cv -0.0119, val_f1 -0.0167) |
| iter003 | Coarse-frequency ODD Gabors replace the weakest group (coarse even Gabors) | 1837 | 0.7843→0.7863 | 0.7765→0.7771 | **REJECTED** | no meaningful val improvement (val +0.0020, cv +0.0006, val_f1 +0.0032) |

Ablation before editing (fit on train, check on val): fine-only 78.1%, coarse-only 77.8%, both 78.4%; odd Gabors most valuable (dropping them 76.8%); coarse-frequency Gabors weakest (dropping them 78.7%). Nonlinear replacements of the coarse copies (local peaks, conjunctions) regressed.

## 7. One-time sealed-test results (3,884 images)
| model | accuracy | 95% CI | precision | recall | F1 | confusion tp/fp/tn/fn |
|---|---|---|---|---|---|---|
| WSP-004 final (= baseline) | 0.7804 | [0.7671, 0.7931] | 0.791 | 0.763 | 0.776 | 1481/392/1550/461 |
| WSP-003 final (reference) | 0.7667 | [0.7532, 0.7798] | 0.774 | 0.753 | 0.763 | 1462/426/1516/480 |

Paired (WSP-004 vs WSP-003 model): {"n": 3884, "final_right_reference_wrong": 199, "reference_right_final_wrong": 146, "mcnemar_exact_p_two_sided": 0.0050434268033464295}. Validation (78.4%) and test (78.0%) agree.

## 8. Hand-written versus fitted (validation, all heads fitted on train)
| configuration | val acc |
|---|---|
| wide extractor + diagonal LDA head | 0.6293 |
| wide extractor + full-covariance head (= final) | 0.7843 |
| random conv filters (48/96/96) + full head, 6 seeds | 0.7498 (min 0.7463, max 0.7540) |
| label-shuffled head | 0.4998 (chance) |

The head estimator alone is worth about **+15.5 pt**; the hand-written filters beat random filters by about **+3.4 pt** (0/6 random nets reach the hand-built one).

## 9. Audit
Automated audit **PASSED** (static, mutation, runtime, 3 iterations replayed hash-exactly, earlier experiments' 190 artifact hashes unchanged, test sealed until the single run, post-test re-run passed).
Independent sub-agent audit: no violations. Issues found and fixed: audit replay used the wrong default head rule for the full-covariance baseline (fixed with `baseline_head_cfg()`, exact hashes still required); stale architecture string in the state (fixed);
pre-evaluation test-image hash gate added. WSP-001/002/003 audits still pass after the architecture refactor.

## 10. Limitations
* Final model = baseline; no edit survived, so this experiment says little about the value of *editing*; it shows the value of a wider hand-built bank.
* Most accuracy is a data-fitted head; validation reused from WSP-003 (selection bias accumulates; the test is the unbiased figure); widening decisions followed WSP-003's validation/test aggregates.
* One dataset, 64x64 inputs, no robustness/out-of-distribution tests, scene-context confound for colour not isolated; controls use 6 random seeds.

## 11. Conclusion
Hand-written, gradient-free, fully audited weights plus a closed-form head reach about **78% on fresh data**, the best of the four experiments, with WSP-003's 77% result confirmed on a new test set. The remaining headroom for this approach lies in more
(and better) hand-built measurements and spatial structure, not in re-weighting the existing ones.

Artifacts: `experiments/wsp-004/`, `checkpoints/wsp-004/`, `logs/wsp-004/`, `reports/wsp-004/`. Full record: `docs/wsp-research-record/`.
