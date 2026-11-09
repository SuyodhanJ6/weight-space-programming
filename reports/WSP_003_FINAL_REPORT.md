# WSP-003 Final Report

## 1. Result in one paragraph
On a fresh sealed test set of **6,000 images**, the final model reaches **77.1%** accuracy (95% CI 76.0%-78.2%) versus
**60.7%** for the hand-programmed starting network (paired McNemar p≈2e-107). The improvement is real and generalises: test accuracy (77.1%) matches validation
(76.6%). **Most of it (+14.2% of the +16.5% validation gain) comes from a data-fitted closed-form head, not from hand-written weights**;
the reasoned extractor edits add a further +2.3%, and the hand-programmed extractor beats random convolution filters by 3-5 points under the same head.
No gradient training, optimizer, pretrained weight, or test-set peeking was used. 100% was not reached; ~77% is the best result across WSP-001/002/003.

## 2. How this relates to WSP-001 and WSP-002
Both earlier experiments are preserved unchanged (artifact hashes verified by the audit; both audits still pass). WSP-002 (1,000/600/2,000 images, diagonal head) plateaued at ~62% test; its
colour-wiring "gain" did not replicate on its own test set. WSP-003 uses 8x the training data and a disjoint, larger evaluation, which resolves what WSP-002 could not:
(i) colour wiring *is* a real +2.3-point effect at this scale, and (ii) the earlier plateau was caused mainly by the **diagonal-covariance LDA head** (it ignores strong feature correlations), not by a ceiling in the features.

## 3. Data (offline except one approved download)
**Exception:** `CLAUDE.md` forbids downloads; the user explicitly approved one further download of the official Microsoft Cats vs. Dogs zip (sha256 `b7974bd00a84a999…`) for WSP-003. Afterwards everything ran offline; the zip and extraction were deleted.
- Valid JPEG ≥100 px; rejects (cat/dog) {'exact_duplicate_or_wsp001': 1827, 'not_jpeg': 94, 'too_small_image': 78, 'too_small_file': 3} / {'exact_duplicate_or_wsp001': 1819, 'too_small_image': 80, 'not_jpeg': 135, 'too_small_file': 2}; near-duplicates removed {'cat': 33, 'dog': 27}.
- **Disjoint from all earlier data:** images used by WSP-001 and WSP-002 (including WSP-002's sealed test) were excluded by hash; the audit verified zero overlap between train/val/test and with WSP-002.
- Fixed seed 3, content-independent sampling, class-balanced: **train 8,000 · validation 3,000 · sealed test 6,000**. Train/val files read-only (hash-verified); test in a mode-000 folder, no cache, loader refuses until `final`.
- Train/val statistics (cat vs dog): brightness 117 vs 117, edge density 10.5 vs 11.4, warmth (R-B) 13.9 vs 20.5. Dogs are a little edgier and warmer, so colour is a modest cue in this larger sample.
- Known data quirk: one byte-identical test pair (cat_2054 / dog_1703) carries opposite labels in the source dataset, so one test image is wrong by construction (0.017%). The test set therefore has 5,999 distinct images.

## 4. Architecture and starting point
`CatDogNet` (unchanged, ~15k parameters): conv5x5(3→16)-pool-conv3x3(16→32)-pool-conv3x3(32→32)-avgpool 4x4-linear(512→1). Initial weights (iter000): analytic Gabor/centre-surround/colour-opponent conv1,
hand-composed conv2/conv3, head fitted on train by diagonal LDA. Final extractor: conv1 (as initial + colour wiring), conv2/conv3 replaced by **multi-scale passthrough** (16 fine + 16 coarse feature maps);
head = **full-covariance LDA** (closed form, fixed ridge 0.05·mean variance).

## 5. Training results (8,000 train; used to reason and fit the head, 5-fold CV refits the head per fold)
| | train acc | 5-fold CV acc |
|---|---|---|
| baseline (iter000) | 0.6021 | 0.6014 |
| final | 0.7674 | 0.7564 |

Train, CV and validation agree closely at every stage: the model was *bias-limited* (not over-fitting), which is why richer heads/features paid off.

## 6. Validation results (3,000; used to accept/reject edits; mildly optimistic from 19 looks)
| | accuracy | 95% CI | F1 |
|---|---|---|---|
| baseline | 0.6013 | [0.5837, 0.6187] | 0.5862 |
| final | 0.7663 | [0.7509, 0.7811] | 0.7587 |

Accepted-step progression: iter001: 0.6013 → 0.6243; iter005: 0.6243 → 0.6357; iter009: 0.6357 → 0.7443; iter013: 0.7443 → 0.7663.

## 7. One-time sealed test results (6,000 images, evaluated once)
| model | accuracy | 95% CI | precision | recall | F1 | confusion tp/fp/tn/fn |
|---|---|---|---|---|---|---|
| initial | 0.6072 | [0.5947, 0.6195] | 0.611 | 0.591 | 0.601 | 1772/1129/1871/1228 |
| final | 0.7713 | [0.7605, 0.7818] | 0.787 | 0.743 | 0.765 | 2230/602/2398/770 |

Paired: final right / initial wrong = 1532, initial right / final wrong = 547, McNemar p = 1.9e-107.
Test (77.1%) ≈ validation (76.6%): the validation estimate was not inflated by selection to any visible degree.

## 8. Accepted vs rejected edits (19 iterations: 4 accepted, 15 rejected)
Acceptance rule (pre-registered): accept iff val accuracy rises ≥1.0 pt AND 5-fold train-CV not down >1 pt AND val F1 not down >1 pt. Train accuracy never used. Columns count changed parameters by origin:
**hand** = values written by the agent; **derived** = constants computed from train statistics (normalisation scales, bias calibration); **fit** = head weights from the one-shot closed-form discriminant.
| id | hypothesis | hand | derived | fit | val acc | CV acc | decision |
|---|---|---|---|---|---|---|---|
| iter001 | Wire the colour pathway into conv2 (replication at 8x data) | 72 | 0 | 513 | 0.6013→0.6243 | 0.6014→0.6192 | **ACCEPTED** |
| iter002 | Multi-scale passthrough: 16 fine + 16 coarse conv1 maps (replication at 8x data) | 1708 | 0 | 513 | 0.6243→0.6263 | 0.6192→0.6286 | **REJECTED** |
| iter003 | Three-scale pyramid: fine blur + mid + coarse box | 1708 | 0 | 513 | 0.6243→0.6260 | 0.6192→0.6281 | **REJECTED** |
| iter004 | Colour x texture conjunction detectors (thresholded soft-AND) | 0 | 1732 | 513 | 0.6243→0.6307 | 0.6192→0.6292 | **REJECTED** |
| iter005 | Fine conjunctions: every colour x every single orientation (32 channels) | 0 | 1718 | 513 | 0.6243→0.6357 | 0.6192→0.6279 | **ACCEPTED** |
| iter006 | Hue-bin and colour x blob conjunctions replace the weakest-orientation slots | 0 | 36 | 513 | 0.6357→0.6327 | 0.6279→0.6221 | **REJECTED** |
| iter007 | Neighbourhood-tolerant conjunctions (colour at centre AND orientation over 3x3) | 0 | 288 | 513 | 0.6357→0.6233 | 0.6279→0.6171 | **REJECTED** |
| iter008 | Higher AND threshold (-2.0 -> -3.0): rarer, stronger co-occurrences | 32 | 0 | 513 | 0.6357→0.6333 | 0.6279→0.6288 | **REJECTED** |
| iter009 | Full-covariance LDA head (closed form) [data-fitted head, not a hand edit] | 0 | 0 | 513 | 0.6357→0.7443 | 0.6279→0.7418 | **ACCEPTED** |
| iter010 | [full-LDA head] 16 marginal passthrough + 16 grouped conjunction channels | 0 | 135 | 513 | 0.7443→0.7177 | 0.7418→0.7177 | **REJECTED** |
| iter011 | [full-LDA head] Multi-scale passthrough (16 fine + 16 coarse conv1 maps) (INVALID test: stale bias) | 206 | 0 | 513 | 0.7443→0.5043 | 0.7418→0.5069 | **REJECTED** |
| iter012 | [full-LDA head] Three-scale pyramid with a coarse box scale (INVALID test: stale bias) | 350 | 0 | 513 | 0.7443→0.5043 | 0.7418→0.5072 | **REJECTED** |
| iter013 | [full-LDA head] Multi-scale passthrough with conv2.bias reset (valid re-run of iter011) | 238 | 0 | 513 | 0.7443→0.7663 | 0.7418→0.7564 | **ACCEPTED** |
| iter014 | [full-LDA head] Three-scale pyramid with conv2.bias reset (valid re-run of iter012) | 144 | 0 | 513 | 0.7663→0.7677 | 0.7564→0.7566 | **REJECTED** |
| iter015 | [full-LDA head] Two spatial-frequency Gabor bands (odd orientations -> wavelength 6) | 300 | 0 | 513 | 0.7663→0.7673 | 0.7564→0.7580 | **REJECTED** |
| iter016 | [full-LDA head] Local-peakiness channels replace the coarse marginal copies | 144 | 0 | 513 | 0.7663→0.7733 | 0.7564→0.7641 | **REJECTED** |
| iter017 | [full-LDA head] Diagonal hue sectors replace the DoG blob filters | 300 | 0 | 513 | 0.7663→0.7597 | 0.7564→0.7510 | **REJECTED** |
| iter018 | [full-LDA head] Odd-phase Gabors replace the DoG blob filters | 300 | 0 | 513 | 0.7663→0.7733 | 0.7564→0.7622 | **REJECTED** |
| iter019 | [full-LDA head] Combine local-peakiness channels (iter016) and odd-phase Gabors (iter018) | 444 | 0 | 1026 | 0.7663→0.7760 | 0.7564→0.7634 | **REJECTED** |

Accepted: iter001 (colour wiring), iter005 (32 colour × orientation soft-AND conjunction channels), iter009 (full-covariance LDA head), iter013 (multi-scale passthrough with the full head).
iter011/iter012 were **invalid tests**: the edits replaced conv2/conv3 weights but left the accepted model's conv2.bias=-2.0, so 384 of 512 features were dead (found by debugging, accuracy exactly 50%).
They were rejected and recorded, and iter013/014 re-ran the hypotheses correctly. iter019 (peakiness + odd-phase Gabors) reached +0.97 pt, just under the 1.00 pt rule, and was rejected as pre-registered.
Note: iter005's conjunction extractor was later superseded: under the full-covariance head, plain multi-scale passthrough beat it.

## 9. Hand-programmed edits versus closed-form data fitting (attribution on validation)
| configuration (all heads fitted on train) | val acc |
|---|---|
| baseline extractor + diagonal LDA head (iter000) | 0.6013 |
| baseline extractor + full-covariance head | 0.7437 |
| final extractor + diagonal LDA head | 0.6263 |
| final extractor + full-covariance head (= final model) | 0.7663 |

The head *estimator* alone accounts for about **+14.2%** (60.1 → 74.4). The reasoned extractor edits add about **+2.3%** on top (74.4 → 76.6).
Interpretation: the headline gain is a *data-fitting* result (a closed-form statistical estimator with 513 parameters fitted on 8,000 images), and the genuinely hand-programmed part is the smaller, still significant, remainder. Both are free of gradients, but only the latter is "weights set by reasoning".

## 10. Controls (validation)
- Random convolution filters + the same full-covariance head (20 seeds): mean val 0.712 (min 0.694, max 0.725). 0/20 reach the final model (0.766) and 0/20 reach the hand-programmed baseline extractor
  (0.744). **Reasoned hand-built filters are better than random ones by ~3-5 points**: the first clear evidence of value from hand programming (unlike WSP-002).
- Label-shuffled head on the final extractor: mean 0.500 (max 0.562): chance, so the accuracy is not an artefact.

## 11. Audit
Own audit (`src/weight_audit.py`): **PASSED** - static AST scan clean, in-place weight-mutation scan clean, grad mode off, no optimizer objects, no parameter/activation in a graph, scoring read only train/val,
engine tripwire for unlogged parameter changes, all 19 iterations replay bit-exactly from parent checkpoint with every change logged (`reports/wsp-003/WEIGHT_UPDATE_AUDIT.md`), baseline rebuilds from code,
test sealed until the single `final`, previous experiments' artifacts unchanged (131 hashes). WSP-001 and WSP-002 audits also still pass. An independent sub-agent audit found **no violations** and raised: head contribution (§9),
validation selection bias (§12), data-derived scale constants (§8 "derived"), and the duplicate test pair (§3).

## 12. Limitations and uncertainty
- The headline accuracy is mainly a closed-form fitted-head result; the hand-programming contribution is ~2-3 points over head-only plus a 3-5 point advantage over random filters.
- Validation (n=3,000, SE≈0.8 pt) guided 19 decisions; accepted gains of ~1 pt (iter005) may be partly noise. The sealed test (n=6,000, CI ±1.1 pt) is the unbiased figure.
- Colour/hue/co-occurrence features may exploit scene context (grass, indoor beige) rather than animal appearance; this was not isolated here.
- One dataset (Microsoft Cats vs. Dogs, web photos), 64x64 inputs, a single split seed, 4x4 pooled spatial layout; no robustness or out-of-distribution test.
- A closed-form discriminant with a fixed ridge (0.05) was chosen a priori, not tuned; other ridge values might differ.

## 13. Conclusion (cautious)
Weight-space programming **can** produce a useful, auditable classifier without gradient training: ~77% on 6,000 unseen images, up from ~61%. Two findings matter for the research question:
(1) the largest improvement came from diagnosing, by reasoning and measurement, that a *diagonal* head ignored feature correlations and replacing it with a closed-form full-covariance one, which is data fitting;
(2) hand-designed filters did measurably beat random ones, and four reasoned edits were accepted. The WSP hypothesis is therefore **partly supported**: an LLM agent can find and fix real defects and design useful features by inspection,
but a fitted head, not hand-set weights, carries most of the accuracy, and the remaining hand-edit gains are modest.

## 14. Recommendations
1. Report hand-set and fitted contributions separately by default (as in §9); consider a pure hand-set head as a control.
2. Test whether colour/hue features survive background-controlled data.
3. Add a richer extractor (more filters per layer) so the full-covariance head has more independent information; this is where headroom remains.
4. Reserve a second untouched validation split to confirm the accepted edits, and pre-register a fixed iteration budget.

## 15. Reproduce
`export WSP_EXPERIMENT=wsp-003`; `python src/dataset_wsp002.py` (needs the zip, already built); `python src/experiment_runner.py run` (resumable); `python src/weight_audit.py`;
`python src/controls_wsp002.py`; `python src/experiment_runner.py final` (once, already run). Artifacts: `experiments/wsp-003/`, `checkpoints/wsp-003/` (initial `iter000_initial.pt`, final `iter013.pt`),
`logs/wsp-003/execution.log`, `reports/wsp-003/`.


---

## Addendum: Round 2 (iterations 20-21, after the sealed test was spent)

The flow continued because WSP-003 was still improving (60% → 77%). Because the sealed test had been read once (§7), Round 2 is **exploratory and unverified on test**: edits were judged on validation/CV only under the same pre-registered rule.

| id | hypothesis | params changed | val acc | CV acc | decision |
|---|---|---|---|---|---|
| iter020 | Dark-bar polarity: negated Gabors replace the DoG blob filters | 813 | 0.7663→0.7630 | 0.7564→0.7580 | **REJECTED** |
| iter021 | Spend all channels on the 10 best conv1 maps x {fine, coarse, peak} | 875 | 0.7663→0.7657 | 0.7564→0.7578 | **REJECTED** |

Both were neutral. **Final totals: 21 iterations, 4 accepted (iter001, 005, 009, 013), 17 rejected.** The accepted checkpoint and the one-time test result (§7) are unchanged (`checkpoints/wsp-003/iter013.pt`; test 77.1%).
Post-round audit: **PASSED** (21 iterations replayed bit-exactly).

**Where headroom ends.** An ablation showed 16 fine conv1 maps pooled to 4x4 already give val 0.765 under the full-covariance head (all 32 channels: 0.773), and every re-allocation of those 16 filters (frequency bands, odd phase, dark-bar polarity,
hue sectors, local peaks, concentrating channels) moved validation by -0.7 to +1.0 point. Inside the fixed 16/32/32 architecture the information is nearly exhausted at ~77%. Raising it needs **more independent measurements**: a wider network
(e.g. 32-64 conv1 filters and 64 final channels), and a fresh, unspent sealed test set to verify any gain. Both are outside what the present data and approvals allow.
