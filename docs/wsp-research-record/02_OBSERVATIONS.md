# 02 - Observations (everything we learned, in order, with the numbers)

Conventions: "val" = validation split, "CV" = 5-fold cross-validation over train (head re-fitted per fold),
"test" = the sealed split read once at the end. "pt" = percentage point. All accuracies are two-class (chance = 50%).

---------------------------------------------------------------------------------------------------

## 1. The first 20 images (WSP-001 data)

Viewed on contact sheets (10 cats, 10 dogs), plus numeric statistics (`experiments/dataset_report.json`).

* **Cats**: roundish face, small triangular ears, short muzzle, large widely spaced eyes (flash glow common), 7 of 10 dark or black-and-white, compact/curled poses, cages/beds/chairs behind them.
* **Dogs**: longer protruding muzzle, dark nose, varied ear shapes (floppy/folded/upright), full standing body with long legs, tan/brown/black fur, dirt/grass/fences/kennels behind; one image has a person in a Santa suit.
* All 20: JPEG, RGB (3 channels), 237-500 px per side, **0 corrupted, 0 exact duplicates, 0 near-duplicates**.
* Numeric: dogs were brighter (117 vs 95), warmer (mean R-B 27.5 vs 13.4) and had more edges (11.4 vs 9.7). **This colour/warmth pattern turned out to be a small-sample artifact** (see 7).
* Both classes contain black animals, so colour is at best a weak cue; backgrounds (cages, fences, grass) correlate with setting, a leakage risk.

## 2. What the hand-programmed WSP-001 network was actually doing (found by inspecting weights and activations)

| Observation | Evidence |
|---|---|
| The logit was an **edge-density meter** | correlation of the logit with image edge density r = +0.80 on the 16 development images (+0.91 on WSP-002's 1,000 train images), while the label correlated only +0.34 (+0.25) with edge density. The head weights were almost all positive (mean +0.119, min -0.034). |
| The **colour pathway was dead** | conv2 weight magnitude from the 4 colour channels = exactly 0.0, versus 28.0 from the edge channels. |
| **conv3 filters 24-31 were identical** | max difference of outputs < 1e-6: eight redundant channels (the head summed one signal eight times). |
| **ReLUs did not gate** | conv2 95% active, conv3 100% active, so the network was nearly linear. |
| Centre pooled cells carried more signal | mean |effect size| 0.43 (centre 4 cells) vs 0.10-0.37 (border/bottom row); bottom row weakest (0.10-0.20). |
| Orientation matters unevenly | near-horizontal Gabors had |d| about 0.33 vs about 0.18 for oblique/vertical ones (later: orientations 0,1,7 strongest at 8,000 images). |

## 3. WSP-001 results (20 images; 12 / 4 / 4)

* Baseline: train 91.7%, val 75% (3/4), leave-one-out (16 images) 81.25%.
* 9 iterations: **2 accepted** (iter002 conv2 bias gating, iter004 colour wiring), 7 rejected. Final: val 100% (4/4), LOO 87.5%.
* One-time test (4 images): initial 2/4, final 3/4. **A one-image difference; statistically meaningless.**
* Controls on the 16 dev images: 40 random-conv-feature networks with the same head scored LOO 0.64 on average (max 0.75); none matched the baseline (0.81). A label-shuffle null reached the final LOO in only 0.5% of runs. So the extractor carried some signal on those 16 images, but the evaluation was far too small to say more.
* The accepted gains depended on single images (e.g. dog_03 flipped to correct).

**Lesson 1:** with 16-20 images, every number is noise-level (1 image = 25 pt on val). Neither success nor failure was informative.

## 4. WSP-002 (3,600 images; 1,000 / 600 / 2,000) - what changed once the sample was larger

* **The WSP-001 "81%" was noise**: the same hand-programmed baseline scored **57.2% val / 62.2% test**. The colour/warmth story did not hold (cat warmth 19.7 vs dog 17.6; label correlation with warmth only +0.08).
* 27 iterations in 6 rounds; **1 accepted** (iter004 colour wiring: val 57.2 -> 60.7%). On the sealed test it did **not** replicate: **62.2% -> 61.6%** (McNemar p = 0.59). Selection bias from 13+ looks at the same 600 images was the explanation.
* **Colour as a cue**: wiring colour in helped validation although warmth barely related to the label; a diagnostic showed the colour signal present in both centre cells (57.2% val) and border cells (58.2% val), so it was not purely a scene cue.
* **Hierarchy ablation (iter023)**: replacing the entire hand-composed conv2/conv3 with plain passthrough of the 16 conv1 maps gave val 59.8%, CV 63.8% vs 60.7% / 63.2% for the accepted model. **The hand-built deeper layers contributed nothing measurable.**
* **Variance was not the limit (iter025/026)**: heads sharing weights across cells (32 or 64 effective weights) gave the same CV (62.9%/63.2% vs 63.2%).
* Luminance-histogram features were worse than the DoG blobs they replaced (CV 63.2 -> 60.6%, iter027).
* Controls on val: random conv features + same head averaged **57.7%** (hand baseline 57.2%): **hand-built features were not better than random ones** at that scale. Label-shuffled head: 51% (chance).
* A single colour-wiring edit was the only accepted change out of 27; every other family (edge scale/phase/orientation tuning, ReLU gating, head rules and masks, pruning, symmetry breaking, blob-pair "eye" priors, multi-scale, luminance ramps) landed at **val 59-61%, CV 61-64%**.

**Lesson 2:** a 600-image validation set cannot resolve 1-3 pt effects and gets over-fitted by repeated use. The WSP-002 plateau looked like a feature ceiling but was mostly the *head*.

## 5. WSP-003 (17,000 images; 8,000 / 3,000 / 6,000) - the breakthrough and its true cause

* Baseline: **train 60.2%, val 60.1%, CV 60.1% - all equal**. The model was **bias-limited** (underfitting), not variance-limited. (WSP-002's 57-62% spread was sample noise.)
* **Colour wiring replicated at scale** (iter001): val 60.1 -> 62.4% (+2.3), CV +1.8 (accepted). At 3,000 val images a 1-pt effect is resolvable.
* Under the diagonal head, extra feature variants gave small gains: multi-scale +0.2 val / +0.9 CV; conjunctions +0.64; **colour x orientation soft-AND conjunctions** (iter005, 32 channels) +1.1 (accepted, val 63.6%).
* **The big discovery (iter009): replacing the diagonal-covariance head by a full-covariance closed-form LDA head raised val from 63.6% to 74.4% (+10.9 pt)**, with train 75.2%, CV 74.2% (no over-fit). The 512 features were strongly correlated (fine/coarse and conjunction families); the diagonal head ignored it.
* Under the full head, plain **multi-scale passthrough** (16 fine + 16 coarse conv1 maps) beat the conjunction extractor: **val 76.6%** (iter013, accepted), CV 75.6%.
* **Invalid tests found and corrected**: iter011/012 left a stale conv2 bias of -2.0 so 384 of 512 features were dead (accuracy exactly 50%, all-dog predictions). Found by debugging; hypotheses re-run correctly as iter013/014. Both invalid runs remain in the history, flagged.
* Gains after iter013 saturated: texture-frequency bands +0.1, local peakiness +0.7, odd-phase Gabors +0.7, their combination +0.97 (just under the 1.00 rule, rejected as pre-registered), diagonal hue sectors -0.7, dark-bar polarity -0.3, channel concentration -0.1.
* **Ablation under the full head** (fit on train, check on val): 16 fine marginals alone (256 features) reach **76.5%** (all 32 channels: 77.3%). Dropping orientation filters costs 7.5 pt (69.8%); only orientation filters 73.3%; colour alone 67.7% (dropping it -2.2 pt); DoG alone 64.5% (dropping it -0.9 pt). **The information lives in what the 16 conv1 filters measure.**
* **Sealed test (6,000 images, once): initial 60.7% -> final 77.1%** (95% CI 76.0-78.2%), McNemar p about 1e-107. Test matched validation (76.6%): no visible selection-bias deflation.

### 5.1 Attribution of the gain (validation, all heads fitted on train)

| Configuration | val acc |
|---|---|
| baseline extractor + diagonal head | 60.1% |
| baseline extractor + full-covariance head | 74.4% (**+14.2 pt from the head estimator alone**) |
| final extractor + diagonal head | 62.6% |
| final extractor + full-covariance head (= final model) | 76.6% (**+2.3 pt from the reasoned extractor edits**) |
| random conv filters + full head (20 seeds) | 71.2% mean (max 72.5%) - 0/20 reach the final model or the hand-built baseline extractor |
| label-shuffled head on final extractor | 49.97% (chance) |

**Lesson 3:** about 85% of the improvement is a data-fitted closed-form head; roughly 15% (2-3 pt) is reasoned hand programming, but that hand-built part is real: it beats random filters by 3-5 pt (WSP-002 could not show this).

## 6. WSP-004 (wide network 48/96/96; same train/val as WSP-003; fresh test)

* Reason for widening: after WSP-003 the 16-filter architecture was saturated at about 77% (every re-allocation of the 16 conv1 filters moved val by -0.7 to +1.0 pt) and fine marginals alone already gave 76.5%. More **independent measurements** were the only evidenced lever.
* **Wide baseline (all hand-written filters + closed-form full head): train 79.5%, val 78.4%, CV 77.7%** - already +1.8 pt over WSP-003's best validation. A small train/CV gap (1.8 pt) appears with 1,536 features.
* Ablation: fine only 78.1%, coarse only 77.8%, both 78.4%; odd-phase Gabors the most valuable group (dropping -> 76.8%); even Gabors -> 77.7%; coarse-frequency Gabors the weakest (dropping them gives 78.7%); DoG, colour axes, hues each contribute <= 0.4 pt.
* 3 iterations, **none accepted**: local-peak channels replacing the coarse copies -0.4 pt; colour/orientation/junction conjunctions -1.4 pt; coarse odd-phase Gabors +0.2 pt. The wide network is saturated near 78.5% with this feature family.
* The WSP-004 test result and controls are in `04_RESULTS_SUMMARY.md` (generated after the one-time evaluation).

## 7. Cross-experiment observations

1. **Sample size decides what you can conclude.** WSP-001 (n=4 val): everything looked like success. WSP-002 (n=600): the one "success" failed on test. WSP-003 (n=3,000, test 6,000): the improvement was real and reproduced on test.
2. **Repeated looks at the same validation set inflate it.** WSP-002's +3.5 pt val gain became -0.6 pt on test. In WSP-003 the effect was large enough to survive.
3. **Initial hypotheses were sometimes wrong, and that was recorded**: WSP-002 iter004 was *predicted to fail* (warmth uncorrelated) and instead raised validation; WSP-003's best hypothesis (full-covariance head) was a data-fitting insight, not a weight edit.
4. **Hand-built features beat random features only at larger scale** (WSP-003: +3-5 pt; WSP-002: no difference).
5. **Pretty plateaus are informative**: many different edits converging on the same accuracy indicated a structural limit (the head, then the number of measurements), not a missing tweak.
6. **Colour was the most consistently useful modality** (WSP-003 colour wiring +2.3 pt, ablation: dropping colour -2.2 pt), possibly because of scene context (grass, indoor tones) rather than animal appearance; this was **not** isolated.
7. **The accuracy ceiling of this family** (hand-written low-level filters pooled to a 4x4 grid + a linear head on 64x64 images) is roughly **77-79%**. 100% is not reachable with it.

## 8. Why the network was widened (documented evidence for the design change)

`CLAUDE.md` says to keep the architecture unless a documented, evidence-based need exists. Evidence: (1) train = val = CV at every stage of WSP-003 (bias-limited); (2) ablation: fine marginals alone gave 76.5% of the 77.3% total; (3) all 16-filter re-allocations were within +-1 pt; (4) coarse copies were near-duplicates with the same effect sizes. Therefore only more independent measurements could help.

## 9. Things that did not behave as expected (kept for honesty)

* Colour wiring "failing" prediction in WSP-002 (it helped validation but not test).
* iter013 of WSP-002 produced identical accuracy before/after although it changed 763 parameters (null effect, not a no-op).
* The audit's replay failed once for WSP-004 because the baseline checkpoint carried no head-rule metadata: a **bug in the audit**, not in the experiment (fixed; see 03).
* WSP-002's final test aggregate was known when round-2 and round-3 hypotheses were written; those rounds are labelled "exploratory, unverified".
