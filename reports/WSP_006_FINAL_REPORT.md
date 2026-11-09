# WSP-006 final report: 20-image replication under a new acceptance rule (pooled leave-one-out)

## Why this experiment exists
WSP-005 rejected 19 of 19 edits under rule v2 (validation +1 point, CV guard, F1 guard). With a 4-image validation set that rule is
nearly impossible to pass. At your request ("you can edit weights, please update the rule") the rule was changed. To keep WSP-005's history
honest, the new rule runs as a separate experiment on the same 20 images.

## Setup
- 20 local images, new split seed 6: 12 train / 4 val / 4 sealed test. No downloads. All images were seen in earlier experiments, so the
  test is sealed but not "fresh".
- Same 16/32/32 hand-written network and diagonal-LDA head. Baseline: train 91.7%, val 50%, CV 83.3%, **pooled LOO over all 16 train+val images 75% (12/16)**.
- **Rule v3 (pre-registered before any edit):** accept iff pooled leave-one-out accuracy over the 16 train+val images (head refit on 15, predict 1) rises by >= 1 image (+0.0625)
  AND LOO F1 does not fall. Train accuracy, the 4-image val split and CV are recorded but not used to accept. Test never read during iterations.
- The 19 WSP-005 hypotheses were re-run unchanged (scientific reason: only the rule and split differ). Rule change was decided after seeing WSP-005's results; this is disclosed, not hidden.

## Result of the loop
- **1 accepted: iter003, wire the colour-opponent pathway into conv2** (72 weights, plus closed-form head refit). LOO 12/16 -> 13/16, LOO F1 0.75 -> 0.80.
  On this split val also rose 50% -> 75% and CV 83% -> 92%, so the other diagnostics agree.
- 18 rejected: 11 for LOO F1 regression, 7 for no LOO gain. Later iterations were judged against the iter003 model.
- Under rule v2 the same edit was also a no-op in WSP-005 (different split). The accept is therefore split- and rule-dependent.

## Honest reading
One accept out of 19 attempts, each judged by +1 image out of 16, is within what noise can produce. Selection bias is real: the same 16 images guided the
hypotheses and decided acceptance. The colour edit is plausible (dogs warmer on average) but not proven. Only the sealed test is unbiased.

## Final one-time test (4 sealed images)
Initial model **3/4 = 75%** (CI 30-95%); final model **3/4 = 75%** (same CI), McNemar p = 1.0. The accepted edit did not change the test result.
With 4 images this cannot show improvement or harm.

## Audit
`WSP_EXPERIMENT=wsp-006 python src/weight_audit.py`: passed before and after the test. Test sealed until `final`, test cache absent before it, train/val unchanged,
268 earlier-experiment artifact hashes unchanged, 19/19 iterations replayed hash-exact, no warnings. No independent sub-agent review was run.

## Conclusion
Changing the rule let an edit be accepted, but it did not produce a measurable improvement on unseen data. The honest summary stays the same as WSP-001/005:
at n=20, weight edits cannot be shown to help; the real, measurable gains come from thousands of images (WSP-003/004), mostly via the closed-form head.

## Reproduce
`WSP_EXPERIMENT=wsp-006 python src/dataset_wsp006.py` -> `src/wsp_train.py init|run|final` -> `src/weight_audit.py`. Files: `experiments/wsp-006/`, `checkpoints/wsp-006/`, `reports/wsp-006/`.
