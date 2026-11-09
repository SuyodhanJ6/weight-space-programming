"""WSP-006 iterations: the 19 WSP-005 hypotheses re-tested under acceptance rule v3 on a NEW 12/4/4 split (seed 6).

Scientific reason for re-testing (CLAUDE.md: no repeats without one): WSP-005 rejected 19/19 edits under rule v2, whose val
gate (4 images) and CV guard dominate decisions at n=16. Rule v3 judges by pooled leave-one-out over all 16 non-test images
(the user's requested change). The edits and hypotheses are unchanged; only the rule and split differ, so any accept/reject
difference is attributable to them. Selection bias warning: the 16 LOO images also inspire the hypotheses; only the sealed test is unbiased.
Append-only.
"""

from __future__ import annotations

from dataclasses import replace

from iterations_wsp005 import ITERATIONS as WSP005

BASE_OBS6 = ("WSP-006 baseline (16/32/32 hand filters, diagonal-LDA head on 12 train images): train 0.917, val 0.500, 5-fold CV 0.833, "
             "pooled leave-one-out over all 16 train+val images 0.750 (12/16), LOO F1 0.750. Decision metric is pooled LOO accuracy (rule v3).")

ITERATIONS = [replace(spec, observed=BASE_OBS6 + f" Same hypothesis as WSP-005 {spec.id} ({spec.title}).") for spec in WSP005]
