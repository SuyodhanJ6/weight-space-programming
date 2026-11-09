# 03 - Audit record (what was checked, how, what was found, what was fixed)

The audit is central to the project: it is what makes the claim "no gradient training, no peeking, every weight change traceable" checkable.
It combines **static code inspection, runtime checks, checkpoint/history consistency, hash-exact reproducibility replay,** and **independent
reviews by fresh sub-agents**. It is implemented in `src/weight_audit.py` (`src/audit.py` is an alias).

---------------------------------------------------------------------------------------------------

## 1. The automated audit (`python src/weight_audit.py`)

### 1.1 Static checks (AST, not text search)
* **Forbidden imports** anywhere in `src/` (network, subprocess, model hubs, `torchvision`, `transformers`, `openai`, `anthropic`, `torch.optim`, `torch.autograd`, ...). `experiment_runner.py` imports `socket` only to *block* it.
* **Forbidden attributes/calls**: `.backward`, `.autograd`, `.optim`, `.hub`, `grad()`, `from_pretrained`, `os.system`, `requires_grad_(True)`, `torch.load` without `weights_only=True`.
* **In-place weight mutation scan** (added in WSP-002 round 5 at the user's request): the agent-authored iteration files, evaluator and every module outside a small allow-list may not use `copy_`, `zero_`, `fill_`, `add_`, `.data`, `load_state_dict`, or assign into `...weight[...]`/`...bias[...]`. Weights may change **only** through `WeightProgrammer`.

### 1.2 Runtime checks
* Gradient mode is off; **no `torch.optim.Optimizer` instance exists** (scanned through the garbage collector).
* No parameter has `requires_grad` or a `.grad`; every activation of a real forward pass has no `grad_fn`.
* **Which splits are read**: `evaluator.split_data` is wrapped during `score_model`; the audit asserts only `["train", "val"]` are requested.
* **Engine tripwire** (in `wsp_engine.run_iteration`, executed on every iteration): any tensor whose changed-element count exceeds what the logged `WeightProgrammer` operations account for makes the iteration invalid.

### 1.3 Consistency and reproducibility
For every iteration in the history the audit:
1. checks the hypothesis and per-operation reasons exist;
2. verifies the candidate checkpoint's SHA-256;
3. **applies the recorded diff to the parent checkpoint and requires the identical hash** (reproduction from parent + diff);
4. (weight-update audit) **re-runs the agent's own edit function on the parent checkpoint through the engine's `EditContext` and requires an identical SHA-256**, full operation coverage, and a reason on every operation;
5. confirms the recorded changed-parameter count;
6. checks the baseline checkpoint hash, that it is read-only, and that **the baseline can be rebuilt bit-exactly from the code** (proving no external or pretrained weights);
7. checks the accepted checkpoint hash, and that all rejected candidates are preserved.
The human-readable trail is written to `reports/<exp>/WEIGHT_UPDATE_AUDIT.md`; the JSON to `experiments/<exp>/audit_report.json`.

### 1.4 WSP-002+ extras
* Sealed test directory mode must be 000 before the final run; the test cache must not exist; train/val image files must hash to the manifest.
* **Earlier experiments' artifacts** (states, histories, checkpoints, manifests, reports, logs, splits) are re-hashed against the manifest taken before the new experiment began (`wsp001_artifact_hashes.sha256`, `previous_artifact_hashes.sha256`); any change fails the audit.

### 1.5 The audit was tested against planted violations
A throw-away module containing `optim.SGD`, `loss.backward()`, `import urllib`, `torch.load` without `weights_only`, `ctx.model.conv1.weight.copy_(...)`, `weight[0,0,0,0] = 5`, and `bias.data.zero_()` was created; the scanners caught every one (then the file was deleted). The tripwire was shown to flag a 50-element change logged as 10.

## 2. Results per experiment (final state)

| Experiment | Automated audit | Replayed iterations | Independent sub-agent audit | Notes |
|---|---|---|---|---|
| WSP-001 | PASSED (also re-run after each refactor) | 9, hash-exact | no violations | its 9 iterations were also replayed by the later, stricter audit |
| WSP-002 | PASSED | 27, hash-exact | no violations | earlier artifacts untouched |
| WSP-003 | PASSED | 21 (19 at time of review), hash-exact | no violations | 131 earlier hashes verified |
| WSP-004 | PASSED (after fixing an audit bug, 5.3) | 3, hash-exact | no violations; minor documentation fixes | 190 earlier hashes verified |

(The final WSP-004 post-test audit result is in `04_RESULTS_SUMMARY.md`.)

## 3. Independent audits (fresh sub-agents, read-only)

A new agent with no knowledge of my reasoning was instructed to be skeptical, read the code, run read-only checks, never touch the sealed test, and report
violations, concerns and a verdict. Four such reviews were run (one per experiment). Their findings and my responses:

### WSP-001 review
* No violations. Concerns: head refits are closed-form data fits ("hand-set weights" overstated); selection leakage on 4 val / 16 LOO images; iter004 accepted on a +1-image gain; the audit partly relies on static analysis; socket block is a monkeypatch; random controls use kaiming init (control only).
* **Response**: all included in the report as limitations; wording changed to "hand edits plus closed-form fits, no gradient descent".

### WSP-002 review
* No violations. Concerns: selection bias (13 candidates vs 600 val images); iter013 looked like a no-op (I verified it changed 763 parameters with no measurable effect); head refit is data fitting; the audit is partly static; 4 val images / 2,000 test limits.
* **Response**: documented; later rounds were labelled exploratory because the test aggregate was known.

### WSP-003 review
* No violations. Concerns: `lda_full` is a legitimate closed form but accounts for most of the gain (about 80%+); val standard error about 0.8 pt makes the 1-pt rule only about 1.3 SE; ablation-informed designs used val; scale constants are data-derived; one byte-identical cross-label pair in the test manifest; a cosmetic "wsp-002" in an error message; headline rounding.
* **Response**: the report separates "hand / derived / fit" parameters and includes the head-versus-extractor attribution; the duplicate test pair is documented; acceptance thresholds confirmed unaltered (iter019, +0.0097, correctly rejected).

### WSP-004 review
* No violations. Confirmed: no gradients/optimizers/pretrained/network; sealing holds; **zero SHA-256 overlap** of the 3,884 test images with WSP-002, WSP-003 or WSP-004 train/val or the pilot images; 190/190 earlier hashes OK; the replay fix is legitimate and compares exact hashes; acceptance matches the rule; final model equals the baseline.
* Concerns: stale architecture string in the state file (**fixed**: now `CatDogNet(widths=(48,96,96)) ... linear(1536->1)`); accuracy mostly from the data-fitted head; validation reused from WSP-003 across experiments (optimistic); scale constants are train-derived; recommended hashing the test images immediately before the evaluation (**implemented**: `final` re-hashes every test image after unsealing and re-seals on any mismatch); the hash file covers stored artifacts, not the image files of earlier experiments.

## 4. Integrity facts established by the audits

* No gradient computation, backpropagation, optimizer, finite-difference estimate, random weight search, pretrained weight, external model, or API call occurs anywhere in the code path of any experiment.
* Every parameter change of every iteration is recorded (operation list with parameter, selector, reason, counts, sample values), has a stated hypothesis, and reproduces exactly (hash-identical) from its parent checkpoint.
* Rejected and invalid iterations were rolled back and remain in the history.
* The test split of each experiment was read exactly once, after all iterations, by `experiment_runner.py final`.
* Earlier experiments are unchanged (hash-verified) after each later experiment.

## 5. Audit failures and bugs that were found (and fixed)

1. **WSP-003 iter011/iter012 were invalid tests** (stale conv2 bias -2.0, 384/512 dead features). Detected by anomalous accuracy (exactly 50%) and a debugging script. Fixed by new, correct iterations (iter013/014); flagged as "INVALID" in the history/report; no decision relied on them.
2. **The audit's replay failed for WSP-004** (`replay hash differs`). Diagnosis: replay was deterministic (two replays identical) but the head differed greatly from the recorded one, because the baseline checkpoint's metadata has no head-rule field so replay defaulted to the diagonal LDA while WSP-004's baseline uses the full-covariance rule. **Fix**: a single `baseline_head_cfg()` function supplies the default; replay still requires an **exact** SHA-256 match. (Not float noise; confirmed before fixing.)
3. A shell command chain (`&&`) silently skipped a code patch once (the wide-baseline builder); caught immediately when `init` raised an ImportError, and re-applied.
4. The WSP-001 artifact write-protection command failed once through a command wrapper that does not support `-exec`; redone with `xargs chmod` before any code was changed.
5. Stale `architecture` string in WSP-004's state (fixed, above).

## 6. Limitations of the audit (be aware)

* The network block is a Python monkeypatch, not an OS sandbox; the audit also AST-scans for network imports.
* The repository has **no git commits**, so "thresholds never altered" and "default-width code unchanged" rest on the audits' re-execution of earlier experiments (all still pass) and on the sub-agents' reading, not on a diff history.
* Static scanning is by AST pattern; replay is the stronger guarantee.
* The hash manifests cover stored artifacts and manifests, not the image files of earlier experiments.
* Validation images were reused across experiments (WSP-003 val also served WSP-004), accumulating selection bias; only each experiment's sealed test is unbiased.
