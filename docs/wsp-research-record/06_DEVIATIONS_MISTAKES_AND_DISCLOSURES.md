# 06 - Deviations, mistakes and disclosures (the honest list)

Everything below is something a reviewer should know. None of it changes the "no gradient training" claim, but several items qualify how strongly the results can be stated.

---------------------------------------------------------------------------------------------------

## A. Deviations from written instructions (all with explicit user approval unless stated)

1. **Downloads vs. `CLAUDE.md`.** `CLAUDE.md` forbids downloads, but the user explicitly approved four downloads of the Microsoft Cats vs. Dogs zip (initial dataset; WSP-002; WSP-003; WSP-004), each "only for this download". Everything after each download ran offline (the runner blocks sockets). Each report records the exception. The zip and extracted trees were deleted after each build.
2. **Dataset deletion.** After the 20 images were selected the user asked to delete the large download ("dont install that, delete that"). I interpreted "that" as the full dataset (not the Python packages) and kept `numpy`/`pillow`/`torch`; the user later approved PyTorch separately.
3. **Model used.** The WSP-002 goal said to use the configured Opus 5.5 model. The session actually ran as **Sonnet 5.5** (the user's `/model` command kept Sonnet 5.5), and the audit sub-agents were Sonnet. This is a deviation from that line of the goal text.
4. **Architecture change.** `CLAUDE.md` says keep the architecture unless a documented, evidence-based need exists. WSP-004 widened the network; the need is documented in 02 section 8 and the user approved it ("Wider net + new data").
5. **Stop-hook goal text vs. user instruction.** An automated Stop hook (text: "don't stop while accuracy is below 100%") conflicted with the user's later instruction to stop if many tries fail. 100% is unattainable here; I asked the user to resolve it twice and followed their answers (continue on existing data, then approve more data and a wider network). The user finally said to stop the goal.

## B. Disclosed information flows (possible indirect influence)

1. **WSP-001 iter004 rationale** cited warmth statistics computed over all 20 images, including the 4 test images. The 16 development images alone give the same numbers (cats 12.4 vs dogs 27.7), and no model output on the test images was used. Disclosed in WSP-001's report.
2. **WSP-002 rounds 2-6** (iterations 14-27) were written *after* the sealed test aggregate had been read (62.2% -> 61.6%). They were labelled "exploratory, unverified on test"; none changed the accepted checkpoint.
3. **WSP-003 round 2** (iterations 20-21) likewise came after its test result was known (and changed nothing).
4. **Validation reuse across experiments**: WSP-003's 3,000-image validation set also served WSP-004. Selection bias accumulates; only each experiment's own sealed test is unbiased. WSP-004's test images are fresh to every model except that the design decisions (e.g. widening) were made after seeing WSP-003's validation/test aggregates.
5. **Design choices informed by ablations** (fit on train, checked on val) - e.g. which conv1 groups to keep or replace - used the validation split, which also acts as the acceptance set.

## C. Mistakes made and corrected

| # | What went wrong | How found | Fix | Effect on results |
|---|---|---|---|---|
| 1 | WSP-003 iter011/012 inherited a stale conv2 bias (-2.0) from the accepted model, killing 384/512 features | accuracy exactly 50% with all-dog predictions; debugging script showed zero-variance features | re-run correctly as iter013/014; originals flagged INVALID | none (no decision relied on the invalid runs) |
| 2 | Audit replay failed for WSP-004 (head-rule default) | hash mismatch; diagnosis: replays identical to each other, head differed by 1.78 | `baseline_head_cfg()`; exact-hash replay retained | none (audit bug, not experiment bug) |
| 3 | A command chain skipped the wide-builder code patch | ImportError at `init` | re-applied | none |
| 4 | WSP-004 state had a stale `architecture` string | independent audit | corrected; engine now derives it from the model widths | none |
| 5 | First write-protection attempt of WSP-001 files failed (a command wrapper rejected `find -exec`) | the command printed an error and the files were not yet read-only | redone with `xargs chmod` before any code change | none |
| 6 | A save path built wrongly in a helper script (contact sheets) | FileNotFoundError | corrected | none |
| 7 | WSP-001: the early WSP-001 iterations were rejected/accepted on 4-image validation | recognised later as statistically meaningless | WSP-002/003 built to fix it | WSP-001 results are exploratory only |
| 8 | In WSP-002 I predicted colour wiring would fail; it improved validation but not test | measured | recorded as a refuted prediction | none |
| 9 | WSP-002 "iter013 looks like a no-op" (raised by the auditor) | checked: 763 parameters changed with null effect | clarified in the report | none |

## D. Choices that were fixed in advance rather than tuned (to avoid hidden hyperparameter search)

* Normalisation constants (0.5, 0.25), Gabor wavelength 4 / sigma 1.5, DoG scales, the soft-AND threshold (-2.0), the full-LDA ridge (0.05 x mean variance), head-logit scale (std 2). Each was chosen once, with a stated reason, and not swept.
* A few single directional tests were made and labelled as such (soft-AND threshold -3.0; coarse-wavelength 6 filters).
* Acceptance thresholds (1.0 pt etc.) never changed; near-misses (WSP-003 iter019, +0.97) were rejected.
* **Not fully free of data-driven choices**: bias centring, channel pruning and the "top-10 channels" selection used train statistics (labelled "derived" in the ledger). The reports say so.

## E. Things that were *not* done (scope limits)

* No per-image visual analysis of the model's errors (activation-level and group-level analyses only).
* No robustness, out-of-distribution or adversarial testing; only one dataset (Microsoft Cats vs. Dogs web photos), 64x64 input, a single split seed per experiment.
* The colour/scene-context confound was not isolated (no background-controlled data).
* The ridge (0.05) and soft-AND threshold were not sensitivity-tested (by design, to avoid a hyperparameter search).
* No git history exists in the repository (all files untracked), so code-change history rests on the audits and these records.

## F. Housekeeping

* Superseded early scripts were removed (`src/main.py`, `src/evaluate.py`, `src/inspect_model.py`, `src/build_initial.py`, `src/tracking.py`); `src/inspect_dataset.py` (from the original setup task) remains.
* Earlier experiments' files were made read-only on purpose; the runner for those experiments will refuse to write (use `--no-write` for audits).
