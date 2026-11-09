# Weight-Space Programming (WSP) - Research record

**Question:** can an LLM agent build and improve a cat-vs-dog classifier by reasoning about and directly editing its weights, with no gradient-based training?

**Short answer (details and caveats inside):**
* Yes, partly. A hand-programmed network plus a closed-form head reached **about 77% on 6,000 unseen images** (WSP-003), up from about 61%. About **85% of that gain is a data-fitted closed-form head**; the reasoned weight edits add about 2-3 pt and the hand-built filters beat random filters by 3-5 pt.
* Small samples mislead: with 4 to 600 validation images (WSP-001/002) the "improvements" were noise or failed on the sealed test set.
* The 16-filter family saturates near 77%; a wider hand-built network (WSP-004) starts at about 78% validation. 100% accuracy is not reachable with this approach.
* No gradient training, optimizer, pretrained weight, external model or test-set peeking occurred; every weight change is logged and reproduces hash-exactly (audited, plus independent reviews).

## What we have (inventory)

| Experiment | Data (train/val/test) | Network | Iterations (accepted) | Sealed-test result | Status |
|---|---|---|---|---|---|
| WSP-001 | 12 / 4 / 4 (20 images) | 16/32/32 | 9 (2) | 2/4 -> 3/4 (not meaningful) | complete |
| WSP-002 | 1,000 / 600 / 2,000 | 16/32/32 | 27 (1) | 62.2% -> 61.6% (no real change) | complete |
| WSP-003 | 8,000 / 3,000 / 6,000 | 16/32/32 | 21 (4) | **60.7% -> 77.1%** | complete |
| WSP-004 | 8,000 / 3,000 / 3,884 | wide 48/96/96 | 3 (0) | **78.0%** (WSP-003 model on same fresh test: 76.7%) | complete |

Totals: 60 reasoned weight-edit iterations across four experiments, about 3,400 lines of source, 4 automated audits (hash-exact replay of every iteration), 4 independent sub-agent reviews, 4 approved dataset downloads.

## Reading guide

| File | What it contains |
|---|---|
| `01_HOW_THE_EXPERIMENT_WAS_DONE.md` | the complete method: environment, data pipeline and sealing, network, every initial weight, the head formulas, the edit/accept/rollback engine, statistics, controls, timeline, reproduction commands |
| `02_OBSERVATIONS.md` | every finding in order with numbers (what the network was really doing, what each experiment showed, attribution, ablations, cross-experiment lessons) |
| `03_AUDIT_RECORD.md` | what the audit checks, how it was tested, results per experiment, the four independent reviews and responses, bugs found in the audit and experiments, limitations of the audit |
| `04_RESULTS_SUMMARY.md` | generated tables of all results straight from the artifacts |
| `05_ITERATION_LEDGER.md` | generated: every iteration with hypothesis, edit size, metrics, decision |
| `06_DEVIATIONS_MISTAKES_AND_DISCLOSURES.md` | the honest list: deviations from instructions, indirect information flows, mistakes and corrections, fixed-in-advance choices, scope limits |
| `07_FILE_MAP.md` | where every code file, artifact, checkpoint, log and report lives |
| `data/<experiment>/` | JSON/markdown snapshots: state, history, parameter changes, controls, test results, audit reports, weight-update audit trails |

Per-experiment final reports (written at the time): `reports/FINAL_REPORT.md` (WSP-001), `reports/WSP_002_FINAL_REPORT.md`, `reports/WSP_003_FINAL_REPORT.md`, `reports/WSP_004_FINAL_REPORT.md`.

To regenerate the generated files after any change: `python scripts/build_research_record.py`.
