# 07 - File map (where everything lives)

## Source code (`src/`)
| File | Role |
|---|---|
| `model.py` | `CatDogNet(widths)`: the network; autograd disabled; all parameters zeroed and `requires_grad=False` |
| `init_weights.py` | hand-written filters (16-filter and 48-filter banks); closed-form head rules (`lda`, `lda_demean`, `centroid`, `lda_pooled`, `lda_pooled_cb`, `lda_full`); baseline builders |
| `weight_programmer.py` | the **weight-editing API** (inspect/get/modify/modify_group/replace/transform/compare/validate/save/restore) with per-operation logging |
| `wsp_engine.py` | transactional iteration engine: edit -> validate -> evaluate -> accept/rollback; persistent state, history, decision log, recovery; tripwire for unlogged changes |
| `iterations.py` | WSP-001 hypotheses and edits (9 iterations) |
| `iterations_wsp002.py` | WSP-002 hypotheses and edits (27 iterations) |
| `iterations_wsp003.py` | WSP-003 hypotheses and edits (21 iterations) |
| `iterations_wsp004.py` | WSP-004 hypotheses and edits (3 iterations, wide network) |
| `evaluator.py` | metrics (acc/precision/recall/F1/confusion/margins/BCE), leave-one-out and 5-fold CV diagnostics, chunked forward passes |
| `data.py` | image loading/preprocessing; split loading; sealed-test guard (`unlock_test`) |
| `paths.py` | experiment namespaces selected by `WSP_EXPERIMENT` |
| `dataset_wsp002.py` | dataset builder for WSP-002/003/004 (validate, de-duplicate, sample, seal) |
| `image_inspector.py`, `inspect_dataset.py` | dataset inspection (WSP-001 manifest/report; the original setup inspection script) |
| `experiment_runner.py` | CLI: `init`, `run`, `status`, `final` (one-time), socket blocking |
| `weight_audit.py` (`audit.py`) | the independent audit (static, runtime, consistency, replay, trail) |
| `controls.py`, `controls_wsp002.py` | null controls: random features, label shuffle, head-versus-extractor attribution, paired tests |

## Per-experiment artifacts
| Experiment | State / history / diffs | Checkpoints | Logs | Reports |
|---|---|---|---|---|
| WSP-001 | `experiments/` (state, history, `parameter_changes.json`, `decision_log.md`, `param_changes/`, `manifest.json`, `controls.json`, `final_test_results.json`, `audit_report.json`) | `checkpoints/iter000_initial.pt` ... `iter009.pt` (final `iter004.pt`) | `logs/` | `reports/FINAL_REPORT.md`, `AUDIT.md`, `PROGRESS.md`, `DATASET_INSPECTION.md` |
| WSP-002 | `experiments/wsp-002/` | `checkpoints/wsp-002/` (final `iter004.pt`) | `logs/wsp-002/` | `reports/WSP_002_FINAL_REPORT.md`, `reports/wsp-002/` |
| WSP-003 | `experiments/wsp-003/` | `checkpoints/wsp-003/` (final `iter013.pt`) | `logs/wsp-003/` | `reports/WSP_003_FINAL_REPORT.md`, `reports/wsp-003/` |
| WSP-004 | `experiments/wsp-004/` | `checkpoints/wsp-004/` (baseline `iter000_initial.pt`) | `logs/wsp-004/` | `reports/WSP_004_FINAL_REPORT.md`, `reports/wsp-004/` |

Each `experiments/<exp>/` contains: `experiment_state.json` (current status, metrics, hashes, acceptance rule), `experiment_history.json` (every iteration with hypothesis, operations,
diff summary, metrics, decision), `parameter_changes.json`, `decision_log.md`, `param_changes/iterNNN.pt` (exact old/new values of every changed element), `manifest.json`
(image paths, labels, splits, SHA-256), `dataset_report.json`, `controls.json`, `final_test_results.json`, `audit_report.json`, and the hash manifest of earlier experiments.

## Data (`dataset/`)
* `experiments/pilot20/` - the 20 WSP-001 images (read-only).
* `dataset/wsp-00N/images/{train,val}/` - read-only images; `sealed_test/` (mode 000 until the single final run; 0 after re-sealing); `cache/` - float32 tensors.
* WSP-004's train/val point to WSP-003's image files.

## Scripts
* `scripts/run_wsp.sh` - restart supervisor around the resumable runner.
* `scripts/build_research_record.py` - regenerates `04_RESULTS_SUMMARY.md`, `05_ITERATION_LEDGER.md` and the `data/` snapshots of this folder from the experiment artifacts.

## This folder (`docs/wsp-research-record/`)
`README.md` (index), `01_HOW_THE_EXPERIMENT_WAS_DONE.md`, `02_OBSERVATIONS.md`, `03_AUDIT_RECORD.md`, `04_RESULTS_SUMMARY.md` (generated), `05_ITERATION_LEDGER.md` (generated),
`06_DEVIATIONS_MISTAKES_AND_DISCLOSURES.md`, `07_FILE_MAP.md`, and `data/` (small JSON snapshots of states, histories, controls, test results and audits for all experiments).
