# Weight-Space Programming (WSP)

Can an LLM agent build and improve a cat-vs-dog neural network by **reasoning about and directly
editing its weights**, with no gradient-based training? This repo holds the experiment, the
engine that executes the agent's edits, the audit, and the report (`reports/FINAL_REPORT.md`).

## Layout
```
src/model.py              CatDogNet (autograd disabled globally, params requires_grad=False)
src/init_weights.py       hand-programmed initial weights; closed-form head rules
src/weight_programmer.py  inspect / modify / replace / transform parameters, checkpoints (logged)
src/wsp_engine.py         transactional iteration: edit -> validate -> evaluate -> accept/rollback
src/iterations.py         the agent's hypotheses and explicit edits (append-only)
src/evaluator.py          accuracy/precision/recall/F1/confusion/loss(metric), LOO diagnostic
src/experiment_runner.py  CLI: init | run | status | final (resumable, network blocked)
src/weight_audit.py       static + runtime + reproducibility audit (audit.py is an alias)
src/controls.py           random-feature and label-shuffle null controls
src/image_inspector.py    dataset inspection, manifest
experiments/              state, history, decision log, parameter_changes, per-iteration diffs
checkpoints/              iter000_initial.pt (read-only baseline), iterNNN.pt candidates
logs/execution.log        runner log          reports/   PROGRESS, AUDIT, DATASET, FINAL_REPORT
```

## Reproduce / resume
```bash
uv sync
uv run python src/image_inspector.py          # manifest + dataset report
uv run python src/experiment_runner.py run    # resumes from experiments/experiment_state.json
uv run python src/weight_audit.py             # audit
uv run python src/controls.py                 # null controls
uv run python src/experiment_runner.py final  # ONE-TIME test-split evaluation
scripts/run_wsp.sh                            # restart loop around `run`
```
If interrupted: re-run `run`; the in-progress iteration is rolled back to the last accepted
checkpoint (hash-verified), completed iterations are skipped. Never delete `checkpoints/` or
`experiments/`. A process supervisor (`scripts/run_wsp.sh`) is needed to relaunch after a crash.

## Rules followed
No backprop, autograd, optimizers, fine-tuning, finite-difference gradients, pretrained weights,
external APIs, or additional images. The test split is read only by `final`.
