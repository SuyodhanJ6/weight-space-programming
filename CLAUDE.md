# Autonomous Weight-Space Programming (WSP) Research Instructions

## Mission and scope

You are an autonomous research agent working on an offline Weight-Space Programming (WSP) experiment. The question is whether explicit, hypothesis-driven direct parameter programming can improve a small cat-versus-dog image classifier without conventional training.

The local initial dataset contains exactly 20 images:

- 10 cats (class `0`)
- 10 dogs (class `1`)

This is an exploratory research project, not evidence of general visual classification. Report limitations and negative results honestly. Never alter original dataset images, labels, or the baseline checkpoint to make results look better.

## Existing repository: preserve and extend

Inspect the current repository before changing anything. Preserve useful existing work and adapt it rather than replacing it with a parallel implementation.

The repository already provides the required architecture and most WSP infrastructure:

- `src/model.py`: scratch-built `CatDogNet`; gradient mode is disabled and all parameters have `requires_grad=False`.
- `src/init_weights.py`: explicit initial weight programming and closed-form head construction.
- `src/weight_programmer.py`: the direct parameter-edit API, checkpointing, comparisons, validation, and per-operation records.
- `src/wsp_engine.py`: transactional WSP iteration engine, persistent state, evaluation, acceptance, rollback, and recovery.
- `src/weight_audit.py`: independent static, runtime, checkpoint, and reproducibility audit. `src/audit.py` is an alias.
- `src/iterations.py`: append-only catalog of agent-authored hypotheses and explicit edits.
- `src/evaluator.py`: forward-only metrics and leave-one-out diagnostic.
- `src/experiment_runner.py`: the resumable runner. Despite its role, it is not a conventional training loop.
- `src/data.py`: local dataset manifest and split loading. Image inspection tools may be used as optional aids, but are not required for Claude to inspect images.
- `checkpoints/`, `experiments/`, `logs/`, and `reports/`: durable experiment artifacts.

Keep this design unless a documented, evidence-based need requires a focused change. Do not start an experiment or alter model weights merely because you are reading these instructions.

## Absolute restrictions

The experiment must use only local resources and explicit direct programming. Never introduce, call, or disguise:

- backpropagation, `loss.backward()`, `torch.autograd.grad()`, automatic differentiation, gradient calculation, or finite-difference gradient estimation;
- gradient descent, SGD, Adam, AdamW, any optimizer, conventional training loops, fine-tuning, hyperparameter searches, evolutionary/random parameter searches, or hidden automatic optimization;
- pretrained models, pretrained weights, transfer learning, downloaded feature extractors, existing trained checkpoints, external classifiers, or model inference APIs;
- browsing, web search, downloads, package installation from the internet, external APIs, `curl`, `wget`, remote datasets, or additional images.

When vision capability is available, Claude should inspect the original local cat/dog images directly and record relevant visible properties. This direct visual inspection is for human-readable observation and hypothesis formation only. Claude's vision capability must never provide labels, embeddings, features, predictions, or other outputs as inputs to the neural network. The scratch network must receive the raw image data through its own forward path and use only newly initialized and explicitly programmed parameters.

The Claude service connection needed to operate the agent is permitted, but must not be used as a route to external experiment resources or external model inference. If a needed local dependency or resource is missing, record it as a blocker; do not download it.

Do not weaken existing safeguards, including network blocking, disabled gradient mode, `requires_grad=False`, the test-split guard, or the audit rules.

## Dataset and evaluation integrity

Before any experiment, inspect the local data and update the dataset manifest/report as needed. Verify image counts, paths, labels, dimensions, channels, formats, corruption, duplicates or near-duplicates, and visible diversity. Use local image-reading tools only. Record observed visual properties such as edges, color, texture, ears, faces, fur, shapes, and backgrounds.

Maintain the repository's split boundaries:

- Use the configured train portion to form explicit hypotheses and program parameters.
- Use validation and the existing leave-one-out diagnostic only according to the predeclared acceptance rule.
- Treat the final test split as sealed. It must never influence a hypothesis, parameter selection, edit magnitude, acceptance decision, or architecture choice.
- Run final-test evaluation exactly once through `src/experiment_runner.py final`, after WSP iterations are complete. Do not re-run it or inspect it during iterative work.

Never accept an edit solely due to training accuracy. Clearly distinguish memorization, diagnostic performance, and unseen-test evidence. With 20 images, make the statistical limitation explicit in every final conclusion.

## Direct weight-programming interface

Use `src/weight_programmer.py` as the controlled interface for every parameter edit. Maintain or extend these explicit capabilities:

- `inspect_parameters()`
- `get_parameter()`
- `modify_parameter()`
- `modify_parameter_group()`
- `replace_parameter()`
- `apply_parameter_transform()`
- `compare_parameters()`
- `save_checkpoint()`
- `restore_checkpoint()`
- `validate_parameter_changes()`

Each edit must directly assign a specified value, subset, tensor, or mathematically defined transformation to named parameters. It must be attributable to an explicit LLM-authored research hypothesis. It must not be a random perturbation, automated search, or optimizer step.

Permissible reasoning-led strategies include programming individual weights or biases, selected filters/neuron groups/channels, structured edge/color/texture detectors, and explicitly defined transformations. Describe why the selected parameters represent the feature or behavior at issue, the expected benefit, and plausible side effects.

## Autonomous WSP iteration protocol

When authorized to run the experiment, work autonomously for the configured execution budget; do not ask for approval for individual code edits, checkpoints, direct parameter edits, evaluations, acceptances, or rollbacks. Do ask only if an essential local resource is truly unavailable or a safety/integrity condition makes continuation impossible.

For each unique, hypothesis-driven iteration, use the existing `IterationSpec` and transactional `src/wsp_engine.py` workflow:

1. Inspect the current accepted model, parameter statistics, relevant activations/feature maps, predictions, confidence, and errors.
2. Record the observation, identified problem, hypothesis, selected parameters, proposed direct edit, expected behavior, and risks in `src/iterations.py` and experiment records.
3. Start from the current accepted checkpoint and save a candidate/transactional checkpoint before applying changes.
4. Apply only explicit edits through `WeightProgrammer`.
5. Validate parameter names, shapes, finiteness, autograd state, and forward inference.
6. Evaluate using the permitted train/validation/LOO diagnostics only; never read the test split.
7. Compare against the predeclared acceptance rule. Record actual metrics, side effects, and a clear accept/reject reason.
8. Save an accepted checkpoint for accepted edits. Roll rejected or interrupted edits back to the parent accepted checkpoint while retaining the failed candidate, diff, history, and decision record.
9. Update durable state and plan the next non-duplicative hypothesis.

Do not repeat completed experiments without recording a specific scientific reason. Stop the run when the time/iteration budget is exhausted, no meaningful hypothesis remains, a required resource is unavailable, or an integrity violation occurs. Record the stopping reason.

## Runner and persistent recovery

`src/experiment_runner.py` is the main autonomous WSP orchestrator. `wsp_train.py`, where present, is an experiment runner and entry point, not a conventional training script. Preserve the runner's `init`, `run`, `status`, and one-time `final` behavior. It must restore the latest valid accepted checkpoint, run explicit WSP iterations, execute periodic integrity checks, and never call an optimizer or gradient update.

Persist and atomically update state after every completed iteration. Maintain at least:

- `experiments/experiment_state.json`
- `experiments/experiment_history.json`
- `experiments/current_iteration.json`
- `experiments/parameter_changes.json`
- `experiments/decision_log.md`
- `experiments/param_changes/`
- `logs/execution.log`
- `reports/PROGRESS.md`
- protected baseline and accepted/rejected/latest checkpoint artifacts under `checkpoints/`

State must identify the experiment and iteration, architecture version, split/manifest identity, current and baseline checkpoints and hashes, metrics, hypotheses, parameter diffs, acceptance decisions, current status, last completed operation, and next action.

On start or restart, inspect persisted state and history, verify the latest checkpoint and metadata/hash, determine whether an operation was incomplete, safely recover it, and continue with the next valid iteration. Never reinitialize or overwrite a valid baseline/accepted state. Before interruption, context limit, or shutdown, finish the current durable write or roll back, then save state, logs, progress, and next action. A supervisor script may relaunch the runner after a process crash; the agent need not claim it can relaunch itself.

## Independent audit and reporting

Use `src/weight_audit.py` during and after experiment work. The audit must combine static inspection, runtime checks, checkpoint/history consistency, and reproducibility replay; a simple text search alone is insufficient.

Verify and report:

- no gradients, backpropagation, optimizers, pretrained loading, network access, downloads, or external model calls;
- every parameter change has an explicit hypothesis, old/new values or reproducible diff, reason, evaluation result, and decision;
- every accepted change has a valid hash-verified checkpoint;
- rejected/interrupted changes were rolled back and remain in history;
- state, history, diffs, and checkpoints are mutually consistent and reproducible.

On any violation, stop the affected experiment, preserve evidence, document the violation, and never describe its results as valid WSP results.

When an authorized experiment finishes, produce `reports/FINAL_REPORT.md` covering: objective; data and split limitations; architecture and explicit initialization; baseline; each important hypothesis and direct edit; accepted and rejected iterations; metrics and final one-time test result; audit results; reproducibility/restart instructions; limitations; and a cautious conclusion about the WSP question. A failure to improve is a valid result and must be reported plainly.
