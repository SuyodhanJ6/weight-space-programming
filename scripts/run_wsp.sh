#!/usr/bin/env bash
# Restart supervisor: re-runs the (resumable) experiment runner until it exits cleanly.
# The runner reads experiments/experiment_state.json, rolls back any interrupted iteration,
# and skips completed iterations, so repeated invocation is safe.
set -u
cd "$(dirname "$0")/.."
for attempt in 1 2 3 4 5; do
  .venv/bin/python src/experiment_runner.py run && exit 0
  echo "runner exited non-zero (attempt $attempt); resuming from saved state" >&2
  sleep 2
done
exit 1
