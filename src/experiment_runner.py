"""CLI: init / run / status / final. Safe to re-run: it resumes from saved state.

    uv run python src/experiment_runner.py run
    uv run python src/experiment_runner.py status
    uv run python src/experiment_runner.py final     # one-time test evaluation
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time

import os
from pathlib import Path

import paths
import wsp_engine as eng
from evaluator import evaluate_split, headline
from weight_programmer import restore_checkpoint


def block_network() -> None:
    """Runtime protection: any outbound connection attempt raises."""
    def refuse(*_a, **_k):
        raise RuntimeError("network access is forbidden in this experiment")
    socket.socket.connect = refuse  # type: ignore[assignment]
    socket.create_connection = refuse  # type: ignore[assignment]


def cmd_run(max_iters: int | None) -> None:
    import importlib
    ITERATIONS = importlib.import_module(paths.ITERATIONS_MODULE).ITERATIONS
    state = eng.recover_interrupted(eng.init_state())
    done = 0
    for spec in ITERATIONS:
        if spec.id in state["completed_ids"]:
            continue
        if max_iters is not None and done >= max_iters:
            break
        state = eng.run_iteration(spec)
        entry = eng.read_json(eng.HISTORY, [])[-1]
        print(f"{spec.id}: {entry['decision']} | {entry['decision_reason']}\n   "
              f"{entry['previous_metrics']} -> {entry['new_metrics']}")
        done += 1
    state = eng.read_json(eng.STATE, {})
    state["next_action"] = "write next iterations in src/iterations.py or run final"
    eng.atomic_write(eng.STATE, json.dumps(state, indent=2))
    eng.write_progress(state)


def cmd_final() -> None:
    state = eng.read_json(eng.STATE, {})
    if state.get("test_evaluated"):
        raise SystemExit("test split already evaluated; refusing a second look")
    if paths.IS_V2:  # open the seal only here, once
        import data
        os.chmod(paths.V2_SEALED_DIR, 0o755)
        import hashlib
        manifest = json.loads(paths.V2_MANIFEST.read_text())
        bad = [e["path"] for e in manifest if e["split"] == "test"
               and hashlib.sha256(Path(e["path"]).read_bytes()).hexdigest() != e["sha256"]]
        if bad:                         # integrity gate: re-seal and abort, nothing is evaluated
            os.chmod(paths.V2_SEALED_DIR, 0o000)
            raise SystemExit(f"test image hash mismatch ({len(bad)} files); re-sealed, not evaluated")
        data.unlock_test()
        state["test_unsealed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        eng.atomic_write(eng.STATE, json.dumps(state, indent=2))
    results = {}
    logits = {}
    models = [("initial", state["baseline_checkpoint"]), ("final", state["accepted_checkpoint"])]
    if state.get("reference_checkpoint"):
        models.append(("reference", state["reference_checkpoint"]))
    for label, path in models:
        model, _ = restore_checkpoint(path)
        if paths.IS_V2:
            from evaluator import forward_all, metrics_from_logits, split_data
            images, labels, _names = split_data("test")
            logits[label] = forward_all(model, images)["logit"]
            results[label] = {"checkpoint": path, **metrics_from_logits(logits[label], labels)}
        else:
            results[label] = {"checkpoint": path, **evaluate_split(model, "test")}
    if paths.IS_V2:  # paired comparisons + Wilson intervals, from the same single read
        from controls_wsp002 import mcnemar_exact, wilson
        ok = {k: (v > 0).long() == labels for k, v in logits.items()}
        for k in results:
            results[k]["accuracy_ci95"] = wilson(int(ok[k].sum()), len(labels))
        def paired(a: str, b_: str) -> dict:
            only_b = int((~ok[a] & ok[b_]).sum()); only_a = int((ok[a] & ~ok[b_]).sum())
            return {"n": len(labels), f"{b_}_right_{a}_wrong": only_b, f"{a}_right_{b_}_wrong": only_a,
                    "mcnemar_exact_p_two_sided": mcnemar_exact(only_a, only_b)}
        results["paired"] = paired("initial", "final")
        if "reference" in ok:
            results["paired_vs_reference"] = paired("reference", "final")
    state.update({"test_evaluated": True, "status": "complete", "test_results": results,
                  "test_time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    eng.atomic_write(eng.STATE, json.dumps(state, indent=2))
    eng.atomic_write(eng.EXP / "final_test_results.json", json.dumps(results, indent=2))
    if paths.IS_V2:
        os.chmod(paths.V2_SEALED_DIR, 0o000)
    eng.write_progress(state)
    for k, v in results.items():
        print(k, {m: v[m] for m in v if m in ("accuracy", "accuracy_ci95", "precision", "recall", "f1",
                                              "confusion", "mcnemar_exact_p_two_sided")})
    return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "run", "status", "final"])
    parser.add_argument("--max-iters", type=int)
    args = parser.parse_args()
    block_network()
    eng.setup_logging()
    if args.command == "init":
        eng.init_state()
    elif args.command == "run":
        cmd_run(args.max_iters)
    elif args.command == "final":
        cmd_final()
    state = eng.read_json(eng.STATE, {})
    print(json.dumps({k: state.get(k) for k in ("status", "iteration", "last_completed",
          "accepted_checkpoint", "metrics", "baseline_metrics", "accepted_ids", "rejected_ids")}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
