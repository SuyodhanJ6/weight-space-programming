"""Independent audit: static (AST), runtime, and consistency checks.

Run: uv run python src/weight_audit.py   -> experiments/audit_report.json, reports/AUDIT.md
"""

from __future__ import annotations

import ast
import gc
import json
import sys
from pathlib import Path

import torch

import evaluator
import hashlib
import os

import paths
import wsp_engine as eng
from data import load_split
from init_weights import build_baseline_for
from model import CatDogNet
from weight_programmer import restore_checkpoint, state_hash

SRC = Path("src")
BAD_IMPORT_ROOTS = {"urllib", "requests", "http", "httpx", "aiohttp", "socket", "ftplib", "smtplib",
                    "subprocess", "transformers", "torchvision", "timm", "huggingface_hub", "openai",
                    "anthropic", "wget"}
NETWORK_OK_FILES = {"experiment_runner.py"}      # imports socket only to BLOCK it
BAD_ATTRS = {"backward", "autograd", "optim", "hub"}
BAD_CALLS = {"grad", "backward", "from_pretrained", "system", "popen"}
AUDIT_FILES_SKIP = {"weight_audit.py", "audit.py"}  # they name forbidden symbols to look for them


def static_scan() -> list[str]:
    findings = []
    for path in sorted(SRC.glob("*.py")):
        if path.name in AUDIT_FILES_SKIP:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                for name in names:
                    root = name.split(".")[0]
                    if root in BAD_IMPORT_ROOTS and path.name not in NETWORK_OK_FILES:
                        findings.append(f"{path.name}:{node.lineno} imports {name}")
                    if name.startswith("torch.optim") or (isinstance(node, ast.ImportFrom)
                            and node.module == "torch" and any(a.name in ("optim", "autograd") for a in node.names)):
                        findings.append(f"{path.name}:{node.lineno} imports torch optim/autograd")
            elif isinstance(node, ast.Attribute) and node.attr in BAD_ATTRS:
                findings.append(f"{path.name}:{node.lineno} uses attribute .{node.attr}")
            elif isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                if name in BAD_CALLS:
                    findings.append(f"{path.name}:{node.lineno} calls {name}()")
                if name == "requires_grad_" and node.args and getattr(node.args[0], "value", None) is not False:
                    findings.append(f"{path.name}:{node.lineno} requires_grad_ not False")
                if name == "load" and isinstance(fn, ast.Attribute) and getattr(fn.value, "id", "") == "torch":
                    kw = {k.arg: getattr(k.value, "value", None) for k in node.keywords}
                    if kw.get("weights_only") is not True:
                        findings.append(f"{path.name}:{node.lineno} torch.load without weights_only=True")
    return findings


MUTATORS = {"copy_", "zero_", "fill_", "add_", "sub_", "mul_", "div_", "normal_", "uniform_",
            "clamp_", "load_state_dict", "data"}
MUTATION_ALLOWED = {"model.py", "init_weights.py", "weight_programmer.py", "wsp_engine.py",
                    "controls.py", "controls_wsp002.py", "dataset_wsp002.py", "image_inspector.py",
                    "audit.py", "weight_audit.py"}


def mutation_scan() -> list[str]:
    """Weights may only be changed through WeightProgrammer: flag in-place mutators anywhere else
    (notably the agent-authored iterations files and the evaluator)."""
    findings = []
    for path in sorted(SRC.glob("*.py")):
        if path.name in MUTATION_ALLOWED:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute) and node.attr in MUTATORS:
                findings.append(f"{path.name}:{node.lineno} uses .{node.attr} outside the programming API")
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    chain = ast.dump(t)
                    if isinstance(t, ast.Subscript) and ("'weight'" in chain or "'bias'" in chain):
                        findings.append(f"{path.name}:{node.lineno} writes a weight/bias tensor directly")
    return findings


def _iteration_specs() -> dict:
    import importlib
    ITERATIONS = importlib.import_module(paths.ITERATIONS_MODULE).ITERATIONS
    return {spec.id: spec for spec in ITERATIONS}


def weight_update_audit(state: dict) -> dict:
    """Replay every iteration's edit on its parent checkpoint through the engine's own
    EditContext and require: (a) identical resulting hash, (b) every changed tensor covered by
    logged WeightProgrammer operations, (c) hypothesis + reasons present."""
    from weight_programmer import WeightProgrammer
    specs = _iteration_specs()
    history = eng.read_json(eng.HISTORY, [])
    trail, problems = [], []
    for entry in history[1:]:
        if entry.get("decision") == "interrupted":
            continue
        iid = entry["id"]
        ckpt = entry["candidate_checkpoint"]
        parent, pmeta = restore_checkpoint(entry["previous_checkpoint"])
        before = {k: v.clone() for k, v in parent.state_dict().items()}
        ctx = eng.EditContext(WeightProgrammer(parent, iid), dict(pmeta.get("head_cfg", eng.baseline_head_cfg())))
        specs[iid].edit(ctx)
        replay_hash = state_hash(parent.state_dict())
        diff = {k: int((before[k] != parent.state_dict()[k]).sum()) for k in before}
        diff = {k: v for k, v in diff.items() if v}
        uncovered = eng.unlogged_changes({k: {"changed": v} for k, v in diff.items()}, ctx.programmer.operations)
        match = replay_hash == ckpt["sha256"]
        if not match:
            problems.append(f"{iid}: replay hash differs from recorded checkpoint")
        if uncovered:
            problems.append(f"{iid}: {uncovered}")
        if not specs[iid].hypothesis or not all(op["reason"] for op in ctx.programmer.operations):
            problems.append(f"{iid}: missing hypothesis or operation reason")
        trail.append({"id": iid, "decision": entry["decision"], "parent": entry["previous_checkpoint"],
                      "operations": [(op["op"], op["parameter"], op["selector"], op["n_changed"]) for op in ctx.programmer.operations],
                      "changed_per_tensor": diff, "candidate_sha256": ckpt["sha256"],
                      "replay_matches": match, "all_changes_logged": not uncovered})
    return {"iterations_replayed": len(trail), "problems": problems, "trail": trail}


def write_trail_report(upd: dict) -> None:
    lines = ["# Weight-update audit trail", "",
             "Every weight change goes through `WeightProgrammer`; each iteration was replayed from its parent checkpoint.", "",
             "| id | decision | operations | changed per tensor | replay hash | all changes logged |", "|---|---|---|---|---|---|"]
    for t in upd["trail"]:
        ops = "; ".join(f"{o[0]}({o[1]}, {o[3]} changed)" for o in t["operations"])
        lines.append(f"| {t['id']} | {t['decision']} | {ops} | {t['changed_per_tensor']} | "
                     f"{'MATCH' if t['replay_matches'] else 'MISMATCH'} | {'yes' if t['all_changes_logged'] else 'NO'} |")
    (paths.REPORT_DIR / "WEIGHT_UPDATE_AUDIT.md").write_text("\n".join(lines) + "\n")


def runtime_checks(state: dict) -> dict:
    out = {}
    out["grad_mode_disabled"] = not torch.is_grad_enabled()
    out["optimizer_instances_alive"] = sum(isinstance(o, torch.optim.Optimizer) for o in gc.get_objects())
    model, _ = restore_checkpoint(state["accepted_checkpoint"])
    out["params_require_grad"] = [n for n, p in model.named_parameters() if p.requires_grad or p.grad is not None]
    images, _, _ = load_split("val")
    acts = model.forward_with_activations(images)
    out["activations_with_graph"] = [k for k, v in acts.items() if v.requires_grad or v.grad_fn is not None]
    requested = []
    original = evaluator.split_data
    evaluator.split_data = lambda s: (requested.append(s), original(s))[1]
    try:
        evaluator.score_model(model, state["head_cfg"])
    finally:
        evaluator.split_data = original
    out["splits_read_by_score_model"] = sorted(set(requested))
    return out


def consistency_checks(state: dict) -> dict:
    history = eng.read_json(eng.HISTORY, [])
    changes = {c["id"]: c for c in eng.read_json(eng.CHANGES, [])}
    problems, replayed, warnings = [], [], []
    base, _ = restore_checkpoint(state["baseline_checkpoint"])
    baseline_ok = state_hash(base.state_dict()) == state["baseline_sha256"]
    tx, ty, _ = load_split("train")
    rebuilt = build_baseline_for(paths.EXPERIMENT, tx, ty)
    rebuild_ok = state_hash(rebuilt.state_dict()) == state["baseline_sha256"]
    for entry in history[1:]:
        iid = entry["id"]
        if entry.get("decision") == "interrupted":
            continue
        if not entry.get("hypothesis") or not entry.get("operations"):
            problems.append(f"{iid}: missing hypothesis or operations")
        if iid not in changes:
            problems.append(f"{iid}: not in parameter_changes.json")
        if not all(op.get("reason") for op in entry.get("operations", [])):
            problems.append(f"{iid}: operation without reason")
        ckpt = entry["candidate_checkpoint"]
        cand, meta = restore_checkpoint(ckpt["path"])
        if state_hash(cand.state_dict()) != ckpt["sha256"]:
            problems.append(f"{iid}: checkpoint hash mismatch")
        parent, _ = restore_checkpoint(meta["parent"])
        diff = torch.load(entry["full_diff_file"], weights_only=True)
        sd = {k: v.clone() for k, v in parent.state_dict().items()}
        for key, d in diff.items():
            for idx, val in zip(d["index"], d["new"]):
                sd[key][tuple(idx.tolist())] = val
        reproduced = state_hash(sd) == ckpt["sha256"]
        zero_only = False
        if not reproduced:   # diffs logged before bit-exact logging omit -0.0 -> +0.0 changes (equal by value)
            zero_only = all(bool(((sd[k] + 0.0).view(torch.int32) == (v + 0.0).view(torch.int32)).all())
                            for k, v in cand.state_dict().items())
        replayed.append({"id": iid, "decision": entry["decision"], "reproduced_from_parent_and_diff": reproduced,
                         "reproduced_up_to_signed_zero_only": zero_only})
        if not reproduced and zero_only:
            warnings.append(f"{iid}: reproduces exactly except sign-of-zero (-0.0 vs +0.0) entries omitted from the "
                            "logged diff (value-equal; logger made bit-exact afterwards)")
        elif not reproduced:
            problems.append(f"{iid}: not reproducible from parent + recorded diff")
        if sum(v["changed"] for v in entry["parameter_diff"].values()) != entry["n_changed_parameters"]:
            problems.append(f"{iid}: changed-count mismatch")
    acc, _ = restore_checkpoint(state["accepted_checkpoint"])
    return {"baseline_hash_ok": baseline_ok, "baseline_rebuilt_from_code_matches": rebuild_ok,
            "accepted_hash_ok": state_hash(acc.state_dict()) == state["accepted_sha256"],
            "baseline_file_read_only": not (Path(state["baseline_checkpoint"]).stat().st_mode & 0o222),
            "iterations_audited": len(replayed), "replays": replayed, "warnings": warnings,
            "rejected_preserved": all((paths.CHECKPOINT_DIR / f"{i}.pt").exists() for i in state["rejected_ids"]),
            "no_test_metrics_in_iteration_history": not any("test" in json.dumps(e.get("new_metrics", {}))
                                                            for e in history),
            "problems": problems}


def v2_checks(state: dict) -> dict:
    """wsp-002: sealed test, dataset immutability, and WSP-001 artifacts untouched."""
    sealed = paths.V2_SEALED_DIR
    mode = sealed.stat().st_mode & 0o777
    manifest = json.loads(paths.V2_MANIFEST.read_text())
    train_val_ok = all(hashlib.sha256(Path(e["path"]).read_bytes()).hexdigest() == e["sha256"]
                       for e in manifest if e["split"] != "test")
    wsp001 = paths.HASH_FILE
    changed = []
    for line in wsp001.read_text().splitlines():
        digest, name = line.split("  ", 1)
        if name.startswith(("src/", "scripts/", "README", "pyproject")):
            continue
        if not Path(name).exists() or hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            changed.append(name)
    return {"test_sealed_mode": oct(mode), "test_cache_absent_before_final":
            state.get("test_evaluated") or not (paths.V2_CACHE_DIR / "test.pt").exists(),
            "train_val_images_unchanged": train_val_ok,
            "test_split_counts": {s: sum(e["split"] == s for e in manifest) for s in ("train", "val", "test")},
            "previous_experiment_artifacts_changed": changed, "wsp001_artifacts_changed": changed}


def run_audit(write: bool = True) -> dict:
    state = eng.read_json(eng.STATE, {})
    report = {"experiment": paths.EXPERIMENT, "static_findings": static_scan(),
              "mutation_findings": mutation_scan(), "runtime": runtime_checks(state),
              "consistency": consistency_checks(state), "weight_updates": weight_update_audit(state)}
    if paths.IS_V2:
        report["wsp002"] = v2_checks(state)
    rt, cs = report["runtime"], report["consistency"]
    report["passed"] = (not report["static_findings"] and not report["mutation_findings"]
                        and not report["weight_updates"]["problems"] and rt["grad_mode_disabled"]
                        and rt["optimizer_instances_alive"] == 0 and not rt["params_require_grad"]
                        and not rt["activations_with_graph"]
                        and rt["splits_read_by_score_model"] == ["train", "val"]
                        and not cs["problems"] and cs["baseline_hash_ok"]
                        and cs["baseline_rebuilt_from_code_matches"] and cs["accepted_hash_ok"]
                        and cs["rejected_preserved"])
    if paths.IS_V2:
        v2 = report["wsp002"]
        report["passed"] = (report["passed"] and not v2["wsp001_artifacts_changed"]
                            and v2["train_val_images_unchanged"] and v2["test_cache_absent_before_final"]
                            and (state.get("test_evaluated") or v2["test_sealed_mode"] == "0o0"))
    if write:
        write_trail_report(report["weight_updates"])
        (paths.EXP / "audit_report.json").write_text(json.dumps(report, indent=2))
        (paths.REPORT_DIR / "AUDIT.md").write_text("# Audit report\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n")
    return report


if __name__ == "__main__":
    rep = run_audit(write="--no-write" not in sys.argv)
    print(json.dumps({k: v for k, v in rep.items() if k != "consistency"}, indent=2))
    cons = rep["consistency"]
    print({k: v for k, v in cons.items() if k != "replays"}); print("PASSED" if rep["passed"] else "FAILED")
    sys.exit(0 if rep["passed"] else 1)
