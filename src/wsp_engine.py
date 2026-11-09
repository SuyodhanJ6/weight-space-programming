"""Transactional Weight-Space Programming engine with persistent, restartable state.

The engine only EXECUTES explicit edits written by the reasoning agent
(src/iterations.py). It never searches for weights and uses no gradients.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

from data import load_split
from evaluator import headline, score_model
from init_weights import program_head
from model import CatDogNet
from weight_programmer import (WeightProgrammer, compare_parameters, differs, restore_checkpoint,
                               save_checkpoint, state_hash, validate_parameter_changes)

import paths
from paths import EXP, IS_V2

STATE = EXP / "experiment_state.json"
HISTORY = EXP / "experiment_history.json"
CURRENT = EXP / "current_iteration.json"
DECISIONS = EXP / "decision_log.md"
CHANGES = EXP / "parameter_changes.json"
DIFF_DIR = EXP / "param_changes"
EXEC_LOG = paths.LOG_DIR / "execution.log"
PROGRESS = paths.REPORT_DIR / "PROGRESS.md"
BASELINE_CKPT = paths.BASELINE_CKPT

# Pre-registered acceptance rule (written before any iteration was run).
ACCEPTANCE_RULE = (
    "Accept iff NO regression on val accuracy AND NO regression on LOO accuracy (train+val, 16 "
    "images, head refit per fold) AND at least one of: val accuracy up by >=1 image (0.25), LOO "
    "accuracy up by >=1 image (0.0625), or (both accuracies unchanged and) val margin up by >=0.25 "
    "with LOO margin not down by >0.10. Train accuracy is recorded but never used to accept. "
    "Test split is never read during iterations."
)
ACCEPTANCE_RULE_V2 = (
    f"{paths.EXPERIMENT}, pre-registered before any edit. Accept iff (a) validation accuracy rises by >= 0.010 "
    "(1 percentage point) over the accepted checkpoint AND (b) 5-fold CV accuracy on TRAIN does not "
    "fall by more than 0.010 AND (c) validation F1 does not fall by more than 0.010. Train accuracy "
    "is recorded but never used to accept. The sealed test split is never read during iterations."
)
EPS = 1e-9


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def read_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def log(msg: str) -> None:
    EXEC_LOG.parent.mkdir(exist_ok=True)
    logging.getLogger("wsp").info(msg)


def setup_logging() -> None:
    EXEC_LOG.parent.mkdir(exist_ok=True)
    fmt = "%(asctime)s %(message)s"
    try:
        logging.basicConfig(filename=EXEC_LOG, level=logging.INFO, force=True, format=fmt)
    except PermissionError:
        # Protected (read-only) experiment: log to stderr instead of crashing; nothing is written to its files.
        logging.basicConfig(level=logging.INFO, force=True, format=fmt)


@dataclass
class IterationSpec:
    id: str
    title: str
    observed: str
    problem: str
    hypothesis: str
    selected: str
    expected: str
    risks: str
    edit: Callable[["EditContext"], None]


@dataclass
class EditContext:
    programmer: WeightProgrammer
    head_cfg: dict
    notes: list[str] = field(default_factory=list)

    @property
    def model(self) -> CatDogNet:
        return self.programmer.model

    def refit_head(self, rule: str | None = None, cell_mask: list[float] | None = None,
                   reason: str = "") -> None:
        """Closed-form head assignment from TRAIN statistics, recorded as parameter edits."""
        if rule is not None:
            self.head_cfg["rule"] = rule
        if cell_mask is not None:
            self.head_cfg["cell_mask"] = cell_mask
        images, labels, _ = load_split("train")
        scratch = CatDogNet(self.model.widths)
        scratch.load_state_dict(self.model.state_dict())
        program_head(scratch, images, labels, self.head_cfg["rule"], self.head_cfg.get("cell_mask"))
        self.programmer.replace_parameter("head.weight", scratch.head.weight, reason + " [closed-form head]")
        self.programmer.replace_parameter("head.bias", scratch.head.bias, reason + " [closed-form head]")


def baseline_head_cfg() -> dict:
    """Head rule the iter000 baseline was built with (not recorded in the read-only baseline checkpoint's meta)."""
    return {"rule": "lda_full" if paths.EXPERIMENT == "wsp-004" else "lda", "cell_mask": None}


def init_state() -> dict:
    """Create experiment state from the protected baseline checkpoint (idempotent)."""
    if STATE.exists():
        return read_json(STATE, {})
    if IS_V2 and not Path(BASELINE_CKPT).exists():
        from init_weights import build_baseline_for
        tx, ty, _ = load_split("train")
        save_checkpoint(build_baseline_for(paths.EXPERIMENT, tx, ty), "iter000_initial",
                        {"note": f"{paths.EXPERIMENT} hand-programmed initial; head fit on its train split"})
        os.chmod(BASELINE_CKPT, 0o444)
    model, _ = restore_checkpoint(BASELINE_CKPT)
    head_cfg = baseline_head_cfg()
    score = score_model(model, head_cfg)
    state = {
        "experiment_id": paths.EXPERIMENT, "status": "running",
        "architecture": "CatDogNet: conv5x5(3->{0})-pool-conv3x3({0}->{1})-pool-conv3x3({1}->{2})-"
                        "avgpool4x4-linear({3}->1)".format(*model.widths, model.widths[2] * 16),
        "iteration": 0, "last_completed": "iter000", "completed_ids": ["iter000"],
        "accepted_ids": ["iter000"], "rejected_ids": [],
        "baseline_checkpoint": BASELINE_CKPT,
        "baseline_sha256": state_hash(model.state_dict()),
        "accepted_checkpoint": BASELINE_CKPT, "accepted_sha256": state_hash(model.state_dict()),
        "head_cfg": head_cfg, "metrics": headline(score), "baseline_metrics": headline(score),
        "splits": read_json(Path("dataset/processed/splits.json"), {}) if not IS_V2 else {"manifest": str(paths.V2_MANIFEST)},
        "manifest": str(EXP / "manifest.json"), "test_evaluated": False,
        "reference_checkpoint": "checkpoints/wsp-003/iter013.pt" if paths.EXPERIMENT == "wsp-004" else None,
        "acceptance_rule": ACCEPTANCE_RULE_V2 if IS_V2 else ACCEPTANCE_RULE, "next_action": "run pending iterations",
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    atomic_write(STATE, json.dumps(state, indent=2))
    atomic_write(HISTORY, json.dumps([{"id": "iter000", "title": "hand-programmed baseline",
                                       "decision": "baseline", "metrics": headline(score),
                                       "checkpoint": BASELINE_CKPT}], indent=2))
    atomic_write(CHANGES, "[]")
    atomic_write(DECISIONS, "# Decision log\n\n## Pre-registered acceptance rule\n\n"
                 f"{ACCEPTANCE_RULE_V2 if IS_V2 else ACCEPTANCE_RULE}\n\n## iter000: baseline\n\nHand-programmed initial network. "
                 f"{headline(score)}\n")
    write_progress(state)
    log("state initialised from baseline")
    return state


def recover_interrupted(state: dict) -> dict:
    """If an iteration was in progress when the process died, mark it and roll back."""
    cur = read_json(CURRENT, None)
    if cur and cur.get("status") == "in_progress":
        cur["status"] = "interrupted_rolled_back"
        atomic_write(CURRENT, json.dumps(cur, indent=2))
        history = read_json(HISTORY, [])
        history.append({"id": cur["id"], "title": cur.get("title", ""), "decision": "interrupted",
                        "checkpoint": state["accepted_checkpoint"]})
        atomic_write(HISTORY, json.dumps(history, indent=2))
        state["rejected_ids"].append(cur["id"])
        state["completed_ids"].append(cur["id"])
        log(f"recovered interrupted iteration {cur['id']}: rolled back to {state['accepted_checkpoint']}")
    return state


def decide_v2(prev: dict, new: dict) -> tuple[bool, str]:
    dv, dc, df = (new["val_acc"] - prev["val_acc"], new["cv_acc"] - prev["cv_acc"],
                  new["val_f1"] - prev["val_f1"])
    if dc < -0.01 - EPS or df < -0.01 - EPS:
        return False, f"diagnostic regression (cv {dc:+.4f}, val_f1 {df:+.4f})"
    if dv >= 0.01 - EPS:
        return True, f"val improvement (val {dv:+.4f}, cv {dc:+.4f}, val_f1 {df:+.4f})"
    return False, f"no meaningful val improvement (val {dv:+.4f}, cv {dc:+.4f}, val_f1 {df:+.4f})"


def decide(prev: dict, new: dict) -> tuple[bool, str]:
    if IS_V2:
        return decide_v2(prev, new)
    dv = new["val_acc"] - prev["val_acc"]
    dl = new["loo_acc"] - prev["loo_acc"]
    if dv < -EPS or dl < -EPS:
        return False, f"regression (val {dv:+.4f}, loo {dl:+.4f})"
    if dv >= 0.25 - EPS or dl >= 0.0625 - EPS:
        return True, f"improvement (val {dv:+.4f}, loo {dl:+.4f})"
    mv, ml = new["val_margin"] - prev["val_margin"], new["loo_margin"] - prev["loo_margin"]
    if abs(dv) < EPS and abs(dl) < EPS and mv >= 0.25 and ml >= -0.10:
        return True, f"margin improvement only (val margin {mv:+.3f}, loo margin {ml:+.3f})"
    return False, f"no meaningful improvement (val {dv:+.4f}, loo {dl:+.4f}, margins {mv:+.3f}/{ml:+.3f})"


def unlogged_changes(diff: dict, operations: list[dict]) -> list[str]:
    """Runtime tripwire: every changed tensor must be covered by recorded WeightProgrammer operations."""
    logged: dict[str, int] = {}
    for op in operations:
        logged[op["parameter"]] = logged.get(op["parameter"], 0) + op["n_changed"]
    return [f"unlogged parameter change in {name} ({info['changed']} changed vs {logged.get(name, 0)} logged)"
            for name, info in diff.items() if info["changed"] > logged.get(name, 0)]


def run_iteration(spec: IterationSpec) -> dict:
    state = read_json(STATE, {})
    if spec.id in state["completed_ids"]:
        log(f"skip completed {spec.id}")
        return state
    prev_model, _ = restore_checkpoint(state["accepted_checkpoint"])
    if state_hash(prev_model.state_dict()) != state["accepted_sha256"]:
        raise RuntimeError("accepted checkpoint does not match recorded hash")
    prev_sd = {k: v.clone() for k, v in prev_model.state_dict().items()}
    shapes = {n: list(p.shape) for n, p in prev_model.named_parameters()}
    record = {"id": spec.id, "title": spec.title, "status": "in_progress",
              "previous_checkpoint": state["accepted_checkpoint"], "observed": spec.observed,
              "problem": spec.problem, "hypothesis": spec.hypothesis, "selected": spec.selected,
              "expected": spec.expected, "risks": spec.risks, "started": time.strftime("%H:%M:%S")}
    atomic_write(CURRENT, json.dumps(record, indent=2))
    log(f"{spec.id} start: {spec.title}")

    model, _ = restore_checkpoint(state["accepted_checkpoint"])
    ctx = EditContext(WeightProgrammer(model, spec.id), dict(state["head_cfg"]))
    spec.edit(ctx)
    problems = validate_parameter_changes(model, shapes)
    diff = compare_parameters(prev_sd, model.state_dict())
    problems += unlogged_changes(diff, ctx.programmer.operations)
    score = score_model(model, ctx.head_cfg) if not problems else None
    new_metrics = headline(score) if score else {}
    ok, reason = (False, f"invalid parameters: {problems}") if problems else decide(state["metrics"], new_metrics)

    ckpt = save_checkpoint(model, spec.id, {"iteration": spec.id, "head_cfg": ctx.head_cfg,
                                            "accepted": ok, "parent": state["accepted_checkpoint"]})
    DIFF_DIR.mkdir(exist_ok=True)
    changed = {k: {"index": differs(prev_sd[k], model.state_dict()[k]).nonzero(),
                   "old": prev_sd[k][differs(prev_sd[k], model.state_dict()[k])],
                   "new": model.state_dict()[k][differs(prev_sd[k], model.state_dict()[k])]} for k in diff}
    torch.save(changed, DIFF_DIR / f"{spec.id}.pt")

    entry = {**record, "status": "completed", "decision": "accepted" if ok else "rejected",
             "decision_reason": reason, "previous_metrics": state["metrics"],
             "new_metrics": new_metrics, "head_cfg": ctx.head_cfg,
             "parameter_diff": diff, "operations": ctx.programmer.operations,
             "candidate_checkpoint": ckpt, "full_diff_file": str(DIFF_DIR / f"{spec.id}.pt"),
             "notes": ctx.notes,
             "n_changed_parameters": sum(v["changed"] for v in diff.values()),
             **wrong_summary(state, score)}
    persist(state, entry, ok, ckpt, ctx.head_cfg, new_metrics)
    log(f"{spec.id} {entry['decision']}: {reason}")
    return read_json(STATE, {})


def wrong_summary(state: dict, score: dict | None) -> dict:
    if IS_V2:
        from evaluator import split_data
        n_val = len(split_data("val")[1])
        return {"prev_n_wrong_val": round((1 - state["metrics"]["val_acc"]) * n_val),
                "new_n_wrong_val": score["val"]["n_wrong"] if score else None,
                "new_n_wrong_cv": score["cv"]["n_wrong"] if score else None}
    prev = score_prev(state)
    return {"prev_wrong_val": [r["name"] for r in prev["val"]["per_image"] if not r["correct"]],
            "new_wrong_val": [r["name"] for r in score["val"]["per_image"] if not r["correct"]] if score else [],
            "new_wrong_loo": score["loo"]["wrong"] if score else []}


def score_prev(state: dict) -> dict:
    model, _ = restore_checkpoint(state["accepted_checkpoint"])
    return score_model(model, state["head_cfg"])


def persist(state: dict, entry: dict, ok: bool, ckpt: dict, head_cfg: dict, metrics: dict) -> None:
    changes = read_json(CHANGES, [])
    changes.append({k: entry[k] for k in ("id", "hypothesis", "operations", "parameter_diff",
                                          "n_changed_parameters", "decision", "full_diff_file")})
    atomic_write(CHANGES, json.dumps(changes, indent=2))
    history = read_json(HISTORY, [])
    history.append(entry)
    atomic_write(HISTORY, json.dumps(history, indent=2))
    with DECISIONS.open("a") as handle:
        handle.write(f"\n## {entry['id']}: {entry['title']} -> {entry['decision'].upper()}\n\n"
                     f"- Observed: {entry['observed']}\n- Hypothesis: {entry['hypothesis']}\n"
                     f"- Parameters: {entry['selected']} ({entry['n_changed_parameters']} changed)\n"
                     f"- Before: {entry['previous_metrics']}\n- After: {entry['new_metrics']}\n"
                     f"- Reason: {entry['decision_reason']}\n")
    state["completed_ids"].append(entry["id"])
    state["last_completed"] = entry["id"]
    state["iteration"] = len(state["completed_ids"]) - 1
    (state["accepted_ids"] if ok else state["rejected_ids"]).append(entry["id"])
    if ok:
        state.update({"accepted_checkpoint": ckpt["path"], "accepted_sha256": ckpt["sha256"],
                      "head_cfg": head_cfg, "metrics": metrics})
    state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    atomic_write(STATE, json.dumps(state, indent=2))
    atomic_write(CURRENT, json.dumps({"id": entry["id"], "status": "completed",
                                      "decision": entry["decision"]}, indent=2))
    write_progress(state)


def write_progress(state: dict) -> None:
    history = read_json(HISTORY, [])
    rows = "\n".join(f"| {h['id']} | {h.get('title','')} | {h.get('decision','')} | "
                     f"{h.get('new_metrics', h.get('metrics', {}))} |" for h in history)
    atomic_write(PROGRESS, f"# WSP progress\n\n- Updated: {state['updated']}\n"
                 f"- Status: {state['status']}\n- Last completed: {state['last_completed']}\n"
                 f"- Accepted checkpoint: `{state['accepted_checkpoint']}`\n"
                 f"- Current metrics: {state['metrics']}\n- Baseline metrics: {state['baseline_metrics']}\n"
                 f"- Accepted: {state['accepted_ids']}\n- Rejected: {state['rejected_ids']}\n"
                 f"- Next action: {state['next_action']}\n\n| id | title | decision | metrics |\n"
                 f"|---|---|---|---|\n{rows}\n")
