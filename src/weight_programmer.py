"""Direct weight-space programming API. Every edit is explicit and recorded.

Parameters are edited in place under torch.no_grad(). Parameters here are
deliberately requires_grad=False (autograd is never used); an editable
parameter is any registered parameter that has no grad attached.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import torch

from model import DEFAULT_WIDTHS, CatDogNet
from paths import CHECKPOINT_DIR
SAMPLE_VALUES = 8


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(model: CatDogNet, name: str, meta: dict | None = None,
                    overwrite: bool = False) -> dict:
    path = CHECKPOINT_DIR / f"{name}.pt"
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "meta": {**(meta or {}), "widths": list(model.widths)}}, path)
    return {"path": str(path), "sha256": state_hash(state)}


def restore_checkpoint(path: str | Path) -> tuple[CatDogNet, dict]:
    blob = torch.load(path, weights_only=True)
    model = CatDogNet(tuple(blob.get("meta", {}).get("widths", DEFAULT_WIDTHS)))
    model.load_state_dict(blob["state_dict"])
    return model, blob.get("meta", {})


def inspect_parameters(model: CatDogNet) -> list[dict]:
    rows = []
    for name, p in model.named_parameters():
        rows.append({"name": name, "shape": list(p.shape), "count": p.numel(),
                     "mean": float(p.mean()), "std": float(p.std()) if p.numel() > 1 else 0.0,
                     "min": float(p.min()), "max": float(p.max()), "l2": float(p.norm()),
                     "frac_zero": float((p == 0).float().mean())})
    return rows


def compare_parameters(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> dict:
    out = {}
    for key in before:
        mask = before[key] != after[key]
        if mask.any():
            out[key] = {"changed": int(mask.sum()), "total": mask.numel(),
                        "max_abs_delta": float((after[key] - before[key]).abs().max()),
                        "l2_delta": float((after[key] - before[key]).norm())}
    return out


def validate_parameter_changes(model: CatDogNet, reference_shapes: dict[str, list[int]]) -> list[str]:
    """Return a list of problems (empty = valid)."""
    problems = []
    seen = {}
    for name, p in model.named_parameters():
        seen[name] = list(p.shape)
        if list(p.shape) != reference_shapes.get(name):
            problems.append(f"{name}: shape changed")
        if not torch.isfinite(p).all():
            problems.append(f"{name}: NaN/inf")
        if p.requires_grad or p.grad is not None:
            problems.append(f"{name}: autograd state present")
    if set(seen) != set(reference_shapes):
        problems.append("parameter set changed")
    return problems


class WeightProgrammer:
    """Edit interface bound to one model and one iteration; logs every operation."""

    def __init__(self, model: CatDogNet, iteration_id: str) -> None:
        self.model, self.iteration_id = model, iteration_id
        self.params = dict(model.named_parameters())
        self.operations: list[dict] = []

    def get_parameter(self, name: str) -> torch.Tensor:
        return self._param(name).detach().clone()

    def _param(self, name: str) -> torch.nn.Parameter:
        if name not in self.params:
            raise ValueError(f"unknown parameter {name}")
        return self.params[name]

    def _record(self, op: str, name: str, selector: str, reason: str,
                old: torch.Tensor, new: torch.Tensor) -> dict:
        changed = old != new
        entry = {"op": op, "parameter": name, "selector": selector, "reason": reason,
                 "n_changed": int(changed.sum()),
                 "sample_old": [round(float(v), 6) for v in old[changed].flatten()[:SAMPLE_VALUES]],
                 "sample_new": [round(float(v), 6) for v in new[changed].flatten()[:SAMPLE_VALUES]]}
        self.operations.append(entry)
        return entry

    def modify_parameter(self, name: str, index: tuple, new_value: float, reason: str) -> dict:
        p = self._param(name)
        if len(index) != p.ndim:
            raise ValueError("index rank mismatch")
        with torch.no_grad():
            old = p[index].clone().reshape(1)
            p[index] = new_value
            new = p[index].clone().reshape(1)
        return self._record("modify_parameter", name, str(list(index)), reason, old, new)

    def modify_parameter_group(self, name: str, selector: tuple, values: torch.Tensor | float,
                               reason: str) -> dict:
        """Assign `values` to p[selector] (slices/ints/index tensors)."""
        p = self._param(name)
        with torch.no_grad():
            old = p[selector].clone()
            p[selector] = values
            new = p[selector].clone()
        return self._record("modify_parameter_group", name, repr(selector), reason, old, new)

    def replace_parameter(self, name: str, values: torch.Tensor, reason: str) -> dict:
        p = self._param(name)
        if tuple(values.shape) != tuple(p.shape):
            raise ValueError("replacement shape mismatch")
        with torch.no_grad():
            old = p.clone()
            p.copy_(values)
            new = p.clone()
        return self._record("replace_parameter", name, "all", reason, old, new)

    def apply_parameter_transform(self, name: str, fn: Callable[[torch.Tensor], torch.Tensor],
                                  desc: str, reason: str, selector: tuple | None = None) -> dict:
        """p[selector] <- fn(p[selector]); `fn` must be an explicit closed-form map."""
        p = self._param(name)
        sel = selector if selector is not None else (slice(None),) * p.ndim
        with torch.no_grad():
            old = p[sel].clone()
            p[sel] = fn(old)
            new = p[sel].clone()
        return self._record("apply_parameter_transform", name, f"{desc} @ {sel!r}", reason, old, new)
