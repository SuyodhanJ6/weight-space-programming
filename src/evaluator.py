"""Metrics and held-out diagnostics. Loss is reported as a metric only."""

from __future__ import annotations

import torch

from data import load_split
from init_weights import head_from_stats
from model import CatDogNet
import paths
from paths import IS_V2

_CACHE: dict[str, tuple] = {}


def split_data(split: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    if split not in _CACHE:
        _CACHE[split] = load_split(split)
    return _CACHE[split]


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    preds = (logits > 0).long()
    tp = int(((preds == 1) & (labels == 1)).sum()); fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum()); fn = int(((preds == 0) & (labels == 1)).sum())
    precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    sign = labels.float() * 2 - 1
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
    return {"n": len(labels), "accuracy": (tp + tn) / len(labels), "precision": precision,
            "recall": recall, "f1": _safe_div(2 * precision * recall, precision + recall),
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "bce_loss_metric": float(bce), "mean_signed_margin": float((logits * sign).mean())}


CHUNK = 250


def forward_all(model: CatDogNet, images: torch.Tensor) -> dict[str, torch.Tensor]:
    """Forward pass; chunked only for large sets (wsp-001's 4-12 images are unchanged)."""
    if len(images) <= CHUNK:
        return model.forward_with_activations(images)
    parts = [model.forward_with_activations(images[i:i + CHUNK]) for i in range(0, len(images), CHUNK)]
    return {k: torch.cat([p[k] for p in parts]) for k in ("gap", "logit", "prob_dog")}


def evaluate_split(model: CatDogNet, split: str, acts: dict | None = None) -> dict:
    images, labels, names = split_data(split)
    acts = acts if acts is not None else forward_all(model, images)
    result = metrics_from_logits(acts["logit"], labels)
    if IS_V2:  # keep history files small: counts instead of 600+ rows
        result["n_wrong"] = int((acts["logit"] > 0).long().ne(labels).sum())
        return result
    result["per_image"] = [{"name": n, "label": int(l), "logit": round(float(z), 4),
                            "p_dog": round(float(p), 4), "correct": bool((z > 0) == bool(l))}
                           for n, l, z, p in zip(names, labels, acts["logit"], acts["prob_dog"])]
    return result


def loo_head_diagnostic(model: CatDogNet, head_cfg: dict) -> dict:
    """Leave-one-out over train+val (16 images): refit the head rule on 15, predict 1.

    Measures whether the CURRENT feature extractor + head rule transfers to an
    image not used to set the head. Closed-form refits only; no gradients.
    """
    tx, ty, tn = split_data("train")
    vx, vy, vn = split_data("val")
    images, labels, names = torch.cat([tx, vx]), torch.cat([ty, vy]), tn + vn
    feats = model.forward_with_activations(images)["gap"]
    logits = []
    for i in range(len(names)):
        keep = torch.arange(len(names)) != i
        w, b = head_from_stats(feats[keep], labels[keep], head_cfg["rule"], head_cfg.get("cell_mask"))
        logits.append(feats[i] @ w + b)
    result = metrics_from_logits(torch.stack(logits), labels)
    result["wrong"] = [n for n, ok in zip(names, ((torch.stack(logits) > 0).long() == labels).tolist())
                       if not ok]
    return result


CV_FOLDS = 5


def cv_head_diagnostic(model: CatDogNet, head_cfg: dict, feats: torch.Tensor | None = None) -> dict:
    """wsp-002: K-fold over TRAIN only. Refit the head rule on K-1 folds, predict the held-out fold."""
    images, labels, _ = split_data("train")
    feats = feats if feats is not None else forward_all(model, images)["gap"]
    fold = torch.arange(len(labels)) % CV_FOLDS
    logits = torch.empty(len(labels))
    for k in range(CV_FOLDS):
        keep = fold != k
        w, b = head_from_stats(feats[keep], labels[keep], head_cfg["rule"], head_cfg.get("cell_mask"))
        logits[~keep] = feats[~keep] @ w + b
    result = metrics_from_logits(logits, labels)
    result["n_wrong"] = int((logits > 0).long().ne(labels).sum())
    return result


def score_model(model: CatDogNet, head_cfg: dict) -> dict:
    """train + val (+ LOO / CV diagnostic). Never touches the test split."""
    if IS_V2:
        train_acts = forward_all(model, split_data("train")[0])
        out = {"train": evaluate_split(model, "train", train_acts), "val": evaluate_split(model, "val"),
               "cv": cv_head_diagnostic(model, head_cfg, train_acts["gap"])}
        if paths.LOO_RULE:
            out["loo"] = loo_head_diagnostic(model, head_cfg)
        return out
    return {"train": evaluate_split(model, "train"), "val": evaluate_split(model, "val"),
            "loo": loo_head_diagnostic(model, head_cfg)}


def headline(score: dict) -> dict:
    if IS_V2:
        extra = ({"loo_acc": round(score["loo"]["accuracy"], 4), "loo_f1": round(score["loo"]["f1"], 4),
                  "loo_margin": round(score["loo"]["mean_signed_margin"], 4)} if "loo" in score else {})
        return {**extra, "train_acc": round(score["train"]["accuracy"], 4),
                "val_acc": round(score["val"]["accuracy"], 4),
                "val_f1": round(score["val"]["f1"], 4),
                "val_margin": round(score["val"]["mean_signed_margin"], 4),
                "cv_acc": round(score["cv"]["accuracy"], 4),
                "cv_margin": round(score["cv"]["mean_signed_margin"], 4)}
    return {"train_acc": round(score["train"]["accuracy"], 4),
            "val_acc": round(score["val"]["accuracy"], 4),
            "val_margin": round(score["val"]["mean_signed_margin"], 4),
            "loo_acc": round(score["loo"]["accuracy"], 4),
            "loo_margin": round(score["loo"]["mean_signed_margin"], 4)}
