"""Null controls (evaluation only; no gradients, train/val/LOO only, never test).

1. random-feature nets: random conv weights + the SAME closed-form head rule.
2. label-shuffle null for the leave-one-out diagnostic.
Answers: is the hand-programmed extractor better than random features + a fitted head?
"""

from __future__ import annotations

import json
import random
import statistics

import torch

from evaluator import evaluate_split, loo_head_diagnostic, split_data
from init_weights import program_head
from model import CatDogNet
from weight_programmer import restore_checkpoint
import evaluator

N_SEEDS = 40


def random_feature_net(seed: int) -> CatDogNet:
    torch.manual_seed(seed)
    model = CatDogNet()
    for conv in (model.conv1, model.conv2, model.conv3):
        torch.nn.init.kaiming_uniform_(conv.weight)
    tx, ty, _ = split_data("train")
    program_head(model, tx, ty, "lda")
    return model


def shuffled_loo(model: CatDogNet, cfg: dict, seed: int) -> float:
    rng = random.Random(seed)
    orig = evaluator.split_data
    tx, ty, tn = orig("train"); vx, vy, vn = orig("val")
    labels = torch.cat([ty, vy]).tolist(); rng.shuffle(labels)
    cache = {"train": (tx, torch.tensor(labels[:len(tn)]), tn),
             "val": (vx, torch.tensor(labels[len(tn):]), vn)}
    evaluator.split_data = lambda s: cache[s]
    try:
        return loo_head_diagnostic(model, cfg)["accuracy"]
    finally:
        evaluator.split_data = orig


def summarize(xs: list[float]) -> dict:
    return {"mean": round(statistics.mean(xs), 4), "std": round(statistics.pstdev(xs), 4),
            "min": min(xs), "max": max(xs)}


def main() -> None:
    state = json.load(open("experiments/experiment_state.json"))
    cfg = state["head_cfg"]
    rand = [random_feature_net(s) for s in range(N_SEEDS)]
    rv = [evaluate_split(m, "val")["accuracy"] for m in rand]
    rl = [loo_head_diagnostic(m, {"rule": "lda"})["accuracy"] for m in rand]
    final, _ = restore_checkpoint(state["accepted_checkpoint"])
    base, _ = restore_checkpoint(state["baseline_checkpoint"])
    shuf = [shuffled_loo(final, cfg, s) for s in range(200)]
    out = {"n_random_seeds": N_SEEDS,
           "random_features_val_acc": summarize(rv), "random_features_loo_acc": summarize(rl),
           "frac_random_loo_ge_final": sum(x >= state["metrics"]["loo_acc"] for x in rl) / N_SEEDS,
           "frac_random_loo_ge_baseline": sum(x >= state["baseline_metrics"]["loo_acc"] for x in rl) / N_SEEDS,
           "label_shuffle_loo_acc_final_model": summarize(shuf),
           "frac_shuffle_ge_final_loo": sum(x >= state["metrics"]["loo_acc"] for x in shuf) / len(shuf),
           "final_metrics": state["metrics"], "baseline_metrics": state["baseline_metrics"]}
    open("experiments/controls.json", "w").write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
