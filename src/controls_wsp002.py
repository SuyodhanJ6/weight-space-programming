"""WSP-002 null controls on TRAIN/VAL only (never test). Evaluation only; no gradients."""

from __future__ import annotations

import json
import math
import os
import statistics

os.environ.setdefault("WSP_EXPERIMENT", "wsp-002")

import torch  # noqa: E402

import paths  # noqa: E402
from evaluator import evaluate_split, forward_all, split_data  # noqa: E402
from init_weights import head_from_stats, program_head  # noqa: E402
from model import CatDogNet  # noqa: E402
from weight_programmer import restore_checkpoint  # noqa: E402

N_SEEDS = 20
N_SHUFFLES = 200


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def mcnemar_exact(b: int, c: int) -> float:
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def val_correct(model: CatDogNet) -> torch.Tensor:
    vx, vy, _ = split_data("val")
    return (forward_all(model, vx)["logit"] > 0).long() == vy


def random_feature_net(seed: int, rule: str = "lda", widths: tuple = (16, 32, 32)) -> CatDogNet:
    torch.manual_seed(seed)
    model = CatDogNet(widths)
    for conv in (model.conv1, model.conv2, model.conv3):
        torch.nn.init.kaiming_uniform_(conv.weight)
    tx, ty, _ = split_data("train")
    program_head(model, tx, ty, rule)
    return model


def main() -> None:
    state = json.loads((paths.EXP / "experiment_state.json").read_text())
    base, _ = restore_checkpoint(state["baseline_checkpoint"])
    final, _ = restore_checkpoint(state["accepted_checkpoint"])
    vx, vy, _ = split_data("val")
    cb, cf = val_correct(base), val_correct(final)
    b, c = int((cb & ~cf).sum()), int((~cb & cf).sum())        # baseline-only right, final-only right
    out = {"n_val": len(vy),
           "baseline_val_acc": round(float(cb.float().mean()), 4),
           "baseline_val_ci95": wilson(int(cb.sum()), len(vy)),
           "final_val_acc": round(float(cf.float().mean()), 4),
           "final_val_ci95": wilson(int(cf.sum()), len(vy)),
           "paired_final_right_baseline_wrong": c, "paired_baseline_right_final_wrong": b,
           "mcnemar_exact_p_two_sided": round(mcnemar_exact(b, c), 5)}
    rule = state["head_cfg"]["rule"]
    tx0, ty0, _ = split_data("train")
    def with_head(model: CatDogNet, head_rule: str) -> float:
        clone = CatDogNet(model.widths); clone.load_state_dict(model.state_dict())
        program_head(clone, tx0, ty0, head_rule, state["head_cfg"].get("cell_mask") if head_rule == rule else None)
        return round(float(val_correct(clone).float().mean()), 4)
    # attribution: head estimator vs extractor edits (all fitted on TRAIN, scored on VAL)
    out["attribution_val_acc"] = {
        "baseline_extractor + diagonal LDA head": with_head(base, "lda"),
        f"baseline_extractor + {rule} head": with_head(base, rule),
        "final_extractor + diagonal LDA head": with_head(final, "lda"),
        f"final_extractor + {rule} head (= final model)": with_head(final, rule)}
    n_seeds = N_SEEDS if tuple(base.widths) == (16, 32, 32) else 6   # wide nets are ~6x slower
    rand = [float(val_correct(random_feature_net(s, rule, tuple(base.widths))).float().mean()) for s in range(n_seeds)]
    out["random_conv_features_same_head_val_acc"] = {"mean": round(statistics.mean(rand), 4),
        "std": round(statistics.pstdev(rand), 4), "min": round(min(rand), 4), "max": round(max(rand), 4)}
    out["frac_random_ge_baseline"] = sum(x >= out["baseline_val_acc"] for x in rand) / n_seeds
    out["frac_random_ge_final"] = sum(x >= out["final_val_acc"] for x in rand) / n_seeds
    tx, ty, _ = split_data("train")
    ft = forward_all(final, tx)["gap"]; fv = forward_all(final, vx)["gap"]
    gen = torch.Generator().manual_seed(0)
    shuf = []
    for _ in range(N_SHUFFLES):
        perm = ty[torch.randperm(len(ty), generator=gen)]
        w, bias = head_from_stats(ft, perm, state["head_cfg"]["rule"], state["head_cfg"].get("cell_mask"))
        shuf.append(float(((fv @ w + bias > 0).long() == vy).float().mean()))
    out["label_shuffled_head_on_final_extractor_val_acc"] = {"mean": round(statistics.mean(shuf), 4),
        "std": round(statistics.pstdev(shuf), 4), "max": round(max(shuf), 4)}
    out["frac_shuffle_ge_final"] = sum(x >= out["final_val_acc"] for x in shuf) / N_SHUFFLES
    (paths.EXP / "controls.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
