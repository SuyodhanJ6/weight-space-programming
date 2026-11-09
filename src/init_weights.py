"""Hand-programmed parameter initialisation (no learning, no iteration).

conv1: analytic filters (oriented Gabor, centre-surround, colour opponent).
conv2/conv3: hand-composed from lower-layer channel groups.
head: one-shot closed-form class-mean statistic from the TRAIN split only.
"""

from __future__ import annotations

import math

import torch

from model import CatDogNet

NUM_ORIENTATIONS = 8
HEAD_LOGIT_STD = 2.0   # rescale so train logits have a readable spread
VAR_EPS = 1e-3
LUMA = torch.tensor([0.299, 0.587, 0.114])


def _grid(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    r = torch.arange(size, dtype=torch.float32) - size // 2
    return torch.meshgrid(r, r, indexing="xy")


def _l1(kernel: torch.Tensor) -> torch.Tensor:
    return kernel / kernel.abs().sum().clamp_min(1e-8)


def _gabor(theta: float, sigma: float = 1.5, wavelength: float = 4.0) -> torch.Tensor:
    x, y = _grid(5)
    xr = x * math.cos(theta) + y * math.sin(theta)
    env = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    k = env * torch.cos(2 * math.pi * xr / wavelength)
    return _l1(k - k.mean())


def _dog(s_in: float, s_out: float, polarity: float) -> torch.Tensor:
    x, y = _grid(5)
    r2 = x**2 + y**2
    k = torch.exp(-r2 / (2 * s_in**2)) / s_in**2 - torch.exp(-r2 / (2 * s_out**2)) / s_out**2
    return polarity * _l1(k - k.mean())


def _blur(sigma: float = 1.5) -> torch.Tensor:
    x, y = _grid(5)
    return _l1(torch.exp(-(x**2 + y**2) / (2 * sigma**2)))


def program_conv1(model: CatDogNet) -> None:
    w = torch.zeros(16, 3, 5, 5)
    for k in range(NUM_ORIENTATIONS):                       # 0-7 oriented edges
        theta = math.pi * k / NUM_ORIENTATIONS
        w[k] = LUMA.view(3, 1, 1) * _gabor(theta)
    dogs = [(1.0, 2.0, 1), (1.0, 2.0, -1), (0.8, 1.6, 1), (0.8, 1.6, -1)]
    for i, (s_in, s_out, pol) in enumerate(dogs):           # 8-11 centre-surround
        w[8 + i] = LUMA.view(3, 1, 1) * _dog(s_in, s_out, pol)
    opponents = [torch.tensor([1.0, -1.0, 0.0]), torch.tensor([-0.5, -0.5, 1.0])]
    for i, vec in enumerate(opponents):                     # 12-15 colour opponent
        w[12 + 2 * i] = vec.view(3, 1, 1) * _blur()
        w[13 + 2 * i] = -vec.view(3, 1, 1) * _blur()
    model.conv1.weight.copy_(w)


def _line_offsets(k: int) -> list[tuple[int, int]]:
    theta = math.pi * k / NUM_ORIENTATIONS
    dx, dy = -math.sin(theta), math.cos(theta)  # along the edge (perp. to filter axis)
    return [(round(t * dy) + 1, round(t * dx) + 1) for t in (-1, 0, 1)]


def program_conv2(model: CatDogNet) -> None:
    """32 filters over 16 channels. Types: line, corner, parallel pair, blob+edge."""
    n = NUM_ORIENTATIONS
    w = torch.zeros(32, 16, 3, 3)
    for k in range(n):
        for r, c in _line_offsets(k):                       # 0-7 line continuation
            w[k, k, r, c] += 1.0
        for nb in ((k + 1) % n, (k - 1) % n):
            w[k, nb, 1, 1] += 0.5
        w[n + k, k] += 1.0 / 9                              # 8-15 corner: k + (k+90 deg)
        w[n + k, (k + n // 2) % n] += 1.0 / 9
        theta = math.pi * k / n
        for sign in (-1, 1):                                # 16-23 parallel pair
            r, c = round(sign * math.sin(theta)) + 1, round(sign * math.cos(theta)) + 1
            w[2 * n + k, k, r, c] += 1.0
        w[2 * n + k, k, 1, 1] -= 0.5
        w[3 * n + k, 8 + k % 4, 1, 1] += 1.0                # 24-31 blob + edge
        w[3 * n + k, k] += 1.0 / 9
    for f in range(32):
        w[f] = _l1(w[f])
    model.conv2.weight.copy_(w)


def program_conv3(model: CatDogNet) -> None:
    """32 filters over 32 channels; spatial kernel is a centre-weighted 3x3."""
    n = NUM_ORIENTATIONS
    spatial = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16
    w = torch.zeros(32, 32, 3, 3)
    for k in range(n):
        for t in range(4):                                  # 0-7 all types, one orientation
            w[k, t * n + k] = spatial
        for ch in (k, (k + n // 2) % n):                    # 8-15 line, k and k+90 deg
            w[n + k, ch] = spatial
        for ch in (n + k, 2 * n + (k + 2) % n):             # 16-23 corner/parallel mix
            w[2 * n + k, ch] = spatial
        w[3 * n + k, 3 * n:] = spatial                      # 24-31 isotropic energy
    for f in range(32):
        w[f] = _l1(w[f])
    model.conv3.weight.copy_(w)


def head_from_stats(feats: torch.Tensor, labels: torch.Tensor, rule: str = "lda",
                    cell_mask: list[float] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form head weights (512,) and bias () from class statistics. No iteration.

    rule: "lda"        w = (mu_dog - mu_cat) / pooled_var
          "lda_demean" same, then subtract mean(w) (removes the global-energy direction)
          "centroid"   w = mu_dog - mu_cat (no variance scaling)
    cell_mask: 16 multipliers over the 4x4 pooled cells (flatten order is channel-major).
    """
    if rule == "lda_full":
        return _head_full_lda(feats, labels)
    if rule in ("lda_pooled", "lda_pooled_cb"):
        return _head_shared_cells(feats, labels, rule)
    cat, dog = feats[labels == 0], feats[labels == 1]
    mu_cat, mu_dog = cat.mean(0), dog.mean(0)
    pooled_var = 0.5 * (cat.var(0) + dog.var(0)) + VAR_EPS
    if rule == "centroid":
        w = mu_dog - mu_cat
    else:
        w = (mu_dog - mu_cat) / pooled_var
        if rule == "lda_demean":
            w = w - w.mean()
        elif rule != "lda":
            raise ValueError(f"unknown head rule {rule}")
    if cell_mask is not None:
        w = (w.view(-1, 16) * torch.tensor(cell_mask, dtype=w.dtype)).flatten()
    b = -(w * (mu_cat + mu_dog) / 2).sum()
    logits = feats @ w + b
    scale = HEAD_LOGIT_STD / logits.std().clamp_min(1e-8)
    return w * scale, b * scale


CENTRE_CELLS = (5, 6, 9, 10)
FULL_LDA_RIDGE = 0.05      # fixed shrinkage: lambda = 0.05 * mean feature variance (not tuned)


def _head_full_lda(feats: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-covariance LDA, closed form (one direct linear solve, no iteration, no gradients).

    w = (Sigma_within + ridge*I)^-1 (mu_dog - mu_cat). This is a data-fitted head (estimator), not a hand edit.
    """
    cat, dog = feats[labels == 0], feats[labels == 1]
    mu_cat, mu_dog = cat.mean(0), dog.mean(0)
    centred = torch.cat([cat - mu_cat, dog - mu_dog])
    sigma = centred.T @ centred / (len(centred) - 2)
    ridge = FULL_LDA_RIDGE * sigma.diagonal().mean()
    w = torch.linalg.solve(sigma + ridge * torch.eye(sigma.shape[0]), mu_dog - mu_cat)
    b = -(w * (mu_cat + mu_dog) / 2).sum()
    logits = feats @ w + b
    scale = HEAD_LOGIT_STD / logits.std().clamp_min(1e-8)
    return w * scale, b * scale


def _head_shared_cells(feats: torch.Tensor, labels: torch.Tensor, rule: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Spatially shared head: LDA over per-channel cell-averaged features.

    lda_pooled    : 32 features (mean over all 16 cells), weights tiled /16
    lda_pooled_cb : 64 features (mean over 4 centre cells, mean over 12 border cells)
    """
    f = feats.view(feats.shape[0], -1, 16)
    channels = f.shape[1]
    centre = torch.zeros(16, dtype=torch.bool)
    centre[list(CENTRE_CELLS)] = True
    groups = [torch.ones(16, dtype=torch.bool)] if rule == "lda_pooled" else [centre, ~centre]
    pooled = torch.cat([f[:, :, g].mean(2) for g in groups], dim=1)
    cat, dog = pooled[labels == 0], pooled[labels == 1]
    mu_cat, mu_dog = cat.mean(0), dog.mean(0)
    var = 0.5 * (cat.var(0) + dog.var(0)) + VAR_EPS
    w_small = (mu_dog - mu_cat) / var
    w = torch.zeros(channels, 16)
    for gi, g in enumerate(groups):
        w[:, g] = (w_small[gi * channels:(gi + 1) * channels] / g.sum()).unsqueeze(1)
    w = w.flatten()
    b = -(w_small * (mu_cat + mu_dog) / 2).sum()
    logits = feats @ w + b
    scale = HEAD_LOGIT_STD / logits.std().clamp_min(1e-8)
    return w * scale, b * scale


def program_head(model: CatDogNet, images: torch.Tensor, labels: torch.Tensor,
                 rule: str = "lda", cell_mask: list[float] | None = None) -> None:
    """Closed-form linear discriminant from TRAIN images only (one shot)."""
    feats = model.forward_with_activations(images)["gap"]
    w, b = head_from_stats(feats, labels, rule, cell_mask)
    model.head.weight.copy_(w.unsqueeze(0))
    model.head.bias.copy_(b.view(1))


WIDE_WIDTHS = (48, 96, 96)


def _odd_gabor(theta: float, sigma: float = 1.5, wavelength: float = 4.0) -> torch.Tensor:
    x, y = _grid(5)
    xr = x * math.cos(theta) + y * math.sin(theta)
    return _l1(torch.exp(-(x**2 + y**2) / (2 * sigma**2)) * torch.sin(2 * math.pi * xr / wavelength))


def program_conv1_wide(model: CatDogNet) -> None:
    """48 hand-written filters: even/odd Gabors x 8 orientations x 2 polarities (32), DoG (4),
    colour axes (4), diagonal hue sectors (4), coarse-frequency Gabors (4)."""
    w = torch.zeros(48, 3, 5, 5)
    luma = LUMA.view(3, 1, 1)
    for k in range(NUM_ORIENTATIONS):
        theta = math.pi * k / NUM_ORIENTATIONS
        w[k] = luma * _gabor(theta)
        w[8 + k] = -luma * _gabor(theta)
        w[16 + k] = luma * _odd_gabor(theta)
        w[24 + k] = -luma * _odd_gabor(theta)
    for i, (s_in, s_out, pol) in enumerate([(1.0, 2.0, 1), (1.0, 2.0, -1), (0.8, 1.6, 1), (0.8, 1.6, -1)]):
        w[32 + i] = luma * _dog(s_in, s_out, pol)
    rg, by = torch.tensor([1.0, -1.0, 0.0]), torch.tensor([-0.5, -0.5, 1.0])
    axes = [rg, -rg, by, -by]
    diagonals = [rg - by, -rg - by, -rg + by, rg + by]
    for i, vec in enumerate(axes):
        w[36 + i] = vec.view(3, 1, 1) * _blur()
    for i, vec in enumerate(diagonals):
        w[40 + i] = vec.view(3, 1, 1) * _blur()
    for i, k in enumerate((0, 2, 4, 6)):
        w[44 + i] = luma * _gabor(math.pi * k / NUM_ORIENTATIONS, sigma=2.0, wavelength=6.0)
    model.conv1.weight.copy_(w)


def build_initial_model_wide(train_images: torch.Tensor, train_labels: torch.Tensor) -> CatDogNet:
    """wsp-004 baseline: 48-filter conv1; conv2 = fine passthrough (48) + 3x3-box coarse (48);
    conv3 = identity with blur (96); head = closed-form full-covariance LDA from train."""
    model = CatDogNet(WIDE_WIDTHS)
    program_conv1_wide(model)
    w2 = torch.zeros(96, 48, 3, 3)
    w3 = torch.zeros(96, 96, 3, 3)
    blur = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16
    for j in range(48):
        w2[j, j, 1, 1] = 1.0
        w2[48 + j, j] = 1.0 / 9
    for c in range(96):
        w3[c, c] = blur
    model.conv2.weight.copy_(w2)
    model.conv3.weight.copy_(w3)
    program_head(model, train_images, train_labels, "lda_full")
    return model


def build_baseline_for(experiment: str, train_images: torch.Tensor, train_labels: torch.Tensor) -> CatDogNet:
    if experiment == "wsp-004":
        return build_initial_model_wide(train_images, train_labels)
    return build_initial_model(train_images, train_labels)


def build_initial_model(train_images: torch.Tensor, train_labels: torch.Tensor) -> CatDogNet:
    model = CatDogNet()
    program_conv1(model)
    program_conv2(model)
    program_conv3(model)
    program_head(model, train_images, train_labels)
    return model
