"""A small, generic training loop with early stopping on a validation loss."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .config import SEED, get_device, set_seed


@dataclass
class FitResult:
    best_epoch: int
    best_val_loss: float
    history: list[tuple[float, float]] = field(default_factory=list)


def to_tensors(*arrays, device=None) -> list[torch.Tensor]:
    device = device or get_device()
    return [torch.as_tensor(np.ascontiguousarray(a, dtype=np.float32), device=device) for a in arrays]


def fit(
    model: nn.Module,
    loss_fn: Callable[[nn.Module, tuple[torch.Tensor, ...]], torch.Tensor],
    train: tuple[torch.Tensor, ...],
    val: tuple[torch.Tensor, ...],
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 128,
    patience: int = 25,
    seed: int = SEED,
    keep_best: bool = True,
) -> FitResult:
    """Mini-batch AdamW; keeps the weights of the best validation epoch.

    ``train``/``val`` are tuples of tensors that share the first dimension; ``loss_fn``
    receives the model and one batch tuple. With ``keep_best=False`` the loop runs all
    ``epochs`` and keeps the final weights (used to refit on train+val for a fixed epoch count).
    """
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = train[0].shape[0]
    g = torch.Generator().manual_seed(seed)
    best_state, best_loss, best_epoch, bad = None, float("inf"), 0, 0
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        tr_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size].to(train[0].device)
            batch = tuple(t[idx] for t in train)
            opt.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            va_loss = loss_fn(model, val).item()
        history.append((tr_loss / n, va_loss))
        if va_loss < best_loss - 1e-6:
            best_loss, best_epoch, bad = va_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        elif keep_best:
            bad += 1
            if bad >= patience:
                break
    if keep_best and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return FitResult(best_epoch, best_loss, history)


def mse_loss(model, batch):
    x, y = batch
    return nn.functional.mse_loss(model(x), y)


def huber_loss(model, batch):
    x, y = batch
    return nn.functional.huber_loss(model(x), y, delta=1.0)


def bce_loss(model, batch):
    x, y = batch
    return nn.functional.binary_cross_entropy_with_logits(model(x), y)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = [model(x[i : i + batch_size]).float().cpu() for i in range(0, x.shape[0], batch_size)]
    return torch.cat(out).numpy()
