"""PyTorch models used by the experiments. All are deliberately small: n=1, a few thousand days."""

from __future__ import annotations

import math

import torch
from torch import nn


class LinearRegressor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class LogisticRegression(nn.Module):
    """Returns logits; train with BCEWithLogitsLoss."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (64, 32), dropout: float = 0.2, out_dim: int = 1):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x):
        out = self.net(x)
        return out.squeeze(-1) if self.out_dim == 1 else out


class RNNForecaster(nn.Module):
    """GRU/LSTM over a (batch, window, features) sequence; predicts one value from the last state."""

    def __init__(self, in_dim: int, hidden: int = 32, layers: int = 1, kind: str = "gru", dropout: float = 0.1):
        super().__init__()
        rnn = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = rnn(in_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class TransformerForecaster(nn.Module):
    """Tiny Transformer encoder with learned positions; reads out the last time step."""

    def __init__(
        self, in_dim: int, window: int, d_model: int = 32, nhead: int = 4, layers: int = 2, dropout: float = 0.1
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, window, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        h = self.encoder(h)
        return self.head(h[:, -1]).squeeze(-1)


class MaskedAutoencoder(nn.Module):
    """Dense autoencoder for daily vectors with missing values.

    Input is the z-scored vector (NaN -> 0) concatenated with its observation mask; the
    loss is computed on observed entries only, so a missing SpO2 in 2019 is not an anomaly.
    """

    def __init__(self, in_dim: int, latent: int = 4, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2 * in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, latent)
        )
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, in_dim))

    def encode(self, x, mask):
        return self.encoder(torch.cat([x * mask, mask], dim=-1))

    def forward(self, x, mask):
        return self.decoder(self.encode(x, mask))


class ConvAutoencoder1D(nn.Module):
    """1D convolutional autoencoder for fixed-length heartbeat segments (batch, 1, length)."""

    def __init__(self, length: int = 256, channels: tuple[int, ...] = (16, 32, 32), latent: int = 16):
        super().__init__()
        assert length % (2 ** len(channels)) == 0
        enc, c_in = [], 1
        for c in channels:
            enc += [nn.Conv1d(c_in, c, 7, stride=2, padding=3), nn.GELU()]
            c_in = c
        self.encoder = nn.Sequential(*enc)
        self.reduced = length // (2 ** len(channels))
        self.to_latent = nn.Linear(c_in * self.reduced, latent)
        self.from_latent = nn.Linear(latent, c_in * self.reduced)
        dec = []
        rev = [*list(channels[::-1]), channels[0]]
        for i in range(len(channels)):
            dec += [nn.ConvTranspose1d(rev[i], rev[i + 1], 4, stride=2, padding=1), nn.GELU()]
        dec.append(nn.Conv1d(channels[0], 1, 7, padding=3))
        self.decoder = nn.Sequential(*dec)
        self.c_last = c_in

    def encode(self, x):
        h = self.encoder(x)
        return self.to_latent(h.flatten(1))

    def forward(self, x):
        z = self.encode(x)
        h = self.from_latent(z).view(-1, self.c_last, self.reduced)
        return self.decoder(h)


def kmeans(
    x: torch.Tensor, k: int, n_init: int = 10, max_iter: int = 300, tol: float = 1e-6, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Lloyd's k-means with k-means++ init in torch. Returns (centers, labels, inertia) of the best init."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = x.float()
    n = x.shape[0]
    best = (None, None, math.inf)
    for _ in range(n_init):
        idx = torch.randint(n, (1,), generator=g)
        centers = x[idx]
        for _ in range(1, k):
            d2 = torch.cdist(x, centers).min(dim=1).values ** 2
            probs = (d2 / d2.sum()).cpu()
            nxt = torch.multinomial(probs, 1, generator=g)
            centers = torch.cat([centers, x[nxt.to(x.device)]], dim=0)
        prev = math.inf
        for _ in range(max_iter):
            dist = torch.cdist(x, centers)
            labels = dist.argmin(dim=1)
            inertia = (dist.gather(1, labels[:, None]) ** 2).sum().item()
            new = torch.stack([x[labels == j].mean(0) if (labels == j).any() else centers[j] for j in range(k)])
            shift = (new - centers).norm(dim=1).max().item()
            centers = new
            if shift < tol or abs(prev - inertia) < tol * max(1.0, prev):
                break
            prev = inertia
        dist = torch.cdist(x, centers)
        labels = dist.argmin(dim=1)
        inertia = (dist.gather(1, labels[:, None]) ** 2).sum().item()
        if inertia < best[2]:
            best = (centers, labels, inertia)
    return best


def pca(x: torch.Tensor, n_components: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PCA via SVD. Returns (scores, components, explained_variance_ratio)."""
    xc = x - x.mean(0, keepdim=True)
    _, s, vh = torch.linalg.svd(xc, full_matrices=False)
    var = s**2 / (x.shape[0] - 1)
    comps = vh[:n_components]
    return xc @ comps.T, comps, var[:n_components] / var.sum()
