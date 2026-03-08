"""Shared trunk MLP for multi-model confidence prediction, with training.

The trunk takes concatenated per-model PCA features and outputs P(correct)
for every target model simultaneously.  Training uses an ensemble of
seeds with early stopping; inference averages sigmoid outputs.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

RANDOM_STATE = 42

DEFAULT_TRUNK_HIDDEN = (256, 128)
DEFAULT_DROPOUT = (0.3, 0.2)
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 150
DEFAULT_BATCH_SIZE = 512
DEFAULT_PATIENCE = 15
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_TRAIN_VAL_SPLIT = 0.85


class SharedTrunkNet(nn.Module):
    """Multi-output trunk: concatenated features -> P(correct) per model."""

    def __init__(
        self,
        d_in: int,
        n_outputs: int,
        hidden: tuple[int, ...] = DEFAULT_TRUNK_HIDDEN,
        dropout: tuple[float, ...] = DEFAULT_DROPOUT,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = d_in
        for i, h in enumerate(hidden):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if i < len(dropout):
                layers.append(nn.Dropout(dropout[i]))
            prev = h
        layers.append(nn.Linear(prev, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def reconstruct_trunk(ckpt: dict, device: str = "cpu") -> list[SharedTrunkNet]:
    """Rebuild the ensemble of trunk nets from checkpoint state dicts."""
    tcfg = ckpt["trunk_config"]
    nets = []
    for sd in ckpt["shared_trunk"]:
        net = SharedTrunkNet(
            tcfg["d_in"], tcfg["n_outputs"], hidden=tuple(tcfg["hidden"]),
        )
        net.load_state_dict(sd)
        net = net.to(device).eval()
        nets.append(net)
    return nets


def predict_proba(
    nets: list[SharedTrunkNet] | SharedTrunkNet,
    X: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Run ensemble prediction and return averaged sigmoid probabilities."""
    Xt = torch.tensor(X, dtype=torch.float32, device=device)

    if isinstance(nets, (list, tuple)):
        preds = []
        for net in nets:
            net = net.to(device).eval()
            with torch.no_grad():
                preds.append(torch.sigmoid(net(Xt)).cpu().numpy())
        return np.mean(preds, axis=0)

    with torch.no_grad():
        return torch.sigmoid(nets.to(device)(Xt)).cpu().numpy()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_mlp(
    net: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    patience: int = DEFAULT_PATIENCE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    train_val_split: float = DEFAULT_TRAIN_VAL_SPLIT,
    device: str = "cpu",
    seed: int | None = None,
) -> nn.Module:
    """Train with early stopping on a held-out validation split."""
    if seed is not None:
        torch.manual_seed(seed)
        net.apply(
            lambda m: m.reset_parameters()
            if hasattr(m, "reset_parameters") else None,
        )

    net = net.to(device)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)

    n = len(Xt)
    g = torch.Generator().manual_seed(seed if seed is not None else RANDOM_STATE)
    idx = torch.randperm(n, generator=g)
    n_tr = int(n * train_val_split)
    tr_idx, val_idx = idx[:n_tr], idx[n_tr:]

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, wait = float("inf"), None, 0

    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n_tr)
        for i in range(0, n_tr, batch_size):
            batch = tr_idx[perm[i: i + batch_size]]
            logits = net(Xt[batch])
            loss = loss_fn(logits, yt[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            vl = loss_fn(net(Xt[val_idx]), yt[val_idx]).item()
        if vl < best_val - 1e-4:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    net._best_val_loss = best_val  # type: ignore[attr-defined]
    return net


def train_ensemble(
    net_factory,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_seeds: int = 10,
    n_keep: int = 5,
    device: str = "cpu",
    **train_kwargs,
) -> list[nn.Module]:
    """Train n_seeds models, keep the top n_keep by validation loss."""
    seeds = [RANDOM_STATE + i for i in range(n_seeds)]
    results: list[tuple[float, int, nn.Module]] = []
    for s in seeds:
        net = net_factory()
        net = train_mlp(net, X, y, device=device, seed=s, **train_kwargs)
        results.append((net._best_val_loss, s, net))  # type: ignore[attr-defined]
    results.sort(key=lambda r: r[0])
    return [r[2] for r in results[:n_keep]]
