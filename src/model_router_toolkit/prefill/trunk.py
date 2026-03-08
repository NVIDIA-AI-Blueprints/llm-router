"""Shared trunk MLP for multi-model confidence prediction.

The trunk takes concatenated per-model features and outputs P(correct)
for every target model simultaneously.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SharedTrunkNet(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_outputs: int,
        hidden: tuple[int, ...] = (256, 128),
        dropout: tuple[float, ...] = (0.3, 0.2),
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
