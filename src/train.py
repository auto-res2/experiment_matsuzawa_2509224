import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------  SAFE-VALS core  ---------------------------------
class HashMLP(nn.Module):
    """Meta-learned hashing network (two-layer MLP, straight-through rounding)."""

    def __init__(self, d_in: int, d_hid: int = 32, buckets: int = 4096):
        super().__init__()
        self.buckets = buckets
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_in, d_hid), nn.ReLU(), nn.Linear(d_hid, 1)
        )

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x_i, x_j], dim=-1)
        logits = self.mlp(z)
        # Straight-through rounding to obtain bucket ids
        h = ((self.buckets * torch.sigmoid(logits)).round()).long()
        return h.squeeze(-1)


class CountMin:
    """Torch-only Count–Min sketch with pair-wise hashing."""

    def __init__(self, w: int = 4096, d: int = 4):
        self.w, self.d = w, d
        self.table = torch.zeros(d, w, dtype=torch.int32, device=_DEVICE)
        # Large random hash coefficients (FNV-like)
        self.a = torch.randint(1, 2 ** 31 - 1, (d,), device=_DEVICE)
        self.b = torch.randint(0, 2 ** 31 - 1, (d,), device=_DEVICE)

    def _idx(self, keys: torch.Tensor, row: int) -> torch.Tensor:
        return ((self.a[row] * keys + self.b[row]) % self.w).long()

    def add(self, keys: torch.Tensor):
        for r in range(self.d):
            idx = self._idx(keys, r)
            self.table[r, idx] += 1

    @torch.no_grad()
    def query(self, keys: torch.Tensor) -> torch.Tensor:
        est = []
        for r in range(self.d):
            idx = self._idx(keys, r)
            est.append(self.table[r, idx])
        return torch.stack(est).min(0).values.float()


class SAFEVALSLayer(nn.Module):
    """One SAFE-VALS sparse attention layer wrapping PyG's GATConv."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        heads: int,
        k: int,
        buckets: int,
        eps: float = 0.05,
        delta: float = 1e-6,
    ):
        super().__init__()
        self.k = k
        self.heads = heads
        self.eps = eps
        self.delta = delta
        self.hash = HashMLP(d_in, buckets=buckets).to(_DEVICE)
        self.cm = CountMin(w=buckets)
        self.score = nn.Parameter(torch.zeros(buckets, device=_DEVICE))
        self.conv = GATConv(
            d_in, d_out // heads, heads=heads, add_self_loops=False
        ).to(_DEVICE)
        self.register_buffer("alpha_max", torch.tensor(1.0))

    def _safe_mask(self, p: torch.Tensor) -> torch.Tensor:
        eps2 = self.eps ** 2
        prob = torch.exp(-eps2 * p / (2 * self.alpha_max.item() ** 2))
        return (prob < self.delta).any()

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:  # noqa: D401
        i, j = edge_index
        b = self.hash(x[i], x[j])
        self.cm.add(b)
        head_sel = torch.zeros(edge_index.size(1), dtype=torch.bool, device=x.device)
        topk = torch.topk(self.score[b], self.k, sorted=False).indices
        head_sel[topk] = True
        p_topk = self.k / (self.cm.query(b[topk]) + 1e-6)
        p_full = torch.ones(edge_index.size(1), device=x.device)
        p_full[topk] = p_topk
        weight = head_sel.float() / p_full.clamp(min=1e-3)

        # Safety check – fallback to dense attention if violated
        if self._safe_mask(p_topk):
            return self.conv((x, x), edge_index)

        # For GAT, we need to use subgraph sampling instead of edge weights
        # Get edges for selected heads
        selected_edges = edge_index[:, head_sel]
        if selected_edges.size(1) == 0:
            # If no edges selected, fallback to full attention
            return self.conv((x, x), edge_index)
        return self.conv((x, x), selected_edges)


class SAFEVALSGAT(nn.Module):
    """Two-layer SAFE-VALS GAT used in the experiments."""

    def __init__(
        self,
        d_in: int,
        d_hid: int,
        d_out: int,
        heads: int,
        k: int,
        buckets: int,
        eps: float,
    ):
        super().__init__()
        self.layer1 = SAFEVALSLayer(
            d_in, d_hid, heads=heads, k=k, buckets=buckets, eps=eps
        )
        self.layer2 = SAFEVALSLayer(
            d_hid, d_out, heads=1, k=k, buckets=buckets, eps=eps
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x, edge_index)
        x = torch.relu(x)
        x = self.layer2(x, edge_index)
        return x


# ---------------------------------  TRAINING  -----------------------------------

def _save_json(out: Dict, tag: str):
    out_dir = Path(".research/iteration1")
    out_dir.mkdir(parents=True, exist_ok=True)
    f_path = out_dir / f"{tag}.json"
    with f_path.open("w") as fp:
        json.dump(out, fp, indent=2)
    print(json.dumps(out, indent=2))


def _sample_subgraph(data: Data, max_nodes: int = 5000, max_edges: int = 50000):
    """Simple random node sampling to create smaller subgraphs."""
    total_nodes = data.x.size(0)

    if total_nodes <= max_nodes:
        return data

    # Sample random nodes
    sampled_indices = torch.randperm(total_nodes)[:max_nodes]
    sampled_indices, _ = torch.sort(sampled_indices)

    # Create node mapping
    node_map = torch.full((total_nodes,), -1, dtype=torch.long)
    node_map[sampled_indices] = torch.arange(max_nodes)

    # Filter edges to only include those between sampled nodes
    src_in_sample = torch.isin(data.edge_index[0], sampled_indices)
    dst_in_sample = torch.isin(data.edge_index[1], sampled_indices)
    edge_mask = src_in_sample & dst_in_sample

    sampled_edge_index = data.edge_index[:, edge_mask]
    if sampled_edge_index.size(1) > max_edges:
        # Further sample edges if too many
        edge_perm = torch.randperm(sampled_edge_index.size(1))[:max_edges]
        sampled_edge_index = sampled_edge_index[:, edge_perm]

    # Remap edge indices to new node indices
    sampled_edge_index[0] = node_map[sampled_edge_index[0]]
    sampled_edge_index[1] = node_map[sampled_edge_index[1]]

    # Create subgraph data object
    sub_data = Data(
        x=data.x[sampled_indices],
        edge_index=sampled_edge_index,
        y=data.y[sampled_indices],
    )

    # Create train mask - sample from original train nodes
    original_train_nodes = sampled_indices[data.train_mask[sampled_indices]]
    if len(original_train_nodes) > 0:
        # Map back to subgraph indices
        sub_train_mask = torch.zeros(max_nodes, dtype=torch.bool)
        train_positions = torch.searchsorted(sampled_indices, original_train_nodes)
        sub_train_mask[train_positions] = True
        sub_data.train_mask = sub_train_mask
    else:
        # Fallback: use all nodes as training if no original train nodes in sample
        sub_data.train_mask = torch.ones(max_nodes, dtype=torch.bool)

    return sub_data


def train(
    data: Data,
    config: Dict,
    tag: str = "smoke_test",
) -> Tuple[nn.Module, Dict]:
    """Training loop with subgraph sampling for large graphs like Reddit."""

    # Check if we need subgraph sampling (for large graphs)
    use_sampling = data.x.size(0) > 10000  # Use sampling for graphs > 10k nodes

    if use_sampling:
        print(f"Using subgraph sampling for large graph with {data.x.size(0)} nodes")
    else:
        print("Using full-batch training")
        data = data.to(_DEVICE)

    model = SAFEVALSGAT(
        d_in=data.x.size(-1),
        d_hid=config["model"]["hidden_dim"],
        d_out=config["model"]["out_dim"],
        heads=config["model"]["heads"],
        k=config["model"]["k"],
        buckets=config["model"]["buckets"],
        eps=config["model"].get("eps", 0.05),
    ).to(_DEVICE)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["lr"]), weight_decay=float(config["training"]["weight_decay"])
    )
    scaler = GradScaler()

    history = {"train_loss": []}
    epochs = int(config["training"]["epochs"])

    model.train()
    for epoch in tqdm(range(epochs), desc=f"Training ({tag})"):
        epoch_loss = 0.0
        num_batches = 0

        if use_sampling:
            # Sample multiple subgraphs per epoch
            batches_per_epoch = 5  # Number of subgraph samples per epoch
            for batch_idx in range(batches_per_epoch):
                # Sample a subgraph
                sub_data = _sample_subgraph(data, max_nodes=5000, max_edges=50000)
                sub_data = sub_data.to(_DEVICE)

                optimiser.zero_grad(set_to_none=True)
                with autocast():
                    out = model(sub_data.x, sub_data.edge_index)
                    if sub_data.train_mask.any():
                        loss = nn.functional.cross_entropy(
                            out[sub_data.train_mask], sub_data.y[sub_data.train_mask]
                        )
                        scaler.scale(loss).backward()
                        scaler.step(optimiser)
                        scaler.update()
                        epoch_loss += loss.item()
                        num_batches += 1
        else:
            # Full-batch training
            optimiser.zero_grad(set_to_none=True)
            with autocast():
                out = model(data.x, data.edge_index)
                loss = nn.functional.cross_entropy(
                    out[data.train_mask], data.y[data.train_mask]
                )
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            epoch_loss = loss.item()
            num_batches = 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        history["train_loss"].append(avg_loss)

    _save_json(history, f"{tag}_train_metrics")
    return model, history
