from typing import Dict

import torch
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.transforms import NormalizeFeatures


def load_data(cfg: Dict):
    ds_name = cfg["dataset"].lower()
    root = ".cache/datasets"

    if ds_name == "cora":
        dataset = Planetoid(root=root, name="Cora", transform=NormalizeFeatures())
    elif ds_name == "reddit":
        dataset = Reddit(root=root, transform=NormalizeFeatures())
    else:
        raise ValueError(f"Unsupported dataset {ds_name}")

    data = dataset[0]
    # Ensure boolean masks exist for Reddit
    if not hasattr(data, "train_mask"):
        n_nodes = data.y.size(0)
        idx = torch.randperm(n_nodes)
        split1 = int(0.6 * n_nodes)
        split2 = int(0.8 * n_nodes)
        data.train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        data.train_mask[idx[:split1]] = True
        data.val_mask[idx[split1:split2]] = True
        data.test_mask[idx[split2:]] = True
    return data
