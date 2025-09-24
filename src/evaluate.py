import json
from pathlib import Path
from typing import Dict

import torch
from torch_geometric.data import Data

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model: torch.nn.Module, data: Data, tag: str = "smoke_test") -> Dict:
    model.eval()

    # For large graphs, evaluate in smaller chunks to avoid OOM
    use_chunks = data.x.size(0) > 10000

    if use_chunks:
        # Evaluate in chunks for large graphs
        all_preds = []
        chunk_size = 1000  # Process 1000 nodes at a time

        with torch.no_grad():
            for start_idx in range(0, data.x.size(0), chunk_size):
                end_idx = min(start_idx + chunk_size, data.x.size(0))

                # Create subgraph for this chunk
                node_mask = torch.zeros(data.x.size(0), dtype=torch.bool)
                node_mask[start_idx:end_idx] = True

                # Get edges that connect nodes in this chunk
                edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
                chunk_edges = data.edge_index[:, edge_mask]

                # Remap edge indices to local chunk indices
                chunk_edges = chunk_edges - start_idx

                # Move chunk to device
                chunk_x = data.x[start_idx:end_idx].to(_DEVICE)
                chunk_edges = chunk_edges.to(_DEVICE)

                if chunk_edges.size(1) > 0:  # Only process if there are edges
                    chunk_out = model(chunk_x, chunk_edges)
                    chunk_preds = chunk_out.argmax(dim=-1)
                else:
                    # If no edges, use zero vector (shouldn't happen with proper graphs)
                    chunk_preds = torch.zeros(chunk_x.size(0), dtype=torch.long, device=_DEVICE)

                all_preds.append(chunk_preds.cpu())

        # Concatenate all predictions
        preds = torch.cat(all_preds, dim=0)

        # Calculate accuracy on test nodes
        acc = (
            (preds[data.test_mask] == data.y[data.test_mask]).sum().float()
            / data.test_mask.sum()
        ).item()
    else:
        # Full evaluation for small graphs
        data = data.to(_DEVICE)
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            preds = out.argmax(dim=-1)
            acc = (
                (preds[data.test_mask] == data.y[data.test_mask]).sum().float()
                / data.test_mask.sum()
            ).item()

    results = {"test_accuracy": acc}
    # Persist & echo
    out_dir = Path(".research/iteration1")
    out_dir.mkdir(parents=True, exist_ok=True)
    f_path = out_dir / f"{tag}_eval_metrics.json"
    with f_path.open("w") as fp:
        json.dump(results, fp, indent=2)
    print(json.dumps(results, indent=2))
    return results
