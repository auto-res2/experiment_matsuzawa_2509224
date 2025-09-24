import json
from pathlib import Path
from typing import Dict

import torch
from torch_geometric.data import Data

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model: torch.nn.Module, data: Data, tag: str = "smoke_test") -> Dict:
    model.eval()
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
