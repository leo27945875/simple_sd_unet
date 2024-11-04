import json
from typing import Any

import torch
from einops import rearrange


def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def get_dict_default(data: dict | None, key: str, default: Any | None = None) -> Any:
    if data is None or key not in data:
        return default
    return data[key]


def spatial_concat(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "(n b) c h w -> b c h (n w)", n=2)


def spatial_batch(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h (n w) -> (n b) c h w", n=2)
