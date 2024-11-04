import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str | None = None, input_dim: int | None = None, output_dim: int | None = None, is_inplace: bool = True) -> nn.Module:
    if name is None:
        return nn.Identity()
    if name in {"swish", "silu"}:
        return nn.SiLU(is_inplace)
    if name == "mish":
        return nn.Mish(is_inplace)
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(is_inplace)
    if name == "geglu":
        return GEGLU(input_dim, output_dim or input_dim)
    
    raise ValueError(f"[GetActivation] Invalid activation function: {name}")


class GEGLU(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim

        self.proj = nn.Linear(input_dim, output_dim * 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.chunk(self.proj(x), 2, dim=-1)
        return a * F.gelu(b)
    