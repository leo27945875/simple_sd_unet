import torch
import torch.nn as nn


class AbsDiffusionBlock(nn.Module):

    in_channel     : int
    out_channel    : int
    temb_channel   : int
    other_channels : list[int] | None


class AbsDiffusionDownBlock(AbsDiffusionBlock):
    
    def forward(self, x: torch.Tensor, temb: torch.Tensor, other_embeds: list[torch.Tensor] | None = None, additional_residual: torch.Tensor | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        raise NotImplementedError


class AbsDiffusionUpBlock(AbsDiffusionBlock):

    skip_channel   : int

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor], temb: torch.Tensor, other_embeds: list[torch.Tensor] | None = None, additional_residual: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class AbsDiffusionMidBlock(AbsDiffusionBlock):

    def forward(self, x: torch.Tensor, temb: torch.Tensor, other_embeds: list[torch.Tensor] | None = None, additional_residual: torch.Tensor | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        raise NotImplementedError
    

class AbsDiffusionCondDownBlock(AbsDiffusionDownBlock):

    cross_attn_channel : int | None

    def forward(self, x: torch.Tensor, temb: torch.Tensor, other_embeds: list[torch.Tensor] | None = None, extract: torch.Tensor = None, additional_residual: torch.Tensor | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        raise NotImplementedError


class AbsDiffusionCondUpBlock(AbsDiffusionUpBlock):

    cross_attn_channel : int | None

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor], temb: torch.Tensor, other_embeds: list[torch.Tensor] | None = None, extract: torch.Tensor = None, additional_residual: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError
    

class AbsDiffusionCondMidBlock(AbsDiffusionMidBlock):

    cross_attn_channel : int | None

    def forward(self, x: torch.Tensor, temb: torch.Tensor, other_embeds: list[torch.Tensor] | None = None, extract: torch.Tensor = None, additional_residual: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError