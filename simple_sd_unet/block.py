import torch
import torch.nn as nn

from . import (
    AbsDiffusionDownBlock,
    AbsDiffusionUpBlock,
    AbsDiffusionMidBlock,
    AbsDiffusionCondDownBlock,
    AbsDiffusionCondMidBlock,
    AbsDiffusionCondUpBlock
)
from .layer import *
from .external import (
    Timesteps, 
    TimestepEmbedding, 
    GaussianFourierProjection
)


############################################### Getters ###############################################
def get_time_block(
    time_embed_type : str, 
    proj_dim        : int,
    time_embed_dim  : int,
    is_flip_sin_cos : bool,
    freq_shift      : float
) -> tuple[GaussianFourierProjection | Timesteps, TimestepEmbedding]:
    
    if time_embed_type == "fourier":
        time_proj = GaussianFourierProjection(embedding_size=proj_dim, scale=16)
        timestep_input_dim = 2 * proj_dim
    elif time_embed_type == "positional":
        time_proj = Timesteps(proj_dim, is_flip_sin_cos, freq_shift)
        timestep_input_dim = proj_dim
    else:
        raise ValueError(f"[get_time_block] Invalid time embedding type : {time_embed_type}.")
    
    time_embed = TimestepEmbedding(timestep_input_dim, time_embed_dim)

    return time_proj, time_embed


def get_down_block(
    down_block_type      : str,
    in_channel           : int, 
    out_channel          : int, 
    time_channel         : int, 
    other_channels       : list[int] | None = None, 
    cross_attn_dim       : int | None       = None,
    num_layer            : int              = 1,
    num_group            : int              = 32, 
    attn_head_dim        : int              = 8,
    act                  : str              = "silu",
    dropout              : float            = 0.,
    eps                  : float            = 1e-5,
    is_downsample        : bool             = True,
    is_linear_projection : bool             = True,
    is_spatial_attention : bool             = False
) -> AbsDiffusionDownBlock:
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            in_channel, out_channel, time_channel, other_channels, num_layer, num_group, act, dropout, eps, is_downsample, is_spatial_attention
        )
    elif down_block_type == "AttnDownBlock2D":
        return AttnDownBlock2D(
            in_channel, out_channel, time_channel, other_channels, num_layer, num_group, attn_head_dim, act, dropout, eps, is_downsample, is_spatial_attention
        )
    elif down_block_type == "CrossAttnDownBlock2D":
        num_attn_head = attn_head_dim
        return CrossAttnDownBlock2D(
            in_channel, out_channel, time_channel, other_channels, num_layer, num_group, num_attn_head, act, dropout, eps, is_downsample, cross_attn_dim, is_linear_projection, is_spatial_attention
        )
    else:
        raise ValueError(f"[get_down_block] Invalid down block type : {down_block_type}.")


def get_up_block(
    up_block_type        : str,
    in_channel           : int, 
    out_channel          : int, 
    skip_channel         : int,
    time_channel         : int, 
    other_channels       : list[int] | None = None, 
    cross_attn_dim       : int | None       = None,
    num_layer            : int              = 1,
    num_group            : int              = 32, 
    attn_head_dim        : int              = 8,
    act                  : str              = "silu",
    dropout              : float            = 0.,
    eps                  : float            = 1e-5,
    is_upsample          : bool             = True,
    is_linear_projection : bool             = True,
    is_spatial_attention : bool             = False
) -> AbsDiffusionUpBlock:
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            in_channel, out_channel, skip_channel, time_channel, other_channels, num_layer, num_group, act, dropout, eps, is_upsample, is_spatial_attention
        )
    elif up_block_type == "AttnUpBlock2D":
        return AttnUpBlock2D(
            in_channel, out_channel, skip_channel, time_channel, other_channels, num_layer, num_group, attn_head_dim, act, dropout, eps, is_upsample, is_spatial_attention
        )
    elif up_block_type == "CrossAttnUpBlock2D":
        num_attn_head = attn_head_dim
        return CrossAttnUpBlock2D(
            in_channel, out_channel, skip_channel, time_channel, other_channels, num_layer, num_group, num_attn_head, act, dropout, eps, is_upsample, cross_attn_dim, is_linear_projection, is_spatial_attention
        )
    else:
        raise ValueError(f"[get_up_block] Invalid up block type : {up_block_type}.")


def get_mid_block(
    mid_block_type       : str,
    num_channel          : int,
    time_channel         : int, 
    other_channels       : list[int] | None = None, 
    cross_attn_dim       : int | None       = None,
    num_layer            : int              = 1,
    num_group            : int              = 32, 
    attn_head_dim        : int              = 8,
    act                  : str              = "silu",
    dropout              : float            = 0.,
    eps                  : float            = 1e-5,
    is_linear_projection : bool             = True,
    is_spatial_attention : bool             = False
) -> AbsDiffusionMidBlock:
    if mid_block_type == "UNetMidBlock2D":
        return UNetMidBlock2D(
            num_channel, num_channel, time_channel, other_channels, num_layer, num_group, attn_head_dim, act, dropout, eps, is_spatial_attention
        )
    elif mid_block_type == "UNetMidBlock2DCrossAttn":
        num_attn_head = attn_head_dim
        return UNetMidBlock2DCrossAttn(
            num_channel, num_channel, time_channel, other_channels, num_layer, num_group, num_attn_head, act, dropout, eps, cross_attn_dim, is_linear_projection, is_spatial_attention
        )
    else:
        raise ValueError(f"[get_mid_block] Invalid up block type : {mid_block_type}.")
    

############################################### Conditional Block Implementation ###############################################
class CrossAttnDownBlock2D(AbsDiffusionCondDownBlock):
    def __init__(
        self,
        in_channel           : int, 
        out_channel          : int, 
        temb_channel         : int, 
        other_channels       : list[int] | None = None, 
        num_layer            : int              = 1,
        num_group            : int              = 32, 
        num_attn_head        : int              = 8,
        act                  : str              = "silu",
        dropout              : float            = 0.,
        eps                  : float            = 1e-5,
        is_downsample        : bool             = True,
        cross_attn_channel   : int | None       = None,
        is_linear_projection : bool             = True,
        is_spatial_attention : bool             = False
    ) -> None:
        super().__init__()
        self.in_channel           = in_channel
        self.out_channel          = out_channel
        self.temb_channel         = temb_channel
        self.other_channels       = other_channels
        self.cross_attn_channel   = cross_attn_channel
        self.is_spatial_attention = is_spatial_attention

        convs, attns = [], []
        for i in range(num_layer):
            in_channel = in_channel if i == 0 else out_channel
            convs.append(DiffusionResLayer(
                in_channel,
                out_channel,
                temb_channel,
                other_channels,
                num_group,
                act,
                dropout,
                eps=eps
            ))
            attns.append(DiffusionTransformerLayer(
                out_channel,
                num_attn_head,
                out_channel // num_attn_head,
                cross_attn_channel,
                num_group=num_group,
                eps=eps,
                is_linear_projection=is_linear_projection,
                is_spatial_attention=is_spatial_attention
            ))

        self.resnets      = nn.ModuleList(convs)
        self.attentions   = nn.ModuleList(attns)
        self.downsamplers = nn.ModuleList([Downsample(out_channel, out_channel)]) if is_downsample else None

    def forward(self, x: torch.Tensor, temb: torch.Tensor, other_embeds: list[torch.Tensor] | None = None, extract: torch.Tensor = None, additional_residual: torch.Tensor | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        output_states = []
        num_layer = len(self.resnets)
        h = x
        for i, (conv, attn) in enumerate(zip(self.resnets, self.attentions)):
            h = conv(h, temb, other_embeds)
            h = attn(h, extract)
            if i == num_layer - 1 and additional_residual is not None:
                h += additional_residual
            output_states.append(h)
        
        if self.downsamplers:
            for down in self.downsamplers:
                h = down(h)
            output_states.append(h)
        
        return h, output_states
    

class CrossAttnUpBlock2D(AbsDiffusionCondUpBlock):
    def __init__(
        self,
        in_channel           : int, 
        out_channel          : int, 
        skip_channel         : int,
        time_channel         : int, 
        other_channels       : list[int] | None = None, 
        num_layer            : int              = 1,
        num_group            : int              = 32, 
        num_attn_head        : int              = 8,
        act                  : str              = "silu",
        dropout              : float            = 0.,
        eps                  : float            = 1e-5,
        is_upsample          : bool             = True,
        cross_attn_channel   : int | None       = None,
        is_linear_projection : bool             = True,
        is_spatial_attention : bool             = False
    ) -> None:
        super().__init__()
        self.in_channel           = in_channel
        self.out_channel          = out_channel
        self.skip_channel         = skip_channel
        self.temb_channel         = time_channel
        self.other_channels       = other_channels
        self.cross_attn_channel   = cross_attn_channel
        self.is_spatial_attention = is_spatial_attention

        convs, attns = [], []
        for i in range(num_layer):
            res_skip_channel = in_channel   if i == num_layer - 1 else out_channel
            res_in_channel   = skip_channel if i == 0            else out_channel
            convs.append(DiffusionResLayer(
                res_in_channel + res_skip_channel,
                out_channel,
                time_channel,
                other_channels,
                num_group,
                act,
                dropout,
                eps=eps
            ))
            attns.append(DiffusionTransformerLayer(
                out_channel,
                num_attn_head,
                out_channel // num_attn_head,
                cross_attn_channel,
                num_group=num_group,
                eps=eps,
                is_linear_projection=is_linear_projection,
                is_spatial_attention=is_spatial_attention
            ))

        self.resnets    = nn.ModuleList(convs)
        self.attentions = nn.ModuleList(attns)
        self.upsamplers = nn.ModuleList([Upsample(out_channel, out_channel)]) if is_upsample else None

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor], temb: torch.Tensor, other_embeds: list[torch.Tensor] | None = None, extract: torch.Tensor = None, additional_residual: torch.Tensor | None = None) -> torch.Tensor:
        num_layer = len(self.resnets)
        h = x
        for i, (conv, attn) in enumerate(zip(self.resnets, self.attentions)):
            skip = skips.pop()
            h = conv(torch.cat([h, skip], dim=1), temb, other_embeds)
            h = attn(h, extract)
            if i == num_layer - 1 and additional_residual is not None:
                h += additional_residual
        
        if self.upsamplers:
            for up in self.upsamplers:
                h = up(h)
        return h


class UNetMidBlock2DCrossAttn(AbsDiffusionCondMidBlock):
    def __init__(
        self, 
        in_channel           : int, 
        out_channel          : int,
        time_channel         : int, 
        other_channels       : list[int] | None = None, 
        num_layer            : int              = 1,
        num_group            : int              = 32, 
        num_attn_head        : int              = 8,
        act                  : str              = "silu",
        dropout              : float            = 0.,
        eps                  : float            = 1e-5,
        cross_attn_channel   : int | None       = None,
        is_linear_projection : bool             = True,
        is_spatial_attention : bool             = False
    ) -> None:
        super().__init__()
        self.in_channel           = in_channel
        self.out_channel          = out_channel
        self.temb_channel         = time_channel
        self.other_channels       = other_channels
        self.cross_attn_channel   = cross_attn_channel
        self.is_spatial_attention = is_spatial_attention

        convs = [DiffusionResLayer(in_channel, out_channel, time_channel, other_channels, num_group, act, dropout, eps=eps)]
        attns = []
        for _ in range(num_layer):
            attns.append(DiffusionTransformerLayer(
                out_channel, num_attn_head, out_channel // num_attn_head, cross_attn_channel, num_group=num_group, eps=eps, is_linear_projection=is_linear_projection, is_spatial_attention=is_spatial_attention
            ))
            convs.append(DiffusionResLayer(
                out_channel, out_channel, time_channel, other_channels, num_group, act, dropout, eps=eps
            ))
        
        self.resnets    = nn.ModuleList(convs)
        self.attentions = nn.ModuleList(attns)

    def forward(self, x: torch.Tensor, temb: torch.Tensor, other_embeds: list[torch.Tensor] | None = None, extract: torch.Tensor = None, additional_residual: torch.Tensor | None = None) -> torch.Tensor:
        num_layer = len(self.resnets)
        h = self.resnets[0](x, temb, other_embeds)
        for i, (attn, conv) in enumerate(zip(self.attentions, self.resnets[1:])):
            h = attn(h, extract)
            h = conv(h, temb, other_embeds)
            if i == num_layer - 1 and additional_residual is not None:
                h += additional_residual
        return h


############################################### Unconditional Block Implementation ###############################################
class AttnDownBlock2D(AbsDiffusionDownBlock):
    def __init__(
        self,
        in_channel           : int, 
        out_channel          : int, 
        temb_channel         : int, 
        other_channels       : list[int] | None = None, 
        num_layer            : int              = 1,
        num_group            : int              = 32, 
        attn_head_dim        : int              = 8,
        act                  : str              = "silu",
        dropout              : float            = 0.,
        eps                  : float            = 1e-5,
        is_downsample        : bool             = True,
        is_spatial_attention : bool             = False
    ) -> None:
        super().__init__()
        self.in_channel           = in_channel
        self.out_channel          = out_channel
        self.temb_channel         = temb_channel
        self.other_channels       = other_channels
        self.is_spatial_attention = is_spatial_attention

        convs, attns = [], []
        for i in range(num_layer):
            in_channel = in_channel if i == 0 else out_channel
            convs.append(DiffusionResLayer(
                in_channel,
                out_channel,
                temb_channel,
                other_channels,
                num_group,
                act,
                dropout,
                eps=eps
            ))
            attns.append(SelfAttention(
                out_channel,
                out_channel // attn_head_dim,
                num_group,
                eps=eps,
                is_spatial_attention=is_spatial_attention
            ))

        self.resnets      = nn.ModuleList(convs)
        self.attentions   = nn.ModuleList(attns)
        self.downsamplers = nn.ModuleList([Downsample(out_channel, out_channel)]) if is_downsample else None

    def forward(self, x: torch.Tensor, temb: torch.Tensor, other_embeds: list[torch.Tensor] | None = None, additional_residual: torch.Tensor | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        num_layer = len(self.resnets)
        output_states = []
        h = x
        for i, (conv, attn) in zip(self.resnets, self.attentions):
            h = conv(h, temb, other_embeds)
            h = attn(h)
            if i == num_layer - 1 and additional_residual is not None:
                h += additional_residual
            output_states.append(h)
        
        if self.downsamplers:
            for down in self.downsamplers:
                h = down(h)
            output_states.append(h)
        
        return h, output_states


class DownBlock2D(AbsDiffusionDownBlock):
    def __init__(
        self, 
        in_channel           : int, 
        out_channel          : int, 
        temb_channel         : int, 
        other_channels       : list[int] | None = None, 
        num_layer            : int              = 1,
        num_group            : int              = 32,
        act                  : str              = "silu",
        dropout              : float            = 0.,
        eps                  : float            = 1e-5,
        is_downsample        : bool             = True,
        is_spatial_attention : bool             = False
    ) -> None:
        super().__init__()
        self.in_channel           = in_channel
        self.out_channel          = out_channel
        self.temb_channel         = temb_channel
        self.other_channels       = other_channels
        self.is_spatial_attention = is_spatial_attention

        self.resnets = nn.ModuleList([
            DiffusionResLayer(
                in_channel if i == 0 else out_channel,
                out_channel,
                temb_channel,
                other_channels,
                num_group,
                act,
                dropout,
                eps=eps
            )
            for i in range(num_layer)
        ])
        self.downsamplers = nn.ModuleList([Downsample(out_channel, out_channel)]) if is_downsample else None

    def forward(self, x: torch.Tensor, temb: torch.Tensor, other_embeds: list[torch.Tensor] | None = None, additional_residual: torch.Tensor | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        num_layer = len(self.resnets)
        output_states = []
        h = x
        for i, conv in enumerate(self.resnets):
            h = conv(h, temb, other_embeds)
            if i == num_layer - 1 and additional_residual is not None:
                h += additional_residual
            output_states.append(h)

        if self.downsamplers:
            for down in self.downsamplers:
                h = down(h)
            output_states.append(h)
        
        return h, output_states


class AttnUpBlock2D(AbsDiffusionUpBlock):
    def __init__(
        self,
        in_channel           : int, 
        out_channel          : int, 
        skip_channel         : int,
        time_channel         : int, 
        other_channels       : list[int] | None = None, 
        num_layer            : int              = 1,
        num_group            : int              = 32, 
        attn_head_dim        : int              = 8,
        act                  : str              = "silu",
        dropout              : float            = 0.,
        eps                  : float            = 1e-5,
        is_upsample          : bool             = True,
        is_spatial_attention : bool             = False
    ) -> None:
        super().__init__()
        self.in_channel           = in_channel
        self.out_channel          = out_channel
        self.skip_channel         = skip_channel
        self.temb_channel         = time_channel
        self.other_channels       = other_channels
        self.is_spatial_attention = is_spatial_attention

        convs, attns = [], []
        for i in range(num_layer):
            res_skip_channel = in_channel   if i == num_layer - 1 else out_channel
            res_in_channel   = skip_channel if i == 0            else out_channel
            convs.append(DiffusionResLayer(
                res_in_channel + res_skip_channel,
                out_channel,
                time_channel,
                other_channels,
                num_group,
                act,
                dropout,
                eps=eps
            ))
            attns.append(SelfAttention(
                out_channel,
                out_channel // attn_head_dim,
                attn_head_dim,
                num_group,
                eps=eps,
                is_spatial_attention=is_spatial_attention
            ))

        self.resnets    = nn.ModuleList(convs)
        self.attentions = nn.ModuleList(attns)
        self.upsamplers = nn.ModuleList([Upsample(out_channel, out_channel)]) if is_upsample else None

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor], temb: torch.Tensor, other_embeds: list[torch.Tensor] | None = None, additional_residual: torch.Tensor | None = None) -> torch.Tensor:
        num_layer = len(self.resnets)
        h = x
        for i, (conv, attn) in zip(self.resnets, self.attentions):
            skip = skips.pop()
            h = conv(torch.cat([h, skip], dim=1), temb, other_embeds)
            h = attn(h)
            if i == num_layer - 1 and additional_residual is not None:
                h += additional_residual
        
        if self.upsamplers:
            for up in self.upsamplers:
                h = up(h)
        return h


class UpBlock2D(AbsDiffusionUpBlock):
    def __init__(
        self,
        in_channel           : int, 
        out_channel          : int, 
        skip_channel         : int,
        time_channel         : int, 
        other_channels       : list[int] | None = None, 
        num_layer            : int              = 1,
        num_group            : int              = 32,
        act                  : str              = "silu",
        dropout              : float            = 0.,
        eps                  : float            = 1e-5,
        is_upsample          : bool             = True,
        is_spatial_attention : bool             = False
    ) -> None:
        super().__init__()
        self.in_channel           = in_channel
        self.out_channel          = out_channel
        self.skip_channel         = skip_channel
        self.temb_channel         = time_channel
        self.other_channels       = other_channels
        self.is_spatial_attention = is_spatial_attention

        convs = []
        for i in range(num_layer):
            res_skip_channel = in_channel  if i == num_layer - 1 else out_channel  # Residual connection channel
            res_in_channel   = skip_channel if i == 0             else out_channel  # Output channel from prev up block
            convs.append(DiffusionResLayer(
                res_in_channel + res_skip_channel,
                out_channel,
                time_channel,
                other_channels,
                num_group,
                act,
                dropout,
                eps=eps
            ))

        self.resnets    = nn.ModuleList(convs)
        self.upsamplers = nn.ModuleList([Upsample(out_channel, out_channel)]) if is_upsample else None

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor], temb: torch.Tensor, other_embeds: list[torch.Tensor] | None = None, additional_residual: torch.Tensor | None = None) -> torch.Tensor:
        num_layer = len(self.resnets)
        h = x
        for i, conv in enumerate(self.resnets):
            skip = skips.pop()
            h = conv(torch.cat([h, skip], dim=1), temb, other_embeds)
            if i == num_layer - 1 and additional_residual is not None:
                h += additional_residual
        
        if self.upsamplers:
            for up in self.upsamplers:
                h = up(h)
        return h
    

class UNetMidBlock2D(AbsDiffusionMidBlock):
    def __init__(
        self, 
        in_channel           : int, 
        out_channel          : int,
        time_channel         : int, 
        other_channels       : list[int] | None = None, 
        num_layer            : int              = 1,
        num_group            : int              = 32, 
        attn_head_dim        : int              = 8,
        act                  : str              = "silu",
        dropout              : float            = 0.,
        eps                  : float            = 1e-5,
        is_spatial_attention : bool             = False
    ) -> None:
        super().__init__()
        self.in_channel           = in_channel
        self.out_channel          = out_channel
        self.temb_channel         = time_channel
        self.other_channels       = other_channels
        self.is_spatial_attention = is_spatial_attention

        convs = [DiffusionResLayer(in_channel, out_channel, time_channel, other_channels, num_group, act, dropout, eps=eps)]
        attns = []
        for _ in range(num_layer):
            attns.append(SelfAttention(
                out_channel, out_channel // attn_head_dim, attn_head_dim, num_group, eps=eps, is_spatial_attention=is_spatial_attention
            ))
            convs.append(DiffusionResLayer(
                out_channel, out_channel, time_channel, other_channels, num_group, act, dropout, eps=eps
            ))
        
        self.resnets    = nn.ModuleList(convs)
        self.attentions = nn.ModuleList(attns)

    def forward(self, x: torch.Tensor, temb: torch.Tensor, other_embeds: list[torch.Tensor] | None = None, additional_residual: torch.Tensor | None = None) -> torch.Tensor:
        num_layer = len(self.resnets)
        h = self.resnets[0](x, temb, other_embeds)
        for i, (attn, conv) in enumerate(zip(self.attentions, self.resnets[1:])):
            h = attn(h)
            h = conv(h, temb, other_embeds)
            if i == num_layer - 1 and additional_residual is not None:
                h += additional_residual
        return h