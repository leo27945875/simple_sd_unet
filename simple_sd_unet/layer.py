import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence, Self

from .utils      import spatial_concat, spatial_batch
from .activation import get_activation


class AdaGroupNorm(nn.Module):
    def __init__(self, num_group: int, emb_dims: list[int], input_dim: int, eps: float = 1e-5, is_load_pretrained_gn: bool = False) -> None:
        super().__init__()
        self.emb_dims   = emb_dims
        self.input_dim  = input_dim
        self.output_dim = input_dim
        self.num_group  = num_group
        self.eps        = eps

        self.linears = nn.ModuleList([
            self.__make_linear(emb_dim, input_dim * 2) for emb_dim in emb_dims
        ])
        
        # Register the pre-hooks of load_state_dict.
        def _load_group_norm_pre_hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None:
            origin_scale_name, origin_shift_name, n_emb, dim = prefix + "weight", prefix + "bias", len(self.emb_dims), self.input_dim
            assert n_emb == 1, "[AdaGroupNorm] only support [len(self.emb_dims) == 1] for loading Stable Diffusion paramters."
            linear_weight_names, linear_bias_names, is_all_in_state_dict = [], [], True
            for i in range(len(self.linears)):
                linear_weight_name, linear_bias_name = prefix + f"linears.{i}.weight" , prefix + f"linears.{i}.bias"
                if linear_weight_name not in state_dict :
                    linear_weight_names.append(linear_weight_name)
                    is_all_in_state_dict = False
                else:
                    linear_weight_names.append(None)

                if linear_bias_name not in state_dict:
                    linear_bias_names.append(linear_bias_name)
                    is_all_in_state_dict = False
                else:
                    linear_bias_names.append(None)

            if is_all_in_state_dict:
                return

            # Load from Stable Diffusion:
            origin_scale = state_dict.pop(origin_scale_name) / n_emb
            origin_shift = state_dict.pop(origin_shift_name) / n_emb
            assert isinstance(origin_scale, torch.Tensor)
            assert isinstance(origin_shift, torch.Tensor)
            for linear, linear_weight_name, linear_bias_name in zip(self.linears, linear_weight_names, linear_bias_names):
                assert isinstance(linear, nn.Linear)
                if linear_weight_name: 
                    new_weight = torch.zeros_like(linear.weight, dtype=origin_scale.dtype)
                    state_dict[linear_weight_name] = new_weight
                if linear_bias_name:
                    new_bias = torch.zeros_like(linear.bias, dtype=origin_shift.dtype)
                    new_bias[:dim] = origin_scale - 1.
                    new_bias[dim:] = origin_shift
                    state_dict[linear_bias_name] = new_bias

        if is_load_pretrained_gn:
            self._register_load_state_dict_pre_hook(_load_group_norm_pre_hook)
        
    def forward(self, x: torch.Tensor, embs: list[torch.Tensor]) -> torch.Tensor:
        h = F.group_norm(x, self.num_group, eps=self.eps)
        for emb, linear in zip(embs, self.linears):
            emb = linear(emb)[:, :, None, None]
            scale, shift = torch.chunk(emb, 2, dim=1)
            h = h * (1. + scale) + shift
        return h
    
    def __make_linear(self, input_dim: int, output_dim: int) -> nn.Linear:
        linear = nn.Linear(input_dim, output_dim)
        torch.nn.init.zeros_(linear.weight)
        torch.nn.init.zeros_(linear.bias)
        return linear


class Downsample(nn.Module):
    def __init__(self, in_channel: int, out_channel: int | None = None) -> None:
        super().__init__()
        self.in_channel  = in_channel
        self.out_channel = out_channel or in_channel

        self.conv = nn.Conv2d(in_channel, self.out_channel, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    

class Upsample(nn.Module):
    def __init__(self, in_channel: int, out_channel: int | None = None) -> None:
        super().__init__()
        self.in_channel  = in_channel
        self.out_channel = out_channel or in_channel

        self.conv = nn.Conv2d(in_channel, self.out_channel, 3, 1, 1)

    def forward(self, x: torch.Tensor, output_size: Sequence[int] | None = None) -> torch.Tensor:
        if output_size is None:
            h = F.interpolate(x, scale_factor=2., mode="nearest")
        else:
            h = F.interpolate(x, output_size, mode="nearest")
        return self.conv(h)


class SelfAttention(nn.Module):
    def __init__(self, input_dim: int, n_head: int, attn_head_dim: int, num_group: int | None = None, dropout: float = 0., eps: float = 1e-5, is_spatial_attention: bool = False) -> None:
        super().__init__()
        self.input_dim            = input_dim
        self.output_dim           = input_dim
        self.inner_dim            = attn_head_dim * n_head
        self.attn_head_dim        = attn_head_dim
        self.n_head               = n_head
        self.is_spatial_attention = is_spatial_attention

        self.group_norm = nn.GroupNorm(num_group, input_dim, eps=eps) if num_group else nn.Identity()
        self.to_q       = nn.Linear(input_dim, self.inner_dim, bias=False)
        self.to_k       = nn.Linear(input_dim, self.inner_dim, bias=False)
        self.to_v       = nn.Linear(input_dim, self.inner_dim, bias=False)
        self.to_out     = nn.ModuleList([
            nn.Linear(self.inner_dim, self.output_dim),
            nn.Dropout(dropout)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if self.is_spatial_attention:
                x = spatial_concat(x)
            B, _, H, W = x.size()
            h = x.view(B, self.input_dim, H * W).transpose(1, 2)
            h = self.__self_attn(h)
            h = h.transpose(1, 2).view(B, self.output_dim, H, W)
            if self.is_spatial_attention:
                h = spatial_batch(h)
            return h
        
        return self.__self_attn(x)
    
    def __self_attn(self, x: torch.Tensor) -> torch.Tensor:
        dim, n_head, head_dim = self.input_dim, self.n_head, self.attn_head_dim
        B, L, _  = x.size()

        q, k, v = self.__get_QKV(self.group_norm(x))
        q = q.view(B, L, n_head, head_dim).transpose(1, 2)
        k = k.view(B, L, n_head, head_dim).transpose(1, 2)
        v = v.view(B, L, n_head, head_dim).transpose(1, 2)

        h = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).view(B, L, dim)
        h = self.to_out[0](h)
        h = self.to_out[1](h)
        return h
    
    def __get_QKV(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.to_q(x), self.to_k(x), self.to_v(x)


class CrossAttention(nn.Module):
    def __init__(self, input_dim: int, n_head: int, attn_head_dim: int, cross_dim: int = 1024, num_group: int | None = None, dropout: float = 0., eps: float = 1e-5, is_spatial_attention: bool = False) -> None:
        super().__init__()
        self.input_dim            = input_dim
        self.output_dim           = input_dim
        self.cross_dim            = cross_dim or input_dim
        self.inner_dim            = n_head * attn_head_dim
        self.attn_head_dim        = attn_head_dim
        self.n_head               = n_head
        self.is_spatial_attention = is_spatial_attention

        self.group_norm = nn.GroupNorm(num_group, input_dim, eps=eps) if num_group else nn.Identity()
        self.to_q       = nn.Linear(input_dim, self.inner_dim, bias=False)
        self.to_k       = nn.Linear(cross_dim, self.inner_dim, bias=False)
        self.to_v       = nn.Linear(cross_dim, self.inner_dim, bias=False)
        self.to_out     = nn.ModuleList([
            nn.Linear(self.inner_dim, self.output_dim),
            nn.Dropout(dropout)
        ])

    def forward(self, x: torch.Tensor, extract: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 4:
            if self.is_spatial_attention:
                x = spatial_concat(x)
            B, _, H, W = x.size()
            h = x.view(B, self.input_dim, H * W).transpose(1, 2)
            h = self.__cross_attn(h, extract)
            h = h.transpose(1, 2).view(B, self.output_dim, H, W)
            if self.is_spatial_attention:
                h = spatial_batch(h)
            return h
        
        return self.__cross_attn(x, extract)

    def __cross_attn(self, x: torch.Tensor, extract: torch.Tensor | None = None) -> torch.Tensor:
        x = self.group_norm(x)
        if extract is None:
            extract = x

        dim, n_head, head_dim = self.input_dim, self.n_head, self.attn_head_dim
        B, L, _ = x.size()
        _, N, _ = extract.size()

        q, k, v = self.__get_QKV(x, extract)
        q = q.view(B, L, n_head, head_dim).transpose(1, 2)
        k = k.view(B, N, n_head, head_dim).transpose(1, 2)
        v = v.view(B, N, n_head, head_dim).transpose(1, 2)

        h = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).view(B, L, dim)
        h = self.to_out[0](h)
        h = self.to_out[1](h)
        return h

    def __get_QKV(self, x: torch.Tensor, extract: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.to_q(x), self.to_k(extract), self.to_v(extract)


class TransformerFeedForward(nn.Module):
    def __init__(self, input_dim: int, output_dim: int | None = None, act: str = "geglu", inner_dim_mult: int = 4, dropout: float = 0., is_final_dropout: bool = False) -> None:
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim or input_dim
        self.inner_dim  = input_dim * inner_dim_mult

        self.net = nn.ModuleList([])
        self.net.append(get_activation(act, self.input_dim, self.inner_dim))
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(self.inner_dim, self.output_dim))
        if is_final_dropout:
            self.net.append(nn.Dropout(dropout))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.net:
            h = layer(h)
        return h


class BasicTransformer(nn.Module):
    def __init__(self, input_dim: int, cross_attn_dim: int | None = None, n_head: int = 16, attn_head_dim: int = 88, act: str = "geglu", dropout: float = 0., eps: float = 1e-5) -> None:
        super().__init__()
        self.input_dim     = input_dim
        self.output_dim    = input_dim
        self.inner_dim     = n_head * attn_head_dim
        self.attn_head_dim = attn_head_dim
        self.n_head        = n_head

        self.norm1 = nn.LayerNorm(input_dim, eps=eps)
        self.norm2 = nn.LayerNorm(input_dim, eps=eps)
        self.norm3 = nn.LayerNorm(input_dim, eps=eps)
        self.attn1 = SelfAttention(input_dim, n_head, attn_head_dim, dropout=dropout)
        self.attn2 = CrossAttention(input_dim, n_head, attn_head_dim, cross_attn_dim, dropout=dropout)
        self.ff    = TransformerFeedForward(input_dim, input_dim, act, dropout=dropout)
    
    def forward(self, x: torch.Tensor, extract: torch.Tensor | None = None) -> torch.Tensor:
        h = self.attn1(self.norm1(x)) + x
        if self.attn2 is not None:
            h = self.attn2(self.norm2(h), extract) + h
        h = self.ff(self.norm3(h)) + h
        return h
    
    def DeleteCrossAttentions(self) -> Self:
        self.norm2 = None
        self.attn2 = None
        return self


class DiffusionTransformerLayer(nn.Module):
    def __init__(
        self, 
        in_channel           : int, 
        n_head               : int        = 16, 
        attn_head_dim        : int        = 88,
        cross_attn_dim       : int | None = None, 
        act                  : str        = "geglu",
        num_group            : int        = 32, 
        num_layer            : int        = 1,
        dropout              : float      = 0., 
        eps                  : float      = 1e-5,
        is_linear_projection : bool       = True,
        is_spatial_attention : bool       = False
    ) -> None:
        super().__init__()
        self.in_channel  = in_channel
        self.out_channel = in_channel
        self.is_linear_projection = is_linear_projection
        self.is_spatial_attention = is_spatial_attention

        inner_dim  = n_head * attn_head_dim
        if is_linear_projection:
            proj_cls      = nn.Linear
            proj_in_args  = dict(in_features=in_channel, out_features=inner_dim)
            proj_out_args = dict(in_features=inner_dim, out_features=in_channel)
        else:
            proj_cls      = nn.Conv2d
            proj_in_args  = dict(in_channels=in_channel, out_channels=inner_dim, kernel_size=1, stride=1, padding=0)
            proj_out_args = dict(in_channels=inner_dim, out_channels=in_channel, kernel_size=1, stride=1, padding=0)
        
        self.norm               = nn.GroupNorm(num_group, in_channel, eps=1e-6, affine=True)
        self.proj_in            = proj_cls(**proj_in_args)
        self.proj_out           = proj_cls(**proj_out_args)
        self.transformer_blocks = nn.ModuleList([
            BasicTransformer(inner_dim, cross_attn_dim, n_head, attn_head_dim, act, dropout, eps)
            for _ in range(num_layer)
        ])
        
    def forward(self, x: torch.Tensor, extract: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm(x)
        h = self.input_proj(h)
        for block in self.transformer_blocks:
            h = block(h, extract)
        h = self.output_proj(h, x.size(-2))
        return h + x
    
    def input_proj(self, h: torch.Tensor) -> torch.Tensor:
        if self.is_spatial_attention:
            h = spatial_concat(h)
        if self.is_linear_projection:
            return self.proj_in(h.permute(0, 2, 3, 1).view(h.size(0), -1, self.in_channel))
        else:
            return self.proj_in(h).permute(0, 2, 3, 1).view(h.size(0), -1, self.in_channel)

    def output_proj(self, h: torch.Tensor, height: int) -> torch.Tensor:
        if self.is_linear_projection:
            h = self.proj_out(h).permute(0, 2, 1).view(h.size(0), self.out_channel, height, -1)
        else:
            h = self.proj_out(h.permute(0, 2, 1).view(h.size(0), self.out_channel, height, -1))
        if self.is_spatial_attention:
            return spatial_batch(h)
        return h


class DiffusionResLayer(nn.Module):
    def __init__(
        self, 
        in_channel      : int, 
        out_channel     : int, 
        temb_channel    : int, 
        other_channels  : list[int] | None = None, 
        num_group       : int              = 32, 
        act             : str              = "silu",
        dropout         : float            = 0., 
        output_scale    : float            = 1., 
        eps             : float            = 1e-6
    ) -> None:
        super().__init__()
        self.in_channel     = in_channel
        self.out_channel    = out_channel
        self.temb_channel   = temb_channel
        self.other_channels = other_channels
        self.output_scale   = 1. / output_scale

        if other_channels is None:
            self.norm1 = nn.GroupNorm(num_group, in_channel , eps=eps)
            self.norm2 = nn.GroupNorm(num_group, out_channel, eps=eps)
        else:
            self.norm1 = AdaGroupNorm(num_group, other_channels, in_channel , eps=eps)
            self.norm2 = AdaGroupNorm(num_group, other_channels, out_channel, eps=eps)
        
        self.time_emb_proj = nn.Linear(temb_channel, out_channel)

        self.conv1 = nn.Conv2d(in_channel , out_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)

        self.nonlinearity = get_activation(act, is_inplace=False)
        self.dropout = nn.Dropout(dropout)

        if in_channel != out_channel:
            self.conv_shortcut = nn.Conv2d(in_channel, out_channel, 1, 1, 0)
        else:
            self.conv_shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, temb: torch.Tensor, other_embeds: list[torch.Tensor] | None = None) -> torch.Tensor:
        if self.other_channels is None:
            h = self.norm1(x)
        else:
            h = self.norm1(x, other_embeds)

        h = self.conv1(self.nonlinearity(h))

        if self.other_channels is None:
            h = self.norm2(h + self.__time_embed_norm(temb))
        else:
            h = self.norm2(h + self.__time_embed_norm(temb), other_embeds)

        h = self.conv2(self.dropout(self.nonlinearity(h)))
        return (h + self.conv_shortcut(x)) * self.output_scale
    
    def __time_embed_norm(self, temb: torch.Tensor) -> torch.Tensor:
        return self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]