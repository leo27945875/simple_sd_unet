import torch
import torch.nn as nn

from typing import Any

from .block import (
    get_time_block,
    get_down_block,
    get_up_block,
    get_mid_block,
)
from . import (
    AbsDiffusionCondDownBlock,
    AbsDiffusionCondUpBlock,
    AbsDiffusionCondMidBlock
)
from .external import (
    UNet2DOutput,
    UNet2DConditionOutput
)
from .activation import get_activation
from .utils import get_dict_default


class Cond_UNet2DModel(nn.Module):
    def __init__(
        self, cfg: dict[str, Any], **kwargs
    ) -> None:
        super().__init__()
        # Parameters:
        in_channels           : int                     = get_dict_default(cfg, "in_channels", 4)
        out_channels          : int                     = get_dict_default(cfg, "out_channels", 4)
        flip_sin_to_cos       : bool                    = get_dict_default(cfg, "flip_sin_to_cos", True)
        freq_shift            : int                     = get_dict_default(cfg, "freq_shift", 0)
        down_block_types      : tuple[str]              = get_dict_default(cfg, "down_block_types", ("CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","DownBlock2D"))
        mid_block_type        : str | None              = get_dict_default(cfg, "mid_block_type", "UNetMidBlock2DCrossAttn")
        up_block_types        : tuple[str]              = get_dict_default(cfg, "up_block_types", ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"))
        block_out_channels    : tuple[int]              = get_dict_default(cfg, "block_out_channels", (320, 640, 1280, 1280))
        layers_per_block      : int | tuple[int]        = get_dict_default(cfg, "layers_per_block", 2)
        dropout               : float                   = get_dict_default(cfg, "dropout", 0.)
        act_fn                : str                     = get_dict_default(cfg, "act_fn", "silu")
        norm_num_groups       : int | None              = get_dict_default(cfg, "norm_num_groups", 32)
        norm_eps              : float                   = get_dict_default(cfg, "norm_eps", 1e-5)
        cross_attention_dim   : int | tuple[int] | None = get_dict_default(cfg, "cross_attention_dim", 1024)
        attention_head_dim    : int | tuple[int]        = get_dict_default(cfg, "attention_head_dim", (5, 10, 20, 20))
        num_attention_heads   : int | tuple[int] | None = get_dict_default(cfg, "num_attention_heads", None)
        time_embedding_type   : str                     = get_dict_default(cfg, "time_embedding_type", "positional")
        use_linear_projection : bool                    = get_dict_default(cfg, "use_linear_projection", True)
        num_class_embeds      : int | None              = get_dict_default(cfg, "num_class_embeds", None)

        # To adapt "diffuers" variable naming issue (https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131)
        if num_attention_heads is not None:
            raise ValueError(
                "At the moment it is not possible to define the number of attention heads via `num_attention_heads` because of a naming issue as described in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131. Passing `num_attention_heads` will only be supported in diffusers v0.19."
            )

        self.in_channels          = in_channels
        self.out_channels         = out_channels
        self.block_out_channels   = block_out_channels
        self.mid_block_channels   = block_out_channels[-1]
        self.time_embed_type      = time_embedding_type
        self.time_embed_dim       = block_out_channels[0] * 4
        self.n_half_block         = len(block_out_channels) + 1
        self.attention_head_dim   = (attention_head_dim,) * self.n_half_block if isinstance(attention_head_dim, int) else tuple(attention_head_dim)
        self.cross_attention_dims = (cross_attention_dim,) * self.n_half_block if cross_attention_dim is None or isinstance(cross_attention_dim, int) else tuple(cross_attention_dim)
        
        down_blocks = []
        out_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            in_channel     = out_channel
            out_channel    = block_out_channels[i]
            is_final_block = (i == len(block_out_channels) - 1)
            down_blocks.append(
                get_down_block(
                    down_block_type, 
                    in_channel, 
                    out_channel, 
                    self.time_embed_dim, 
                    None,
                    self.cross_attention_dims[i],
                    layers_per_block, 
                    norm_num_groups, 
                    self.attention_head_dim[i], 
                    act_fn, 
                    dropout, 
                    norm_eps, 
                    not is_final_block,
                    use_linear_projection
                )
            )

        up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        out_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            skip_channel   = out_channel
            out_channel    = reversed_block_out_channels[i]
            in_channel     = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            is_final_block = (i == len(block_out_channels) - 1)
            up_blocks.append(
                get_up_block(
                    up_block_type, 
                    in_channel, 
                    out_channel, 
                    skip_channel, 
                    self.time_embed_dim, 
                    None, 
                    self.cross_attention_dims[-i-1], 
                    layers_per_block + 1, 
                    norm_num_groups, 
                    self.attention_head_dim[-i-1], 
                    act_fn, 
                    dropout, 
                    norm_eps, 
                    not is_final_block,
                    use_linear_projection
                )
            )

        mid_block = get_mid_block(
            mid_block_type, 
            self.mid_block_channels, 
            self.time_embed_dim, 
            None, 
            self.cross_attention_dims[-1], 
            1, 
            norm_num_groups, 
            self.attention_head_dim[-1], 
            act_fn, 
            dropout, 
            norm_eps,
            use_linear_projection
        )

        # Timestep blocks:
        self.time_proj, self.time_embedding = get_time_block(time_embedding_type, block_out_channels[0], self.time_embed_dim, flip_sin_to_cos, freq_shift)
        
        # Class embedding layer:
        self.class_embedding = nn.Embedding(num_class_embeds, self.time_embed_dim) if num_class_embeds else None

        # Encoder & Decoder blocks:
        self.conv_in       = nn.Conv2d(in_channels, block_out_channels[0], 3, 1, 1)
        self.down_blocks   = nn.ModuleList(down_blocks)
        self.mid_block     = mid_block
        self.up_blocks     = nn.ModuleList(up_blocks)
        self.conv_norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[0], eps=norm_eps)
        self.conv_act      = get_activation(act_fn, block_out_channels[0])
        self.conv_out      = nn.Conv2d(block_out_channels[0], out_channels, 3, 1, 1)
    
    def forward(
        self,
        sample                               : torch.FloatTensor,
        timestep                             : torch.Tensor | float | int,
        encoder_hidden_states                : torch.Tensor | None         = None,
        class_labels                         : torch.Tensor | None         = None,
        down_intrablock_additional_residuals : list[torch.Tensor] | None   = None,
        return_dict                          : bool                        = True
    ) -> UNet2DConditionOutput | tuple[torch.Tensor]:
        
        dtype, device, B = sample.dtype, sample.device, sample.shape[0]

        # Process timesteps:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, dtype=torch.float, device=device)
        if len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device, torch.float)
            timesteps = timesteps.expand(B)
        temb = self.time_embedding(self.time_proj(timesteps).to(dtype)) # [Timesteps] always returns float32 tensor.
 
        # Class embeddings:
        if self.class_embedding is not None:
            temb += self.class_embedding(class_labels)

        # Adapter residuals:
        if down_intrablock_additional_residuals is None:
            down_intrablock_additional_residuals = [None] * len(self.down_blocks)

        # Initial conv:
        h = self.conv_in(sample)

        # Encoder blocks:
        res_hidden_states_list = [h]
        for i, downBlock in enumerate(self.down_blocks):
            if isinstance(downBlock, AbsDiffusionCondDownBlock):
                h, out_hidden_states = downBlock(h, temb, None, encoder_hidden_states, down_intrablock_additional_residuals[i])
            else:
                h, out_hidden_states = downBlock(h, temb, None, down_intrablock_additional_residuals[i])

            res_hidden_states_list.extend(out_hidden_states)

        # Middle block:
        h = self.mid_block(h, temb, None, encoder_hidden_states) if isinstance(self.mid_block, AbsDiffusionCondMidBlock) else self.mid_block(h, temb)

        # Decoder blocks:
        for i, up_block in enumerate(self.up_blocks, 2):
            res_hidden_states      = res_hidden_states_list[-len(up_block.resnets):]
            res_hidden_states_list = res_hidden_states_list[:-len(up_block.resnets)]
            if isinstance(up_block, AbsDiffusionCondUpBlock):
                h = up_block(h, res_hidden_states, temb, None, encoder_hidden_states)
            else:
                h = up_block(h, res_hidden_states, temb, None)
        
        # Final conv:
        h = self.conv_out(self.conv_act(self.conv_norm_out(h)))

        # Post-processing:
        if self.time_embed_type == "fourier":
            timesteps = timesteps.view(B, *([1] * (sample.dim() - 1)))
            h = h / timesteps
        
        if return_dict:
            return UNet2DConditionOutput(sample=h)
        
        return (h,)


class Uncond_UNet2DModel(nn.Module):
    def __init__(
        self,
        cfg: dict[str, Any], **kwargs
    ) -> None:
        super().__init__()
        # Parameters:
        in_channels               : int        = get_dict_default(cfg, "in_channels", 3)
        out_channels              : int        = get_dict_default(cfg, "out_channels", 3)
        time_embedding_type       : str        = get_dict_default(cfg, "time_embedding_type", "positional")
        freq_shift                : int        = get_dict_default(cfg, "freq_shift", 0)
        flip_sin_to_cos           : bool       = get_dict_default(cfg, "flip_sin_to_cos", True)
        down_block_types          : tuple[str] = get_dict_default(cfg, "down_block_types", ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"))
        up_block_types            : tuple[str] = get_dict_default(cfg, "up_block_types", ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"))
        block_out_channels        : tuple[int] = get_dict_default(cfg, "block_out_channels", (224, 448, 672, 896))
        layers_per_block          : int        = get_dict_default(cfg, "layers_per_block", 2)
        dropout                   : float      = get_dict_default(cfg, "dropout", 0.)
        act_fn                    : str        = get_dict_default(cfg, "act_fn", "silu")
        attention_head_dim        : int | None = get_dict_default(cfg, "attention_head_dim", 8)
        norm_num_groups           : int        = get_dict_default(cfg, "norm_num_groups", 32)
        norm_eps                  : float      = get_dict_default(cfg, "norm_eps", 1e-5)

        self.in_channels          = in_channels
        self.out_channels         = out_channels
        self.block_out_channels   = block_out_channels
        self.mid_block_channels   = block_out_channels[-1]
        self.time_embed_dim       = block_out_channels[0] * 4
        self.time_embed_type      = time_embedding_type
        self.n_half_block         = len(block_out_channels) + 1

        time_embed_dim = self.time_embed_dim

        down_blocks = []
        out_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            in_channel   = out_channel
            out_channel  = block_out_channels[i]
            is_final_blk = (i == len(block_out_channels) - 1)
            down_blocks.append(get_down_block(
                down_block_type, in_channel, out_channel, time_embed_dim, None, None, layers_per_block, norm_num_groups, attention_head_dim, act_fn, dropout, norm_eps, not is_final_blk
            ))

        up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        out_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            skip_channel = out_channel
            out_channel  = reversed_block_out_channels[i]
            in_channel   = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            is_final_blk = (i == len(block_out_channels) - 1)
            up_blocks.append(get_up_block(
                up_block_type, in_channel, out_channel, skip_channel, time_embed_dim, None, None, layers_per_block + 1, norm_num_groups, attention_head_dim, act_fn, dropout, norm_eps, not is_final_blk
            ))

        # Timestep blocks:
        self.time_proj, self.time_embedding = get_time_block(time_embedding_type, block_out_channels[0], time_embed_dim, flip_sin_to_cos, freq_shift)
        
        # Encoder & Decoder blocks:
        self.conv_in       = nn.Conv2d(in_channels, block_out_channels[0], 3, 1, 1)
        self.down_blocks   = nn.ModuleList(down_blocks)
        self.mid_block     = get_mid_block("UNetMidBlock2D", self.mid_block_channels, time_embed_dim, None, None, 1, norm_num_groups, attention_head_dim, act_fn, dropout, norm_eps)
        self.up_blocks     = nn.ModuleList(up_blocks)
        self.conv_norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[0], eps=norm_eps)
        self.conv_act      = get_activation(act_fn, block_out_channels[0])
        self.conv_out      = nn.Conv2d(block_out_channels[0], out_channels, 3, 1, 1)
    
    def forward(
        self,
        sample           : torch.FloatTensor,
        timestep         : torch.Tensor | float | int,
        return_dict      : bool                        = True
    ) -> UNet2DOutput | tuple[torch.Tensor]:
        
        dtype, device, B = sample.dtype, sample.device, sample.shape[0]
        
        # Process timesteps:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, dtype=torch.float, device=device)
        if len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device, torch.float)
            timesteps = timesteps.expand(B)
        
        temb = self.time_embedding(self.time_proj(timesteps).to(dtype)) # [Timesteps] always returns float32 tensor.
 
        # Initial conv:
        h = self.conv_in(sample)

        # Encoder blocks:
        res_hidden_states_list = [h]
        for down_block in self.down_blocks:
            h, out_hidden_states = down_block(h, temb)
            res_hidden_states_list.extend(out_hidden_states)

        # Middle block:
        h = self.mid_block(h, temb)

        # Decoder blocks:
        for up_block in self.up_blocks:
            res_hidden_states      = res_hidden_states_list[-len(up_block.resnets):]
            res_hidden_states_list = res_hidden_states_list[:-len(up_block.resnets)]
            h = up_block(h, res_hidden_states, temb)
        
        # Final conv:
        h = self.conv_out(self.conv_act(self.conv_norm_out(h)))

        # Post-processing:
        if self.time_embed_type == "fourier":
            timesteps = timesteps.view(B, *([1] * (sample.dim() - 1)))
            h = h / timesteps
        
        if return_dict:
            return UNet2DOutput(sample=h)
        
        return (h,)