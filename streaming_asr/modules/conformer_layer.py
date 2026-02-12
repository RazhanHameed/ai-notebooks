"""
Single Conformer layer with cache-aware streaming support.
Implements the Macaron-style sandwich architecture:
  FFN → MHA → Conv → FFN → LayerNorm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

from ..kernels.fused_ops import fused_swish, fused_add_norm


class ConvolutionModule(nn.Module):
    """
    Conformer convolution module:
    LayerNorm → Pointwise Conv (2x expand) → GLU → Depthwise Conv →
    BatchNorm → Swish → Pointwise Conv → Dropout

    For streaming: uses causal depthwise convolution with left-context cache.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 31,
        norm_type: str = "batch_norm",
        use_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size

        self.layer_norm = nn.LayerNorm(d_model)

        # Pointwise expand 1x → 2x (for GLU gating)
        self.pointwise_conv1 = nn.Conv1d(
            d_model, 2 * d_model, kernel_size=1, bias=use_bias,
        )

        # Depthwise conv (groups=d_model for true depthwise)
        # Causal: pad only on left side
        self.causal_pad = kernel_size - 1
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=0,  # we handle padding manually for causal
            bias=use_bias,
        )

        if norm_type == "batch_norm":
            self.norm = nn.BatchNorm1d(d_model)
        elif norm_type == "layer_norm":
            self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = nn.BatchNorm1d(d_model)

        # Pointwise compress 1x → 1x
        self.pointwise_conv2 = nn.Conv1d(
            d_model, d_model, kernel_size=1, bias=use_bias,
        )
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        x: Tensor,                         # (B, T, D)
        cache: Optional[Tensor] = None,    # (B, D, cache_size)
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x: (B, T, D)
            cache: (B, D, kernel_size-1) left-context for causal conv

        Returns:
            out: (B, T, D)
            new_cache: (B, D, kernel_size-1)
        """
        residual = x
        x = self.layer_norm(x)

        # (B, T, D) → (B, D, T) for conv1d
        x = x.transpose(1, 2)

        # Pointwise conv → GLU gating
        x = self.pointwise_conv1(x)  # (B, 2D, T)
        x = F.glu(x, dim=1)          # (B, D, T)

        # Causal depthwise convolution with cache
        if cache is not None:
            x = torch.cat([cache, x], dim=2)  # prepend cached context
            new_cache = x[:, :, -(self.kernel_size - 1):].clone()
        else:
            # Zero-pad left for causal
            x = F.pad(x, (self.causal_pad, 0))
            new_cache = x[:, :, -(self.kernel_size - 1):].clone()

        x = self.depthwise_conv(x)  # (B, D, T)

        # Normalization
        if isinstance(self.norm, nn.BatchNorm1d):
            x = self.norm(x)
        else:
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)

        # Swish activation (fused Triton kernel if available)
        x = fused_swish(x)

        # Pointwise conv
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        # Back to (B, T, D)
        x = x.transpose(1, 2)
        return x + residual, new_cache


class FeedForwardModule(nn.Module):
    """
    Conformer feed-forward module:
    LayerNorm → Linear → Swish → Dropout → Linear → Dropout
    Scaled by 0.5 (Macaron-style).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        use_bias: bool = True,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """(B, T, D) → (B, T, D)"""
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = fused_swish(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return residual + 0.5 * x


class ConformerLayer(nn.Module):
    """
    Single Conformer block (Macaron-style):
      x → FFN_1 → MHA → Conv → FFN_2 → LayerNorm → out

    With residual connections at each sub-module.
    Cache-aware for streaming: stores MHA key/value cache and Conv cache.
    """

    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        n_heads: int = 8,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        dropout_att: float = 0.0,
        use_bias: bool = True,
        conv_norm_type: str = "batch_norm",
    ):
        super().__init__()
        from .multi_head_attention import RelPositionMultiHeadAttention

        self.ffn1 = FeedForwardModule(d_model, d_ff, dropout, use_bias)
        self.self_attn = RelPositionMultiHeadAttention(
            d_model, n_heads, dropout_att, use_bias,
        )
        self.conv = ConvolutionModule(
            d_model, conv_kernel_size, conv_norm_type, use_bias,
        )
        self.ffn2 = FeedForwardModule(d_model, d_ff, dropout, use_bias)
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Pre-attention layer norm
        self.norm_attn = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        mask: Optional[Tensor] = None,
        attn_cache: Optional[Tensor] = None,
        conv_cache: Optional[Tensor] = None,
        cache_len: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Args:
            x: (B, T, D)
            pos_emb: (1, L, D) relative position encoding
            mask: attention mask
            attn_cache: (B, T_cache, D) key cache from previous chunk
            conv_cache: (B, D, kernel_size-1) conv cache from previous chunk
            cache_len: (B,) valid cache lengths

        Returns:
            out: (B, T, D)
            new_attn_cache: (B, T_new, D)
            new_conv_cache: (B, D, kernel_size-1)
        """
        # FFN 1 (Macaron)
        x = self.ffn1(x)

        # Self-attention with relative position
        attn_in = self.norm_attn(x)
        if attn_cache is not None:
            kv = torch.cat([attn_cache, attn_in], dim=1)
        else:
            kv = attn_in

        attn_out, _ = self.self_attn(
            query=attn_in,
            key=kv,
            value=kv,
            pos_emb=pos_emb,
            mask=mask,
            cache=None,  # we handle caching at this level
            cache_len=cache_len,
        )
        x = x + self.dropout(attn_out)

        # New attention cache: store the processed input for next chunk
        new_attn_cache = attn_in

        # Convolution module
        x, new_conv_cache = self.conv(x, cache=conv_cache)

        # FFN 2 (Macaron)
        x = self.ffn2(x)

        # Final layer norm
        x = self.norm_out(x)

        return x, new_attn_cache, new_conv_cache
