"""
Cache-Aware FastConformer Encoder.
Full encoder stack with streaming cache management.

Architecture (nemotron-speech-streaming-en-0.6b):
- Depthwise-separable convolutional subsampling (4x)
- 24 Conformer layers with relative position attention
- Cache-aware: each layer maintains MHA + Conv caches
- Configurable chunk sizes via att_context_size
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from .subsampling import ConvSubsampling
from .conformer_layer import ConformerLayer
from .multi_head_attention import RelPositionalEncoding


@dataclass
class StreamingConfig:
    """Configuration for cache-aware streaming inference."""
    chunk_size: int           # input chunk length in subsampled frames
    shift_size: int           # stride between chunks
    valid_out_len: int        # valid output frames per chunk
    cache_size: int           # left-context cache size per layer (MHA)
    conv_cache_size: int      # convolution cache size per layer
    cache_drop_size: int      # frames to drop from cache
    pre_encode_cache_size: int  # subsampling cache size


class CacheAwareConformerEncoder(nn.Module):
    """
    24-layer Cache-Aware FastConformer encoder for streaming ASR.

    Features:
    - Depthwise-separable strided subsampling (4x reduction)
    - Relative positional encoding (Transformer-XL style)
    - Per-layer MHA cache (attention key/value context)
    - Per-layer convolution cache (causal depthwise conv context)
    - Configurable latency via att_context_size
    """

    def __init__(
        self,
        feat_in: int = 80,
        d_model: int = 512,
        d_ff: int = 2048,
        n_layers: int = 24,
        n_heads: int = 8,
        conv_kernel_size: int = 31,
        subsampling_factor: int = 4,
        subsampling_channels: int = 256,
        dropout: float = 0.1,
        dropout_att: float = 0.0,
        att_context_size: Tuple[int, int] = (70, 13),
        att_context_style: str = "regular",
        use_bias: bool = True,
        conv_norm_type: str = "batch_norm",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.att_context_size = att_context_size
        self.att_context_style = att_context_style

        # Subsampling
        self.subsampling = ConvSubsampling(
            feat_in=feat_in,
            feat_out=d_model,
            conv_channels=subsampling_channels,
            subsampling_factor=subsampling_factor,
        )

        # Positional encoding
        self.pos_enc = RelPositionalEncoding(d_model, max_len=5000, xscale=True)

        # Conformer layers
        self.layers = nn.ModuleList([
            ConformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                dropout_att=dropout_att,
                use_bias=use_bias,
                conv_norm_type=conv_norm_type,
            )
            for _ in range(n_layers)
        ])

        # Compute streaming config
        self._streaming_config = self._compute_streaming_config(
            att_context_size, att_context_style, n_layers,
            conv_kernel_size, subsampling_factor,
        )

    @staticmethod
    def _compute_streaming_config(
        att_context_size: Tuple[int, int],
        att_context_style: str,
        n_layers: int,
        conv_kernel_size: int,
        subsampling_factor: int,
    ) -> StreamingConfig:
        """Derive streaming cache sizes from attention context config."""
        left_ctx = att_context_size[0]
        right_ctx = att_context_size[1]
        conv_ctx = (conv_kernel_size - 1) // 2

        if att_context_style == "chunked_limited":
            lookahead = right_ctx
            cache_drop = 0
        else:  # "regular"
            lookahead = right_ctx * n_layers + conv_ctx * n_layers
            cache_drop = lookahead

        # Chunk size in subsampled frames
        chunk_size = right_ctx + 1
        shift_size = chunk_size
        valid_out_len = chunk_size

        return StreamingConfig(
            chunk_size=chunk_size,
            shift_size=shift_size,
            valid_out_len=valid_out_len,
            cache_size=left_ctx,
            conv_cache_size=conv_kernel_size - 1,
            cache_drop_size=cache_drop,
            pre_encode_cache_size=2 * subsampling_factor,
        )

    @property
    def streaming_config(self) -> StreamingConfig:
        return self._streaming_config

    def init_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32,
    ) -> Dict[str, Tensor]:
        """
        Initialize empty caches for streaming inference.

        Returns dict with:
            - cache_last_channel: (n_layers, B, cache_size, d_model) - MHA cache
            - cache_last_time: (n_layers, B, d_model, conv_cache_size) - conv cache
            - cache_last_channel_len: (B,) - valid cache lengths
            - cache_pre_encode: (B, pre_encode_cache_size, feat_in) - subsampling cache
        """
        cfg = self._streaming_config
        return {
            "cache_last_channel": torch.zeros(
                self.n_layers, batch_size, cfg.cache_size, self.d_model,
                device=device, dtype=dtype,
            ),
            "cache_last_time": torch.zeros(
                self.n_layers, batch_size, self.d_model, cfg.conv_cache_size,
                device=device, dtype=dtype,
            ),
            "cache_last_channel_len": torch.zeros(
                batch_size, device=device, dtype=torch.long,
            ),
            "cache_pre_encode": None,  # initialized on first chunk
        }

    def forward(
        self,
        x: Tensor,
        lengths: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Full (non-streaming) forward pass.

        Args:
            x: (B, T, feat_in) mel features
            lengths: (B,) frame counts

        Returns:
            encoded: (B, T_sub, d_model)
            encoded_lengths: (B,)
        """
        # Subsampling
        x, lengths, _ = self.subsampling(x, lengths)

        # Positional encoding
        x, pos_emb = self.pos_enc(x)

        # Conformer layers
        for layer in self.layers:
            x, _, _ = layer(x, pos_emb)

        return x, lengths

    @torch.inference_mode()
    def forward_streaming(
        self,
        x: Tensor,
        lengths: Tensor,
        cache: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Cache-aware streaming forward pass.

        Processes one chunk at a time, using cached states from
        previous chunks for context.

        Args:
            x: (B, T_chunk, feat_in) mel features for this chunk
            lengths: (B,) frame counts for this chunk
            cache: dict from init_cache() or previous call

        Returns:
            encoded: (B, T_out, d_model) encoded output for this chunk
            out_lengths: (B,)
            new_cache: updated cache dict for next chunk
        """
        cfg = self._streaming_config

        # Subsampling with cache
        x, lengths, pre_cache = self.subsampling(
            x, lengths, cache=cache.get("cache_pre_encode"),
        )

        # Positional encoding
        # For streaming, we need pos encoding that covers cache + current
        total_len = x.shape[1] + cfg.cache_size
        pos_input = torch.zeros(
            1, total_len, self.d_model, device=x.device, dtype=x.dtype,
        )
        _, pos_emb = self.pos_enc(pos_input)

        # Process through layers with caching
        new_channel_cache = []
        new_time_cache = []

        for i, layer in enumerate(self.layers):
            attn_cache = cache["cache_last_channel"][i]  # (B, cache_size, D)
            conv_cache = cache["cache_last_time"][i]      # (B, D, conv_size)

            x, new_attn, new_conv = layer(
                x, pos_emb[:, :x.shape[1] + attn_cache.shape[1], :],
                attn_cache=attn_cache,
                conv_cache=conv_cache,
                cache_len=cache["cache_last_channel_len"],
            )

            # Update caches: keep most recent cache_size frames
            if new_attn.shape[1] > cfg.cache_size:
                new_attn = new_attn[:, -cfg.cache_size:, :]
            elif new_attn.shape[1] < cfg.cache_size:
                pad = torch.zeros(
                    new_attn.shape[0],
                    cfg.cache_size - new_attn.shape[1],
                    self.d_model,
                    device=new_attn.device, dtype=new_attn.dtype,
                )
                new_attn = torch.cat([pad, new_attn], dim=1)

            new_channel_cache.append(new_attn)
            new_time_cache.append(new_conv)

        # Stack caches
        new_cache = {
            "cache_last_channel": torch.stack(new_channel_cache, dim=0),
            "cache_last_time": torch.stack(new_time_cache, dim=0),
            "cache_last_channel_len": cache["cache_last_channel_len"] + lengths,
            "cache_pre_encode": pre_cache,
        }

        return x, lengths, new_cache
