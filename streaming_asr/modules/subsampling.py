"""
Convolutional subsampling for FastConformer encoder.
Reduces temporal resolution by a factor of 4 or 8 using strided convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional


class ConvSubsampling(nn.Module):
    """
    Depthwise-separable strided convolutional subsampling.

    NeMo's FastConformer uses 'dw_striding' subsampling by default:
    - Two depthwise-separable conv blocks with stride 2 each → 4x subsampling
    - Each block: depthwise conv → pointwise conv → activation → norm

    For streaming, maintains a left-context cache for causal convolution.
    """

    def __init__(
        self,
        feat_in: int = 80,
        feat_out: int = 512,
        conv_channels: int = 256,
        subsampling_factor: int = 4,
        activation: str = "relu",
    ):
        super().__init__()
        self.feat_in = feat_in
        self.feat_out = feat_out
        self.subsampling_factor = subsampling_factor

        assert subsampling_factor in (2, 4, 8), \
            f"subsampling_factor must be 2, 4, or 8, got {subsampling_factor}"

        n_blocks = {2: 1, 4: 2, 8: 3}[subsampling_factor]

        act_fn = nn.SiLU() if activation == "silu" else nn.ReLU()

        layers = []
        in_ch = 1  # Conv2D: (B, 1, T, F) → treats mel as 2D image
        out_ch = conv_channels

        for i in range(n_blocks):
            layers.extend([
                nn.Conv2d(
                    in_ch, out_ch,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(out_ch),
                act_fn,
            ])
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)

        # Compute output frequency dimension after subsampling
        freq_out = feat_in
        for _ in range(n_blocks):
            freq_out = (freq_out + 2 * 1 - 3) // 2 + 1  # conv2d formula

        self.linear = nn.Linear(conv_channels * freq_out, feat_out)
        self.norm = nn.LayerNorm(feat_out)

        # Cache for streaming
        self._pre_encode_cache_size = self._compute_cache_size(n_blocks)

    def _compute_cache_size(self, n_blocks: int) -> int:
        """
        Compute how many input frames need to be cached for causal streaming.
        Each conv block with kernel=3, stride=2 needs 2 extra left-context frames.
        """
        cache = 0
        for _ in range(n_blocks):
            cache = cache * 2 + 2  # account for stride and kernel
        return cache

    @property
    def pre_encode_cache_size(self) -> int:
        return self._pre_encode_cache_size

    def forward(
        self,
        x: Tensor,
        lengths: Tensor,
        cache: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Args:
            x: (B, T, feat_in) input mel features
            lengths: (B,) valid frame counts
            cache: (B, cache_size, feat_in) optional left-context cache

        Returns:
            out: (B, T_sub, feat_out) subsampled output
            out_lengths: (B,) output frame counts
            new_cache: (B, cache_size, feat_in) updated cache
        """
        # Prepend cache for causal streaming
        new_cache = None
        if cache is not None:
            new_cache = x[:, -self._pre_encode_cache_size:, :].clone()
            x = torch.cat([cache, x], dim=1)
        elif self.training is False and self._pre_encode_cache_size > 0:
            new_cache = x[:, -self._pre_encode_cache_size:, :].clone()

        B, T, F = x.shape
        # (B, T, F) → (B, 1, T, F) for Conv2D
        x = x.unsqueeze(1)

        x = self.conv(x)  # (B, C, T_sub, F_sub)

        B, C, T_sub, F_sub = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T_sub, C * F_sub)

        x = self.linear(x)
        x = self.norm(x)

        # Compute output lengths
        out_lengths = lengths
        n_blocks = {2: 1, 4: 2, 8: 3}[self.subsampling_factor]
        for _ in range(n_blocks):
            out_lengths = (out_lengths + 2 * 1 - 3) // 2 + 1

        return x, out_lengths, new_cache
