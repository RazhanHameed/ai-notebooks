"""
RNN-T Joint Network.
Combines encoder and prediction network outputs to produce logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from ..kernels.fused_ops import fused_joint_net


class JointNetwork(nn.Module):
    """
    RNN-T Joint Network.

    Architecture:
        encoder_proj(enc) + decoder_proj(pred) → ReLU → Linear → logits

    The joint network broadcasts over time dimensions:
        enc: (B, T, 1, H_joint) + pred: (B, 1, U, H_joint) → (B, T, U, V+1)

    Optimizations:
    - Fused add + ReLU + linear via Triton kernel
    - Supports log-softmax temperature scaling
    """

    def __init__(
        self,
        encoder_hidden: int,
        pred_hidden: int,
        joint_hidden: int,
        vocab_size: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder_hidden = encoder_hidden
        self.pred_hidden = pred_hidden
        self.joint_hidden = joint_hidden
        self.vocab_size = vocab_size
        self.num_classes = vocab_size + 1  # +1 for blank

        # Projections
        self.enc_proj = nn.Linear(encoder_hidden, joint_hidden)
        self.pred_proj = nn.Linear(pred_hidden, joint_hidden)

        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Output projection
        self.output = nn.Linear(joint_hidden, self.num_classes)

    def forward(
        self,
        encoder_out: Tensor,    # (B, T, D_enc)
        pred_out: Tensor,       # (B, U, D_pred)
        use_fused: bool = True,
    ) -> Tensor:
        """
        Compute joint network output.

        Args:
            encoder_out: (B, T, D_enc) encoder output
            pred_out: (B, U, D_pred) prediction network output
            use_fused: try to use fused Triton kernel

        Returns:
            logits: (B, T, U, num_classes)
        """
        # Project to joint dimension
        enc = self.enc_proj(encoder_out)    # (B, T, H)
        pred = self.pred_proj(pred_out)     # (B, U, H)

        # Broadcast: (B, T, 1, H) + (B, 1, U, H) → (B, T, U, H)
        enc = enc.unsqueeze(2)
        pred = pred.unsqueeze(1)

        # Try fused kernel
        if use_fused and enc.is_cuda:
            try:
                return fused_joint_net(
                    enc, pred, self.output.weight, self.output.bias,
                )
            except Exception:
                pass  # fallback to standard path

        # Standard path
        joint = enc + pred
        joint = self.activation(joint)
        joint = self.dropout(joint)
        logits = self.output(joint)

        return logits

    @torch.inference_mode()
    def forward_one_step(
        self,
        encoder_out: Tensor,    # (B, 1, D_enc) single encoder frame
        pred_out: Tensor,       # (B, 1, D_pred) single prediction output
    ) -> Tensor:
        """
        Single-step joint for greedy decoding.

        Returns:
            logits: (B, 1, 1, num_classes)
        """
        enc = self.enc_proj(encoder_out).unsqueeze(2)   # (B, 1, 1, H)
        pred = self.pred_proj(pred_out).unsqueeze(1)     # (B, 1, 1, H)
        joint = F.relu(enc + pred)
        return self.output(joint)
