"""
RNN-T Prediction Network (Decoder).
LSTM-based prediction network that models label dependencies.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, List


class RNNTDecoder(nn.Module):
    """
    RNN-T prediction network using LSTM.

    Architecture:
    - Embedding layer (vocab_size + 1 for blank)
    - Stacked LSTM layers with optional projection
    - Dropout regularization

    For streaming: maintains LSTM hidden/cell states across chunks.
    """

    def __init__(
        self,
        vocab_size: int,
        pred_hidden: int = 640,
        pred_rnn_layers: int = 1,
        dropout: float = 0.1,
        blank_as_pad: bool = True,
        forget_gate_bias: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pred_hidden = pred_hidden
        self.pred_rnn_layers = pred_rnn_layers
        self.blank_idx = vocab_size  # blank is last token

        # Embedding: vocab_size + 1 (for blank token)
        self.embedding = nn.Embedding(
            vocab_size + 1, pred_hidden,
            padding_idx=self.blank_idx if blank_as_pad else None,
        )

        # LSTM prediction network
        self.lstm = nn.LSTM(
            input_size=pred_hidden,
            hidden_size=pred_hidden,
            num_layers=pred_rnn_layers,
            batch_first=True,
            dropout=dropout if pred_rnn_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Initialize forget gate bias
        self._init_forget_gate(forget_gate_bias)

    def _init_forget_gate(self, bias: float):
        """Initialize LSTM forget gate bias to a positive value."""
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                n = param.size(0)
                # LSTM bias layout: [input, forget, cell, output]
                start = n // 4
                end = n // 2
                param.data[start:end].fill_(bias)

    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32,
    ) -> Tuple[Tensor, Tensor]:
        """
        Initialize LSTM hidden/cell states.

        Returns:
            h: (num_layers, B, hidden) hidden state
            c: (num_layers, B, hidden) cell state
        """
        h = torch.zeros(
            self.pred_rnn_layers, batch_size, self.pred_hidden,
            device=device, dtype=dtype,
        )
        c = torch.zeros_like(h)
        return h, c

    def forward(
        self,
        targets: Tensor,                                    # (B, U)
        target_lengths: Optional[Tensor] = None,
        state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Full forward pass over target sequence.

        Args:
            targets: (B, U) token indices
            target_lengths: (B,) valid lengths
            state: optional (h, c) initial states

        Returns:
            pred: (B, U, pred_hidden) prediction network output
            state: (h, c) final LSTM states
        """
        B = targets.shape[0]

        if state is None:
            state = self.init_state(B, targets.device, targets.dtype
                                     if targets.is_floating_point()
                                     else torch.float32)

        emb = self.embedding(targets)  # (B, U, pred_hidden)
        emb = self.dropout(emb)

        # Pack for variable lengths if provided
        if target_lengths is not None:
            emb = nn.utils.rnn.pack_padded_sequence(
                emb, target_lengths.cpu(), batch_first=True, enforce_sorted=False,
            )

        pred, state = self.lstm(emb, state)

        if target_lengths is not None:
            pred, _ = nn.utils.rnn.pad_packed_sequence(pred, batch_first=True)

        pred = self.dropout(pred)
        return pred, state

    @torch.inference_mode()
    def forward_one_step(
        self,
        token: Tensor,                                # (B, 1) single token
        state: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Single-step forward for greedy decoding.

        Args:
            token: (B, 1) single token index
            state: (h, c) LSTM states

        Returns:
            pred: (B, 1, pred_hidden)
            new_state: updated (h, c)
        """
        emb = self.embedding(token)  # (B, 1, pred_hidden)
        pred, state = self.lstm(emb, state)
        return pred, state
