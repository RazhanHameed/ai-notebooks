"""
Greedy RNN-T decoder for streaming inference.
Implements frame-synchronous greedy search with blank suppression.
"""

import torch
from torch import Tensor
from typing import Tuple, List, Optional


class GreedyRNNTDecoder:
    """
    Greedy frame-synchronous decoder for RNN-T.

    At each encoder frame:
    1. Compute joint(encoder_frame, prediction_state)
    2. If argmax != blank: emit token, update prediction state, repeat
    3. If argmax == blank: move to next encoder frame

    Optimizations:
    - Max symbols per step limit to prevent infinite loops
    - Batched single-step decoder forward
    - Optional CUDA graph capture for the decode loop
    """

    def __init__(
        self,
        decoder,       # RNNTDecoder module
        joint,         # JointNetwork module
        vocab_size: int,
        blank_id: int,
        max_symbols_per_step: int = 10,
    ):
        self.decoder = decoder
        self.joint = joint
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.max_symbols_per_step = max_symbols_per_step

    @torch.inference_mode()
    def decode(
        self,
        encoder_out: Tensor,   # (B, T, D)
        encoder_lengths: Tensor,
        decoder_state: Optional[Tuple[Tensor, Tensor]] = None,
        last_token: Optional[Tensor] = None,
    ) -> Tuple[List[List[int]], Tuple[Tensor, Tensor], Tensor]:
        """
        Greedy decode over encoder output.

        Args:
            encoder_out: (B, T, D_enc) encoder features
            encoder_lengths: (B,) valid encoder frame counts
            decoder_state: initial (h, c) from previous chunk, or None
            last_token: (B, 1) last predicted token from previous chunk

        Returns:
            hypotheses: list of B token-id lists
            final_state: (h, c) for next chunk
            last_tokens: (B, 1) last predicted tokens
        """
        B, T, D = encoder_out.shape
        device = encoder_out.device

        # Initialize decoder state
        if decoder_state is None:
            decoder_state = self.decoder.init_state(B, device, encoder_out.dtype)

        # Initialize with blank token
        if last_token is None:
            last_token = torch.full(
                (B, 1), self.blank_id, dtype=torch.long, device=device,
            )

        hypotheses = [[] for _ in range(B)]

        for t in range(T):
            # Get encoder frame: (B, 1, D)
            enc_frame = encoder_out[:, t:t+1, :]

            # Inner loop: emit tokens until blank
            for _ in range(self.max_symbols_per_step):
                # Decoder single step
                pred_out, decoder_state_new = self.decoder.forward_one_step(
                    last_token, decoder_state,
                )
                # pred_out: (B, 1, D_pred)

                # Joint: (B, 1, 1, V+1)
                logits = self.joint.forward_one_step(enc_frame, pred_out)
                logits = logits.squeeze(1).squeeze(1)  # (B, V+1)

                # Greedy: argmax
                pred_tokens = logits.argmax(dim=-1)  # (B,)

                # Check which samples emitted blank
                is_blank = (pred_tokens == self.blank_id)

                # Update hypotheses for non-blank predictions
                for b in range(B):
                    if t < encoder_lengths[b] and not is_blank[b]:
                        hypotheses[b].append(pred_tokens[b].item())

                # Update state and last_token only for non-blank
                if not is_blank.all():
                    # For samples that predicted non-blank, accept new state
                    for b in range(B):
                        if not is_blank[b]:
                            decoder_state = (
                                _update_state_dim(
                                    decoder_state[0], decoder_state_new[0], b
                                ),
                                _update_state_dim(
                                    decoder_state[1], decoder_state_new[1], b
                                ),
                            )
                            last_token[b, 0] = pred_tokens[b]

                # If all samples emitted blank, move to next frame
                if is_blank.all():
                    break

        return hypotheses, decoder_state, last_token

    @torch.inference_mode()
    def decode_streaming_chunk(
        self,
        encoder_out: Tensor,
        decoder_state: Tuple[Tensor, Tensor],
        last_token: Tensor,
    ) -> Tuple[List[int], Tuple[Tensor, Tensor], Tensor]:
        """
        Decode a single streaming chunk (batch=1).
        Optimized single-sample path.

        Returns:
            tokens: list of emitted token IDs
            state: updated decoder state
            last_token: updated last token
        """
        T = encoder_out.shape[1]
        device = encoder_out.device
        tokens = []

        for t in range(T):
            enc_frame = encoder_out[:, t:t+1, :]

            for _ in range(self.max_symbols_per_step):
                pred_out, new_state = self.decoder.forward_one_step(
                    last_token, decoder_state,
                )

                logits = self.joint.forward_one_step(enc_frame, pred_out)
                pred_id = logits.squeeze().argmax().item()

                if pred_id == self.blank_id:
                    break

                tokens.append(pred_id)
                decoder_state = new_state
                last_token = torch.tensor(
                    [[pred_id]], dtype=torch.long, device=device,
                )

        return tokens, decoder_state, last_token


def _update_state_dim(
    old: Tensor, new: Tensor, batch_idx: int,
) -> Tensor:
    """Update a specific batch element in LSTM state."""
    result = old.clone()
    result[:, batch_idx, :] = new[:, batch_idx, :]
    return result
