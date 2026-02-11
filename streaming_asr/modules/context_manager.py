"""
Context Manager for streaming ASR.
Manages cache slots for multi-stream/multi-speaker scenarios,
handles chunk scheduling and cache lifecycle.
"""

import torch
from torch import Tensor
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
import time


@dataclass
class StreamSlot:
    """Single streaming session's cache state."""
    slot_id: int
    cache: Optional[Dict[str, Tensor]] = None
    decoder_state: Optional[Tuple[Tensor, Tensor]] = None
    last_token: Optional[Tensor] = None
    partial_text: str = ""
    total_frames_processed: int = 0
    last_access_time: float = 0.0
    is_active: bool = True

    def touch(self):
        self.last_access_time = time.monotonic()


class ContextManager:
    """
    Manages cache state for multiple concurrent streaming sessions.

    Features:
    - Slot-based cache management for multi-stream scenarios
    - LRU eviction when slot limit reached
    - Cache state lifecycle (init, update, invalidate)
    - Batched cache assembly for efficient GPU processing
    - Chunk scheduling for variable-latency configurations

    This corresponds to the "Context Manager (Slot and Cache Lookup)"
    block in the streaming ASR architecture diagram.
    """

    def __init__(
        self,
        max_slots: int = 16,
        encoder_n_layers: int = 24,
        d_model: int = 512,
        cache_size: int = 70,
        conv_cache_size: int = 30,
        pred_hidden: int = 640,
        pred_rnn_layers: int = 1,
        vocab_size: int = 1024,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.max_slots = max_slots
        self.encoder_n_layers = encoder_n_layers
        self.d_model = d_model
        self.cache_size = cache_size
        self.conv_cache_size = conv_cache_size
        self.pred_hidden = pred_hidden
        self.pred_rnn_layers = pred_rnn_layers
        self.vocab_size = vocab_size
        self.device = device
        self.dtype = dtype

        self.slots: Dict[int, StreamSlot] = {}
        self._next_id = 0

    def create_slot(self) -> int:
        """
        Allocate a new streaming slot with fresh caches.
        Evicts LRU slot if at capacity.

        Returns:
            slot_id: unique identifier for this stream
        """
        if len(self.slots) >= self.max_slots:
            self._evict_lru()

        slot_id = self._next_id
        self._next_id += 1

        slot = StreamSlot(slot_id=slot_id)
        slot.cache = self._init_encoder_cache(batch_size=1)
        slot.decoder_state = self._init_decoder_state(batch_size=1)
        slot.last_token = torch.full(
            (1, 1), self.vocab_size, dtype=torch.long, device=self.device,
        )  # blank token
        slot.touch()

        self.slots[slot_id] = slot
        return slot_id

    def get_slot(self, slot_id: int) -> StreamSlot:
        """Retrieve a slot's cached state."""
        if slot_id not in self.slots:
            raise KeyError(f"Slot {slot_id} not found. Active slots: {list(self.slots.keys())}")
        slot = self.slots[slot_id]
        slot.touch()
        return slot

    def update_slot(
        self,
        slot_id: int,
        cache: Dict[str, Tensor],
        decoder_state: Tuple[Tensor, Tensor],
        last_token: Tensor,
        new_text: str = "",
    ):
        """Update a slot with new cache states after processing a chunk."""
        slot = self.slots[slot_id]
        slot.cache = cache
        slot.decoder_state = decoder_state
        slot.last_token = last_token
        slot.partial_text += new_text
        slot.total_frames_processed += 1
        slot.touch()

    def release_slot(self, slot_id: int) -> str:
        """
        Release a slot, returning final transcription.
        Frees GPU memory.
        """
        if slot_id not in self.slots:
            return ""
        slot = self.slots.pop(slot_id)
        text = slot.partial_text
        # Explicit cleanup
        del slot.cache
        del slot.decoder_state
        return text

    def batch_caches(
        self, slot_ids: List[int],
    ) -> Tuple[Dict[str, Tensor], Tuple[Tensor, Tensor], Tensor]:
        """
        Assemble batched caches from multiple slots for parallel processing.

        Args:
            slot_ids: list of slot IDs to batch together

        Returns:
            batched_cache: encoder cache dict with batch dim = len(slot_ids)
            batched_decoder_state: (h, c) batched LSTM states
            batched_last_tokens: (B, 1) last predicted tokens
        """
        B = len(slot_ids)
        slots = [self.get_slot(sid) for sid in slot_ids]

        # Stack encoder caches
        channel_caches = torch.cat(
            [s.cache["cache_last_channel"] for s in slots], dim=1,
        )  # (n_layers, B, cache_size, d_model)
        time_caches = torch.cat(
            [s.cache["cache_last_time"] for s in slots], dim=1,
        )  # (n_layers, B, d_model, conv_cache_size)
        channel_lens = torch.cat(
            [s.cache["cache_last_channel_len"] for s in slots], dim=0,
        )

        pre_encode_caches = None
        if slots[0].cache.get("cache_pre_encode") is not None:
            pre_encode_caches = torch.cat(
                [s.cache["cache_pre_encode"] for s in slots], dim=0,
            )

        batched_cache = {
            "cache_last_channel": channel_caches,
            "cache_last_time": time_caches,
            "cache_last_channel_len": channel_lens,
            "cache_pre_encode": pre_encode_caches,
        }

        # Stack decoder states
        h = torch.cat([s.decoder_state[0] for s in slots], dim=1)
        c = torch.cat([s.decoder_state[1] for s in slots], dim=1)
        batched_decoder_state = (h, c)

        # Stack last tokens
        batched_last_tokens = torch.cat(
            [s.last_token for s in slots], dim=0,
        )

        return batched_cache, batched_decoder_state, batched_last_tokens

    def unbatch_and_update(
        self,
        slot_ids: List[int],
        cache: Dict[str, Tensor],
        decoder_state: Tuple[Tensor, Tensor],
        last_tokens: Tensor,
        texts: List[str],
    ):
        """Scatter batched results back to individual slots."""
        for i, sid in enumerate(slot_ids):
            individual_cache = {
                "cache_last_channel": cache["cache_last_channel"][:, i:i+1, :, :],
                "cache_last_time": cache["cache_last_time"][:, i:i+1, :, :],
                "cache_last_channel_len": cache["cache_last_channel_len"][i:i+1],
                "cache_pre_encode": (
                    cache["cache_pre_encode"][i:i+1]
                    if cache.get("cache_pre_encode") is not None else None
                ),
            }
            individual_state = (
                decoder_state[0][:, i:i+1, :],
                decoder_state[1][:, i:i+1, :],
            )
            self.update_slot(
                sid, individual_cache, individual_state,
                last_tokens[i:i+1], texts[i] if i < len(texts) else "",
            )

    def _evict_lru(self):
        """Evict least-recently-used slot."""
        if not self.slots:
            return
        lru_id = min(self.slots, key=lambda k: self.slots[k].last_access_time)
        self.release_slot(lru_id)

    def _init_encoder_cache(self, batch_size: int) -> Dict[str, Tensor]:
        return {
            "cache_last_channel": torch.zeros(
                self.encoder_n_layers, batch_size, self.cache_size, self.d_model,
                device=self.device, dtype=self.dtype,
            ),
            "cache_last_time": torch.zeros(
                self.encoder_n_layers, batch_size, self.d_model, self.conv_cache_size,
                device=self.device, dtype=self.dtype,
            ),
            "cache_last_channel_len": torch.zeros(
                batch_size, device=self.device, dtype=torch.long,
            ),
            "cache_pre_encode": None,
        }

    def _init_decoder_state(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        h = torch.zeros(
            self.pred_rnn_layers, batch_size, self.pred_hidden,
            device=self.device, dtype=self.dtype,
        )
        return h, torch.zeros_like(h)

    @property
    def active_slot_count(self) -> int:
        return len(self.slots)

    def get_all_active_ids(self) -> List[int]:
        return list(self.slots.keys())
