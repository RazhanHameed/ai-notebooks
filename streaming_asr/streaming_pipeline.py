"""
Streaming ASR Pipeline Orchestrator.
Implements the full streaming flow from the architecture diagram:

  Audio Stream → Chunk Splitter → Feature Buffer → Context Manager →
  Cache-Aware Encoder → RNN-T Decoder → Text Output → Cache Update

Supports:
- Real-time microphone streaming
- File-based streaming simulation
- Multi-stream concurrent decoding
- Configurable chunk sizes for latency/accuracy tradeoff
"""

import time
import math
import threading
import queue
from typing import Optional, Callable, List, Tuple, Dict, Any

import torch
from torch import Tensor

from .model import StreamingASRModel
from .modules.context_manager import ContextManager
from .utils.tokenizer import SentencePieceTokenizer


class ChunkSplitter:
    """
    Splits continuous audio into fixed-size chunks (default 80ms).
    Handles sample-level buffering across calls.
    """

    def __init__(
        self,
        chunk_duration_ms: float = 80.0,
        sample_rate: int = 16000,
    ):
        self.chunk_samples = int(chunk_duration_ms * sample_rate / 1000)
        self.sample_rate = sample_rate
        self._buffer = torch.zeros(0)

    def reset(self):
        self._buffer = torch.zeros(0)

    def feed(self, audio: Tensor) -> List[Tensor]:
        """
        Feed audio samples and get back complete chunks.

        Args:
            audio: (T,) raw audio samples

        Returns:
            list of (chunk_samples,) tensors, one per complete chunk
        """
        # Append to buffer
        self._buffer = torch.cat([self._buffer, audio.cpu().flatten()])

        chunks = []
        while self._buffer.shape[0] >= self.chunk_samples:
            chunk = self._buffer[:self.chunk_samples]
            self._buffer = self._buffer[self.chunk_samples:]
            chunks.append(chunk)

        return chunks

    def flush(self) -> Optional[Tensor]:
        """Flush remaining buffered audio (zero-padded to chunk size)."""
        if self._buffer.shape[0] == 0:
            return None
        padded = torch.zeros(self.chunk_samples)
        padded[:self._buffer.shape[0]] = self._buffer
        self._buffer = torch.zeros(0)
        return padded


class FeatureBuffer:
    """
    Accumulates mel spectrogram chunks for the encoder.
    Manages multi-chunk buffering when encoder needs more frames
    than a single 80ms chunk provides.
    """

    def __init__(
        self,
        n_mels: int = 80,
        frames_per_chunk: int = 8,  # typical: 80ms / 10ms = 8 frames
        chunks_per_encode: int = 14,  # att_context_size[1] + 1
    ):
        self.n_mels = n_mels
        self.frames_per_chunk = frames_per_chunk
        self.chunks_per_encode = chunks_per_encode
        self._buffer: Optional[Tensor] = None
        self._chunk_count = 0

    def reset(self):
        self._buffer = None
        self._chunk_count = 0

    def add_chunk(self, features: Tensor) -> Optional[Tensor]:
        """
        Add a mel chunk. Returns batched features when enough chunks
        have been accumulated for one encoder forward pass.

        Args:
            features: (1, n_mels, T_frames) mel spectrogram chunk

        Returns:
            None if more chunks needed, else (1, T_total, n_mels)
        """
        if self._buffer is None:
            self._buffer = features
        else:
            self._buffer = torch.cat([self._buffer, features], dim=2)

        self._chunk_count += 1

        if self._chunk_count >= self.chunks_per_encode:
            out = self._buffer.transpose(1, 2)  # (1, T, n_mels)
            self._buffer = None
            self._chunk_count = 0
            return out

        return None


class StreamingPipeline:
    """
    Full streaming ASR pipeline orchestrator.

    Implements the architecture diagram flow with all optimizations:
    - Chunk splitting at configurable intervals
    - Feature buffering and mel extraction
    - Context management for cache state
    - Cache-aware encoding + RNN-T decoding
    - Cache update after each encoder step
    """

    def __init__(
        self,
        model: StreamingASRModel,
        tokenizer: Optional[SentencePieceTokenizer] = None,
        chunk_duration_ms: float = 80.0,
        att_context_size: Tuple[int, int] = (70, 13),
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

        # Chunk splitter (80ms default per diagram)
        self.chunk_splitter = ChunkSplitter(
            chunk_duration_ms=chunk_duration_ms,
            sample_rate=model.config.get("sample_rate", 16000),
        )

        # Encoder chunk size = att_context_size[1] + 1 chunks
        chunks_per_encode = att_context_size[1] + 1
        self.feature_buffer = FeatureBuffer(
            n_mels=model.config.get("n_mels", 80),
            chunks_per_encode=chunks_per_encode,
        )

        # Context manager
        cfg = model.config
        self.context_manager = ContextManager(
            max_slots=16,
            encoder_n_layers=cfg.get("n_layers", 24),
            d_model=cfg.get("d_model", 512),
            cache_size=att_context_size[0],
            conv_cache_size=cfg.get("conv_kernel_size", 31) - 1,
            pred_hidden=cfg.get("pred_hidden", 640),
            pred_rnn_layers=cfg.get("pred_rnn_layers", 1),
            vocab_size=cfg.get("vocab_size", 1024),
            device=device,
            dtype=torch.float32,
        )

    def start_stream(self) -> int:
        """
        Start a new streaming session.
        Returns a slot_id for referencing this stream.
        """
        slot_id = self.context_manager.create_slot()
        self.chunk_splitter.reset()
        self.feature_buffer.reset()
        return slot_id

    def process_audio(
        self,
        slot_id: int,
        audio: Tensor,
        final: bool = False,
    ) -> str:
        """
        Process incoming audio for a streaming session.

        This is the main entry point implementing the diagram flow:
        1. Audio → Chunk Splitter
        2. Chunks → Feature extraction → Feature Buffer
        3. Feature Buffer (full) → Context Manager lookup → Encoder → Decoder
        4. Cache update → Context Manager store
        5. Return new text

        Args:
            slot_id: stream slot from start_stream()
            audio: (T,) raw audio samples at 16kHz
            final: True if this is the last audio for this stream

        Returns:
            Newly decoded text for this audio segment
        """
        # Step 1: Split into chunks
        chunks = self.chunk_splitter.feed(audio.to("cpu"))
        if final:
            last = self.chunk_splitter.flush()
            if last is not None:
                chunks.append(last)

        new_text_parts = []

        for chunk in chunks:
            # Step 2: Extract features
            chunk_tensor = chunk.unsqueeze(0).to(self.device)  # (1, T_samples)
            features = self.model.preprocessor.forward_chunk(chunk_tensor)
            # features: (1, n_mels, T_frames)

            # Step 3: Buffer features until we have enough for encoder
            encoder_input = self.feature_buffer.add_chunk(features)

            if encoder_input is not None:
                # Step 4: Get cached state from context manager
                slot = self.context_manager.get_slot(slot_id)

                # Step 5: Encoder forward with cache
                with torch.inference_mode():
                    lengths = torch.tensor(
                        [encoder_input.shape[1]], device=self.device,
                    )
                    encoded, enc_lengths, new_cache = \
                        self.model.encoder.forward_streaming(
                            encoder_input, lengths, slot.cache,
                        )

                    # Step 6: RNN-T decode
                    tokens, new_dec_state, new_last_token = \
                        self.model.greedy.decode(
                            encoded, enc_lengths,
                            decoder_state=slot.decoder_state,
                            last_token=slot.last_token,
                        )

                # Step 7: Decode tokens to text
                if tokens[0] and self.tokenizer:
                    text = self.tokenizer.decode(tokens[0])
                    new_text_parts.append(text)
                elif tokens[0]:
                    new_text_parts.append(str(tokens[0]))

                # Step 8: Update cache in context manager
                self.context_manager.update_slot(
                    slot_id,
                    cache=new_cache,
                    decoder_state=new_dec_state,
                    last_token=new_last_token,
                    new_text=" ".join(new_text_parts),
                )

        return " ".join(new_text_parts)

    def end_stream(self, slot_id: int) -> str:
        """
        End a streaming session. Returns final transcription.
        Flushes any remaining audio and frees resources.
        """
        # Flush remaining
        final_text = self.process_audio(slot_id, torch.zeros(0), final=True)
        full_text = self.context_manager.release_slot(slot_id)
        return full_text

    @torch.inference_mode()
    def transcribe_file(
        self,
        audio: Tensor,
        sample_rate: int = 16000,
        show_progress: bool = True,
    ) -> str:
        """
        Simulate streaming on a complete audio file.

        Args:
            audio: (T,) complete waveform
            sample_rate: audio sample rate (resampled if != 16kHz)

        Returns:
            Full transcription text
        """
        if sample_rate != 16000:
            try:
                import torchaudio
                audio = torchaudio.functional.resample(audio, sample_rate, 16000)
            except ImportError:
                raise ValueError(
                    f"Audio is {sample_rate}Hz but torchaudio not available for resampling"
                )

        slot_id = self.start_stream()
        chunk_samples = self.chunk_splitter.chunk_samples

        total_chunks = math.ceil(audio.shape[0] / chunk_samples)
        text_parts = []

        for i in range(0, audio.shape[0], chunk_samples):
            chunk = audio[i:i + chunk_samples]
            is_final = (i + chunk_samples >= audio.shape[0])
            text = self.process_audio(slot_id, chunk, final=is_final)
            if text:
                text_parts.append(text)
                if show_progress:
                    elapsed_sec = (i + chunk_samples) / 16000
                    print(f"[{elapsed_sec:.1f}s] {text}")

        full_text = self.end_stream(slot_id)
        return full_text


class AsyncStreamingPipeline:
    """
    Async wrapper for real-time streaming with separate
    audio ingestion and text output threads.
    """

    def __init__(
        self,
        pipeline: StreamingPipeline,
        on_text: Optional[Callable[[str], None]] = None,
    ):
        self.pipeline = pipeline
        self.on_text = on_text or (lambda t: print(f"[ASR] {t}"))
        self._audio_queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._slot_id: Optional[int] = None

    def start(self):
        """Start async processing."""
        self._slot_id = self.pipeline.start_stream()
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def feed_audio(self, audio: Tensor):
        """Feed audio samples (thread-safe)."""
        self._audio_queue.put(audio)

    def stop(self) -> str:
        """Stop processing, return final text."""
        self._running = False
        self._audio_queue.put(None)  # sentinel
        if self._thread:
            self._thread.join(timeout=5.0)
        text = self.pipeline.end_stream(self._slot_id)
        return text

    def _process_loop(self):
        while self._running:
            try:
                audio = self._audio_queue.get(timeout=0.1)
                if audio is None:
                    break
                text = self.pipeline.process_audio(self._slot_id, audio)
                if text:
                    self.on_text(text)
            except queue.Empty:
                continue
