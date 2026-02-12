"""
StreamingASRModel: Top-level model combining all components.
Cache-Aware FastConformer RNN-T for streaming speech recognition.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, Optional, Tuple, List

from .modules.preprocessing import AudioPreprocessor
from .modules.conformer_encoder import CacheAwareConformerEncoder
from .modules.rnnt_decoder import RNNTDecoder
from .modules.joint_network import JointNetwork
from .modules.greedy_decoder import GreedyRNNTDecoder


class StreamingASRModel(nn.Module):
    """
    Full streaming ASR model: Preprocessor → Encoder → Decoder → Joint.

    Matches the architecture of nvidia/nemotron-speech-streaming-en-0.6b:
    - AudioPreprocessor: mel spectrogram (80 bins, 16kHz)
    - CacheAwareConformerEncoder: 24-layer FastConformer with caches
    - RNNTDecoder: LSTM prediction network
    - JointNetwork: encoder_proj + pred_proj → logits

    Supports both full-utterance and streaming inference.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        # Convert window_size/stride from seconds to samples if needed
        sample_rate = config.get("sample_rate", 16000)
        win_length = config.get("win_length", 0.025)
        hop_length = config.get("hop_length", 0.01)
        if isinstance(win_length, float) and win_length < 1.0:
            win_length = int(win_length * sample_rate)
        if isinstance(hop_length, float) and hop_length < 1.0:
            hop_length = int(hop_length * sample_rate)

        # Preprocessor
        self.preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            n_mels=config.get("n_mels", 80),
            n_fft=config.get("n_fft", 512),
            win_length=win_length,
            hop_length=hop_length,
            preemph=config.get("preemph", 0.97),
            dither=config.get("dither", 0.0),
            normalize=config.get("normalize", "per_feature"),
        )

        # Encoder
        self.encoder = CacheAwareConformerEncoder(
            feat_in=config.get("feat_in", 80),
            d_model=config.get("d_model", 512),
            d_ff=config.get("d_ff", 2048),
            n_layers=config.get("n_layers", 24),
            n_heads=config.get("n_heads", 8),
            conv_kernel_size=config.get("conv_kernel_size", 31),
            subsampling_factor=config.get("subsampling_factor", 4),
            subsampling_channels=config.get("subsampling_channels", 256),
            dropout=config.get("dropout", 0.1),
            dropout_att=config.get("dropout_att", 0.0),
            att_context_size=config.get("att_context_size", (70, 13)),
            att_context_style=config.get("att_context_style", "regular"),
            use_bias=config.get("use_bias", True),
            conv_norm_type=config.get("conv_norm_type", "batch_norm"),
        )

        # Decoder (prediction network)
        vocab_size = config.get("vocab_size", 1024)
        pred_hidden = config.get("pred_hidden", 640)
        self.decoder = RNNTDecoder(
            vocab_size=vocab_size,
            pred_hidden=pred_hidden,
            pred_rnn_layers=config.get("pred_rnn_layers", 1),
            dropout=config.get("dropout", 0.1),
        )

        # Joint network
        self.joint = JointNetwork(
            encoder_hidden=config.get("d_model", 512),
            pred_hidden=pred_hidden,
            joint_hidden=config.get("joint_hidden", 640),
            vocab_size=vocab_size,
            activation=config.get("joint_activation", "relu"),
        )

        self.vocab_size = vocab_size
        self.blank_id = vocab_size

        # Greedy decoder
        self.greedy = GreedyRNNTDecoder(
            decoder=self.decoder,
            joint=self.joint,
            vocab_size=vocab_size,
            blank_id=self.blank_id,
        )

    def forward(
        self,
        audio: Tensor,
        audio_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Full forward pass (non-streaming).

        Args:
            audio: (B, T_samples) raw waveform
            audio_lengths: (B,) sample counts

        Returns:
            encoded: (B, T_enc, D)
            encoded_lengths: (B,)
        """
        # Preprocess
        features, feat_lengths = self.preprocessor(audio, audio_lengths)

        # (B, n_mels, T) → (B, T, n_mels) for encoder
        features = features.transpose(1, 2)

        # Encode
        encoded, encoded_lengths = self.encoder(features, feat_lengths)

        return encoded, encoded_lengths

    @torch.inference_mode()
    def transcribe(
        self,
        audio: Tensor,
        audio_lengths: Tensor,
    ) -> List[List[int]]:
        """
        Full-utterance transcription (non-streaming).

        Args:
            audio: (B, T_samples) raw waveform at 16kHz
            audio_lengths: (B,)

        Returns:
            List of B token-ID sequences
        """
        self.eval()
        encoded, enc_lengths = self.forward(audio, audio_lengths)
        hypotheses, _, _ = self.greedy.decode(encoded, enc_lengths)
        return hypotheses

    @torch.inference_mode()
    def init_streaming(
        self, batch_size: int = 1,
        device: torch.device = None, dtype: torch.dtype = None,
    ) -> Dict[str, Any]:
        """
        Initialize streaming state.

        Returns a state dict to pass to transcribe_chunk().
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        return {
            "encoder_cache": self.encoder.init_cache(batch_size, device, dtype),
            "decoder_state": self.decoder.init_state(batch_size, device, dtype),
            "last_token": torch.full(
                (batch_size, 1), self.blank_id,
                dtype=torch.long, device=device,
            ),
            "preprocessor_state": None,
        }

    @torch.inference_mode()
    def transcribe_chunk(
        self,
        audio_chunk: Tensor,      # (B, T_samples)
        audio_lengths: Tensor,    # (B,)
        state: Dict[str, Any],
    ) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        Streaming transcription of a single audio chunk.

        Args:
            audio_chunk: (B, T_samples) raw waveform chunk
            audio_lengths: (B,) valid sample counts
            state: from init_streaming() or previous transcribe_chunk()

        Returns:
            tokens: list of B token-ID lists for this chunk
            new_state: updated state for next chunk
        """
        self.eval()

        # Preprocess
        features, feat_lengths = self.preprocessor(audio_chunk, audio_lengths)
        features = features.transpose(1, 2)  # (B, T, n_mels)

        # Encode with cache
        encoded, enc_lengths, new_enc_cache = self.encoder.forward_streaming(
            features, feat_lengths, state["encoder_cache"],
        )

        # Greedy decode with decoder state
        tokens, new_dec_state, new_last_token = self.greedy.decode(
            encoded, enc_lengths,
            decoder_state=state["decoder_state"],
            last_token=state["last_token"],
        )

        new_state = {
            "encoder_cache": new_enc_cache,
            "decoder_state": new_dec_state,
            "last_token": new_last_token,
            "preprocessor_state": None,
        }

        return tokens, new_state

    @classmethod
    def from_nemo(cls, nemo_path: str, device: str = "cpu") -> "StreamingASRModel":
        """
        Load model from a .nemo checkpoint file.

        Args:
            nemo_path: path to .nemo file
            device: target device

        Returns:
            initialized and weight-loaded StreamingASRModel
        """
        from .utils.nemo_converter import NemoConverter

        converter = NemoConverter(nemo_path)
        config = converter.get_model_config()

        print("Model config:")
        for k, v in config.items():
            print(f"  {k}: {v}")

        model = cls(config)
        loaded, missing = converter.load_weights_into_model(model)

        model = model.to(device)
        model.eval()

        # Store tokenizer path for decode
        try:
            model._tokenizer_path = converter.get_tokenizer_path()
        except FileNotFoundError:
            model._tokenizer_path = None

        return model

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
