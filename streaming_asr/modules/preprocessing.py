"""
Audio preprocessing: Mel spectrogram extraction with streaming support.
Matches NeMo's AudioToMelSpectrogramPreprocessor defaults.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False


class AudioPreprocessor(nn.Module):
    """
    Streaming-capable mel spectrogram extraction.

    Matches the exact configuration used by nemotron-speech-streaming-en-0.6b:
    - 16kHz sample rate
    - 80 mel filterbanks
    - 25ms window, 10ms hop
    - Preemphasis, dithering, per-feature normalization
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        win_length: int = 400,     # 25ms at 16kHz
        hop_length: int = 160,     # 10ms at 16kHz
        window_fn: str = "hann",
        preemph: float = 0.97,
        dither: float = 0.0,       # disabled at inference
        normalize: str = "per_feature",
        log: bool = True,
        log_zero_guard_type: str = "add",
        log_zero_guard_value: float = 2**-24,
        pad_to: int = 0,
        pad_value: float = 0.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.preemph = preemph
        self.dither = dither
        self.normalize = normalize
        self.log = log
        self.log_zero_guard_value = log_zero_guard_value
        self.pad_to = pad_to
        self.pad_value = pad_value

        # Build window
        if window_fn == "hann":
            self.register_buffer("window", torch.hann_window(win_length))
        elif window_fn == "hamming":
            self.register_buffer("window", torch.hamming_window(win_length))
        else:
            self.register_buffer("window", torch.hann_window(win_length))

        # Mel filterbank
        mel_fb = self._create_mel_filterbank(
            n_fft=n_fft, n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=0.0, f_max=sample_rate / 2.0,
        )
        self.register_buffer("mel_fb", mel_fb)

        # Streaming state: carry over samples from previous chunk
        self._prev_samples: Optional[Tensor] = None

    def _create_mel_filterbank(
        self, n_fft: int, n_mels: int, sample_rate: int,
        f_min: float, f_max: float,
    ) -> Tensor:
        """Create mel filterbank matrix (n_fft//2+1, n_mels) using Slaney normalization."""
        if HAS_TORCHAUDIO:
            fb = torchaudio.functional.melscale_fbanks(
                n_freqs=n_fft // 2 + 1,
                f_min=f_min,
                f_max=f_max,
                n_mels=n_mels,
                sample_rate=sample_rate,
                norm="slaney",
                mel_scale="slaney",
            )
            return fb  # (n_fft//2+1, n_mels)

        # Manual mel filterbank (HTK-compatible)
        low_mel = self._hz_to_mel(f_min)
        high_mel = self._hz_to_mel(f_max)
        mel_points = torch.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_points = (hz_points * n_fft / sample_rate).long()

        fb = torch.zeros(n_fft // 2 + 1, n_mels)
        for i in range(n_mels):
            left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
            for j in range(left, center):
                fb[j, i] = (j - left) / max(center - left, 1)
            for j in range(center, right):
                fb[j, i] = (right - j) / max(right - center, 1)
            # Slaney normalization
            enorm = 2.0 / (hz_points[i + 2] - hz_points[i])
            fb[:, i] *= enorm
        return fb

    @staticmethod
    def _hz_to_mel(hz):
        return 2595.0 * torch.log10(1.0 + hz / 700.0) if isinstance(hz, Tensor) \
            else 2595.0 * math.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def reset_stream(self):
        """Reset streaming state for a new utterance."""
        self._prev_samples = None

    @torch.no_grad()
    def forward(
        self, audio: Tensor, audio_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            audio: (B, T) raw waveform at self.sample_rate
            audio_lengths: (B,) valid sample counts

        Returns:
            features: (B, n_mels, T_frames)
            feature_lengths: (B,)
        """
        B = audio.shape[0]

        # Preemphasis
        if self.preemph > 0:
            audio = torch.cat(
                [audio[:, :1], audio[:, 1:] - self.preemph * audio[:, :-1]],
                dim=1,
            )

        # Dithering (training only)
        if self.dither > 0 and self.training:
            audio = audio + self.dither * torch.randn_like(audio)

        # STFT
        # Pad window to n_fft if needed
        window = self.window
        if self.win_length < self.n_fft:
            pad_left = (self.n_fft - self.win_length) // 2
            pad_right = self.n_fft - self.win_length - pad_left
            window = F.pad(window, (pad_left, pad_right))

        stft = torch.stft(
            audio, n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            return_complex=True,
        )
        power_spec = stft.abs().pow(2)  # (B, n_fft//2+1, T_frames)

        # Mel filterbank
        mel = torch.matmul(
            power_spec.transpose(1, 2), self.mel_fb
        ).transpose(1, 2)  # (B, n_mels, T_frames)

        # Log
        if self.log:
            mel = torch.log(mel + self.log_zero_guard_value)

        # Per-feature normalization (mean/var over time for each frequency)
        if self.normalize == "per_feature":
            T = mel.shape[-1]
            mean = mel.mean(dim=-1, keepdim=True)
            std = mel.std(dim=-1, keepdim=True).clamp(min=1e-5)
            mel = (mel - mean) / std

        # Compute output lengths
        feature_lengths = torch.div(
            audio_lengths, self.hop_length, rounding_mode='floor'
        ) + 1

        # Pad to multiple of pad_to
        if self.pad_to > 0:
            T = mel.shape[-1]
            pad_amt = (self.pad_to - T % self.pad_to) % self.pad_to
            if pad_amt > 0:
                mel = F.pad(mel, (0, pad_amt), value=self.pad_value)

        return mel, feature_lengths

    @torch.no_grad()
    def forward_chunk(self, audio_chunk: Tensor) -> Tensor:
        """
        Process a single streaming chunk. Maintains internal state
        for seamless frame boundaries.

        Args:
            audio_chunk: (1, T) raw waveform chunk

        Returns:
            features: (1, n_mels, T_frames)
        """
        if self._prev_samples is not None:
            audio_chunk = torch.cat([self._prev_samples, audio_chunk], dim=1)

        # Save overlap for next chunk
        overlap = self.win_length - self.hop_length
        if audio_chunk.shape[1] > overlap:
            self._prev_samples = audio_chunk[:, -overlap:]
        else:
            self._prev_samples = audio_chunk

        lengths = torch.tensor([audio_chunk.shape[1]], device=audio_chunk.device)
        features, _ = self.forward(audio_chunk, lengths)
        return features
