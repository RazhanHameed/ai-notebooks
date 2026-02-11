#!/usr/bin/env python3
"""
Streaming ASR inference demo.

Demonstrates:
1. Loading converted model (or converting from .nemo on-the-fly)
2. Full-file transcription (simulated streaming)
3. Real-time streaming with chunk-by-chunk processing
4. Multi-stream concurrent decoding
5. Benchmarking mode

Usage:
    # From converted checkpoint:
    python inference.py --model_dir ./converted_model --audio test.wav

    # Directly from .nemo:
    python inference.py --nemo_path ./checkpoints/model.nemo --audio test.wav

    # Benchmark mode:
    python inference.py --model_dir ./converted_model --benchmark

    # Streaming simulation:
    python inference.py --model_dir ./converted_model --audio test.wav --streaming
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from streaming_asr.model import StreamingASRModel
from streaming_asr.streaming_pipeline import StreamingPipeline
from streaming_asr.utils.tokenizer import SentencePieceTokenizer
from streaming_asr.kernels.cuda_extensions import (
    optimize_for_inference,
    apply_half_precision,
    InferenceProfiler,
)


def load_audio(path: str, target_sr: int = 16000):
    """Load audio file and resample to target sample rate."""
    try:
        import torchaudio
        waveform, sr = torchaudio.load(path)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0), target_sr
    except ImportError:
        try:
            import soundfile as sf
            import numpy as np
            data, sr = sf.read(path)
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            waveform = torch.from_numpy(data).float()
            if sr != target_sr:
                raise RuntimeError(
                    f"Audio is {sr}Hz, need {target_sr}Hz. "
                    "Install torchaudio for resampling."
                )
            return waveform, target_sr
        except ImportError:
            raise ImportError(
                "Install torchaudio or soundfile: pip install torchaudio soundfile"
            )


def load_model(args):
    """Load model from converted checkpoint or .nemo file."""
    device = torch.device(args.device)

    if args.model_dir and os.path.isdir(args.model_dir):
        print(f"Loading converted model from {args.model_dir}...")
        model_path = os.path.join(args.model_dir, "model.pt")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        model = StreamingASRModel(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device).eval()

        # Load tokenizer
        tokenizer = None
        tok_path = os.path.join(args.model_dir, "tokenizer.model")
        if os.path.isfile(tok_path):
            tokenizer = SentencePieceTokenizer(tok_path)
            print(f"Loaded tokenizer: vocab_size={tokenizer.vocab_size}")

        return model, tokenizer

    elif args.nemo_path:
        print(f"Loading from .nemo: {args.nemo_path}...")
        model = StreamingASRModel.from_nemo(args.nemo_path, device=str(device))

        tokenizer = None
        if hasattr(model, "_tokenizer_path") and model._tokenizer_path:
            tokenizer = SentencePieceTokenizer(model._tokenizer_path)

        return model, tokenizer

    else:
        raise ValueError("Provide either --model_dir or --nemo_path")


def transcribe_full(model, tokenizer, audio, device):
    """Full (non-streaming) transcription."""
    print("\n--- Full Transcription ---")
    audio = audio.unsqueeze(0).to(device)
    lengths = torch.tensor([audio.shape[1]], device=device)

    start = time.perf_counter()
    tokens = model.transcribe(audio, lengths)
    elapsed = time.perf_counter() - start

    text = tokenizer.decode(tokens[0]) if tokenizer else str(tokens[0])
    audio_duration = audio.shape[1] / 16000

    print(f"Text: {text}")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Inference time: {elapsed:.3f}s")
    print(f"RTF: {elapsed / audio_duration:.3f}x")
    return text


def transcribe_streaming(model, tokenizer, audio, device, chunk_ms=80.0):
    """Simulated streaming transcription."""
    print(f"\n--- Streaming Transcription (chunk={chunk_ms}ms) ---")

    att_ctx = model.config.get("att_context_size", (70, 13))
    pipeline = StreamingPipeline(
        model=model,
        tokenizer=tokenizer,
        chunk_duration_ms=chunk_ms,
        att_context_size=att_ctx,
        device=device,
    )

    start = time.perf_counter()
    text = pipeline.transcribe_file(audio, show_progress=True)
    elapsed = time.perf_counter() - start

    audio_duration = audio.shape[0] / 16000
    print(f"\nFinal text: {text}")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Total time: {elapsed:.3f}s")
    print(f"RTF: {elapsed / audio_duration:.3f}x")
    return text


def benchmark(model, device, n_iterations=50):
    """
    Benchmark inference speed with synthetic audio.
    Tests both full and streaming modes.
    """
    print("\n--- Benchmark ---")
    profiler = InferenceProfiler()

    # Warmup
    dummy = torch.randn(1, 16000, device=device)
    dummy_len = torch.tensor([16000], device=device)
    for _ in range(5):
        with torch.inference_mode():
            model.forward(dummy, dummy_len)

    # Full forward benchmark (1 second audio)
    print(f"\nFull forward pass ({n_iterations} iterations, 1s audio):")
    for i in range(n_iterations):
        profiler.start("full_forward")
        with torch.inference_mode():
            encoded, lengths = model.forward(dummy, dummy_len)
        profiler.stop("full_forward")

    # Streaming chunk benchmark (80ms chunks)
    print(f"\nStreaming chunk ({n_iterations} iterations, 80ms chunk):")
    chunk = torch.randn(1, 1280, device=device)  # 80ms at 16kHz
    chunk_len = torch.tensor([1280], device=device)
    state = model.init_streaming(1, device)
    for i in range(n_iterations):
        profiler.start("stream_chunk")
        with torch.inference_mode():
            tokens, state = model.transcribe_chunk(chunk, chunk_len, state)
        profiler.stop("stream_chunk")

    # Greedy decode benchmark
    print(f"\nGreedy decode ({n_iterations} iterations, 25 encoder frames):")
    enc_out = torch.randn(1, 25, model.config["d_model"], device=device)
    enc_len = torch.tensor([25], device=device)
    for i in range(n_iterations):
        profiler.start("greedy_decode")
        with torch.inference_mode():
            model.greedy.decode(enc_out, enc_len)
        profiler.stop("greedy_decode")

    print("\n" + profiler.summary())

    # Memory usage
    if torch.cuda.is_available():
        print(f"\nGPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Streaming ASR Inference")
    parser.add_argument("--model_dir", help="Path to converted model directory")
    parser.add_argument("--nemo_path", help="Path to .nemo checkpoint")
    parser.add_argument("--audio", help="Path to audio file (.wav)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode")
    parser.add_argument("--chunk_ms", type=float, default=80.0, help="Chunk duration in ms")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args)
    device = torch.device(args.device)

    # Apply optimizations
    if args.half:
        print("Applying FP16 optimization...")
        model = apply_half_precision(model)

    if args.compile:
        print("Applying torch.compile...")
        model.encoder = optimize_for_inference(model.encoder)

    print(f"\nModel loaded on {device}")
    print(f"Parameters: {model.num_parameters():,}")

    if args.benchmark:
        benchmark(model, device)
        return

    if args.audio:
        if not os.path.isfile(args.audio):
            print(f"Audio file not found: {args.audio}")
            sys.exit(1)

        audio, sr = load_audio(args.audio)
        print(f"Loaded audio: {audio.shape[0]/sr:.2f}s @ {sr}Hz")

        if args.streaming:
            transcribe_streaming(model, tokenizer, audio, device, args.chunk_ms)
        else:
            transcribe_full(model, tokenizer, audio, device)
    else:
        # Demo with synthetic audio
        print("\nNo audio file provided. Running with synthetic audio (1s)...")
        audio = torch.randn(16000)
        if args.streaming:
            transcribe_streaming(model, tokenizer, audio, device, args.chunk_ms)
        else:
            audio_t = audio.unsqueeze(0).to(device)
            lengths = torch.tensor([16000], device=device)
            with torch.inference_mode():
                encoded, enc_lengths = model.forward(audio_t, lengths)
            print(f"Encoder output shape: {encoded.shape}")
            print(f"Encoder output lengths: {enc_lengths}")
            print("(Tokens would be meaningless on random audio)")


if __name__ == "__main__":
    main()
