#!/usr/bin/env python3
"""
Convert nvidia/nemotron-speech-streaming-en-0.6b .nemo checkpoint
to our from-scratch streaming ASR implementation.

Usage:
    # Download the .nemo file first:
    # huggingface-cli download nvidia/nemotron-speech-streaming-en-0.6b \
    #     nemotron-speech-streaming-en-0.6b.nemo --local-dir ./checkpoints

    python convert_nemo.py \
        --nemo_path ./checkpoints/nemotron-speech-streaming-en-0.6b.nemo \
        --output_dir ./converted_model \
        [--inspect_only]  # just print config and weight names without converting
"""

import argparse
import json
import os
import sys

import torch

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from streaming_asr.utils.nemo_converter import NemoConverter
from streaming_asr.model import StreamingASRModel


def inspect_nemo(nemo_path: str):
    """Print config and weight tensor names/shapes from a .nemo file."""
    converter = NemoConverter(nemo_path)

    print("=" * 70)
    print("MODEL CONFIGURATION")
    print("=" * 70)
    config = converter.get_config()
    _print_config(config)

    print("\n" + "=" * 70)
    print("PARSED MODEL CONFIG (for our implementation)")
    print("=" * 70)
    model_config = converter.get_model_config()
    for k, v in sorted(model_config.items()):
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("WEIGHT TENSORS")
    print("=" * 70)
    sd = converter.get_state_dict()
    total_params = 0
    for name, tensor in sorted(sd.items()):
        n = tensor.numel()
        total_params += n
        print(f"  {name:80s} {str(list(tensor.shape)):20s} {n:>12,}")
    print(f"\n  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    print("\n" + "=" * 70)
    print("WEIGHT KEY MAPPING (NeMo → ours)")
    print("=" * 70)
    wmap = converter.build_weight_map()
    for nemo_key, our_key in sorted(wmap.items()):
        print(f"  {nemo_key:80s} → {our_key}")

    # Find tokenizer
    try:
        tok_path = converter.get_tokenizer_path()
        print(f"\nTokenizer found: {tok_path}")
    except FileNotFoundError:
        print("\nNo tokenizer found in archive")

    converter.cleanup()


def _print_config(cfg, indent=2):
    """Recursively print YAML config."""
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if isinstance(v, (dict, list)):
                print(" " * indent + f"{k}:")
                _print_config(v, indent + 2)
            else:
                print(" " * indent + f"{k}: {v}")
    elif isinstance(cfg, list):
        for item in cfg:
            if isinstance(item, (dict, list)):
                _print_config(item, indent + 2)
            else:
                print(" " * indent + f"- {item}")


def convert_nemo(nemo_path: str, output_dir: str):
    """
    Full conversion pipeline:
    1. Extract .nemo archive
    2. Parse config → our model config
    3. Build our model
    4. Map and load weights
    5. Save converted checkpoint + config + tokenizer
    """
    os.makedirs(output_dir, exist_ok=True)

    converter = NemoConverter(nemo_path, extract_dir=os.path.join(output_dir, "_nemo_extract"))

    # Step 1: Get config
    print("Step 1: Parsing model configuration...")
    model_config = converter.get_model_config()

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2, default=str)
    print(f"  Saved config to {config_path}")

    # Step 2: Build model
    print("\nStep 2: Building model from config...")
    model = StreamingASRModel(model_config)
    n_params = model.num_parameters()
    print(f"  Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Step 3: Load weights
    print("\nStep 3: Loading and mapping weights...")
    loaded, missing = converter.load_weights_into_model(model)

    # Step 4: Save converted model
    print("\nStep 4: Saving converted model...")
    model_path = os.path.join(output_dir, "model.pt")
    torch.save({
        "config": model_config,
        "state_dict": model.state_dict(),
    }, model_path)
    print(f"  Saved model to {model_path}")

    # Step 5: Copy tokenizer
    print("\nStep 5: Extracting tokenizer...")
    try:
        import shutil
        tok_src = converter.get_tokenizer_path()
        tok_dst = os.path.join(output_dir, "tokenizer.model")
        shutil.copy2(tok_src, tok_dst)
        print(f"  Saved tokenizer to {tok_dst}")
    except FileNotFoundError:
        print("  No tokenizer found in .nemo archive")
        print("  You may need to download it separately")

    # Step 6: Verify
    print("\nStep 6: Verification...")
    # Reload and check
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model2 = StreamingASRModel(checkpoint["config"])
    model2.load_state_dict(checkpoint["state_dict"])
    print(f"  Reload successful. Parameters: {model2.num_parameters():,}")

    # Test forward pass with dummy data
    print("\n  Testing forward pass with dummy audio...")
    model2.eval()
    with torch.inference_mode():
        dummy_audio = torch.randn(1, 16000)  # 1 second
        dummy_lengths = torch.tensor([16000])
        try:
            encoded, enc_lengths = model2.forward(dummy_audio, dummy_lengths)
            print(f"  Forward pass OK. Encoder output shape: {encoded.shape}")
        except Exception as e:
            print(f"  Forward pass error (expected if shapes don't match yet): {e}")

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Files:")
    for f in os.listdir(output_dir):
        if not f.startswith("_"):
            fpath = os.path.join(output_dir, f)
            if os.path.isfile(fpath):
                size_mb = os.path.getsize(fpath) / 1e6
                print(f"  {f}: {size_mb:.1f} MB")

    converter.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Convert .nemo checkpoint to streaming ASR format"
    )
    parser.add_argument(
        "--nemo_path", required=True,
        help="Path to .nemo checkpoint file",
    )
    parser.add_argument(
        "--output_dir", default="./converted_model",
        help="Output directory for converted model",
    )
    parser.add_argument(
        "--inspect_only", action="store_true",
        help="Only inspect the .nemo file without converting",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.nemo_path):
        print(f"Error: .nemo file not found: {args.nemo_path}")
        print("\nTo download the model:")
        print("  pip install huggingface_hub")
        print("  huggingface-cli download nvidia/nemotron-speech-streaming-en-0.6b \\")
        print("      nemotron-speech-streaming-en-0.6b.nemo --local-dir ./checkpoints")
        sys.exit(1)

    if args.inspect_only:
        inspect_nemo(args.nemo_path)
    else:
        convert_nemo(args.nemo_path, args.output_dir)


if __name__ == "__main__":
    main()
