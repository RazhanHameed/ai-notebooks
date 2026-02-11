"""
.nemo file converter: extracts weights and config from NVIDIA NeMo checkpoint
and loads them into our from-scratch implementation.

.nemo format: tar.gz archive containing:
  - model_config.yaml (OmegaConf/YAML config)
  - model_weights.ckpt (PyTorch state_dict)
  - *.model (SentencePiece tokenizer model, if bundled)
  - other artifacts (optional)
"""

import os
import re
import tarfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import yaml


class NemoConverter:
    """
    Converts nvidia/nemotron-speech-streaming-en-0.6b .nemo checkpoint
    to our from-scratch streaming ASR model.

    Usage:
        converter = NemoConverter("path/to/model.nemo")
        config = converter.get_config()
        model = build_model(config)
        model = converter.load_weights(model)
        tokenizer_path = converter.get_tokenizer_path()
    """

    def __init__(self, nemo_path: str, extract_dir: Optional[str] = None):
        """
        Args:
            nemo_path: path to .nemo file
            extract_dir: where to extract (uses tempdir if None)
        """
        self.nemo_path = nemo_path

        if extract_dir is None:
            self._tmpdir = tempfile.mkdtemp(prefix="nemo_extract_")
            self.extract_dir = self._tmpdir
        else:
            self._tmpdir = None
            self.extract_dir = extract_dir
            os.makedirs(extract_dir, exist_ok=True)

        self._extracted = False
        self._config = None
        self._state_dict = None

    def extract(self):
        """Extract the .nemo tar.gz archive."""
        if self._extracted:
            return

        print(f"Extracting {self.nemo_path} → {self.extract_dir}")

        # .nemo is a tar archive (may or may not be gzipped)
        try:
            with tarfile.open(self.nemo_path, "r:gz") as tar:
                tar.extractall(self.extract_dir)
        except tarfile.ReadError:
            with tarfile.open(self.nemo_path, "r:") as tar:
                tar.extractall(self.extract_dir)

        self._extracted = True
        self._list_contents()

    def _list_contents(self):
        """List extracted files for debugging."""
        files = []
        for root, dirs, filenames in os.walk(self.extract_dir):
            for f in filenames:
                rel = os.path.relpath(os.path.join(root, f), self.extract_dir)
                files.append(rel)
        print(f"Extracted {len(files)} files:")
        for f in sorted(files):
            print(f"  {f}")
        self._files = files

    def get_config(self) -> Dict[str, Any]:
        """Load and return model config from model_config.yaml."""
        if self._config is not None:
            return self._config

        self.extract()

        # Find config file
        config_path = self._find_file("model_config.yaml")
        if config_path is None:
            # Some .nemo files store it differently
            config_path = self._find_file(".yaml")

        if config_path is None:
            raise FileNotFoundError("No YAML config found in .nemo archive")

        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

        return self._config

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Load PyTorch state dict from the archive."""
        if self._state_dict is not None:
            return self._state_dict

        self.extract()

        # Find weights file
        ckpt_path = self._find_file("model_weights.ckpt")
        if ckpt_path is None:
            ckpt_path = self._find_file(".ckpt")
        if ckpt_path is None:
            ckpt_path = self._find_file(".pt")
        if ckpt_path is None:
            ckpt_path = self._find_file(".bin")

        if ckpt_path is None:
            raise FileNotFoundError("No weights file found in .nemo archive")

        print(f"Loading weights from {ckpt_path}")
        self._state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"Loaded {len(self._state_dict)} weight tensors")
        return self._state_dict

    def get_tokenizer_path(self) -> str:
        """Find and return path to SentencePiece .model file."""
        self.extract()
        path = self._find_file(".model")
        if path is None:
            raise FileNotFoundError("No SentencePiece .model found in .nemo archive")
        return path

    def _find_file(self, pattern: str) -> Optional[str]:
        """Find a file matching pattern in extracted dir."""
        for root, dirs, files in os.walk(self.extract_dir):
            for f in files:
                if f.endswith(pattern) or f == pattern:
                    return os.path.join(root, f)
        return None

    def get_model_config(self) -> Dict[str, Any]:
        """
        Parse NeMo config into our model's constructor arguments.
        Maps NeMo config keys → our parameter names.
        """
        cfg = self.get_config()

        # Encoder config
        enc_cfg = cfg.get("encoder", {})
        # Decoder (prediction) config
        dec_cfg = cfg.get("decoder", {})
        # Joint config
        joint_cfg = cfg.get("joint", {})
        # Preprocessor config
        pre_cfg = cfg.get("preprocessor", {})

        # Parse att_context_size from string or list
        att_ctx = enc_cfg.get("att_context_size", [-1, -1])
        if isinstance(att_ctx, str):
            att_ctx = [int(x.strip()) for x in att_ctx.strip("[]").split(",")]

        model_config = {
            # Preprocessor
            "sample_rate": pre_cfg.get("sample_rate", 16000),
            "n_mels": pre_cfg.get("features", 80),
            "n_fft": pre_cfg.get("n_fft", 512),
            "win_length": pre_cfg.get("window_size", 0.025),
            "hop_length": pre_cfg.get("window_stride", 0.01),
            "dither": pre_cfg.get("dither", 0.0),
            "preemph": pre_cfg.get("preemph", 0.97),
            "normalize": pre_cfg.get("normalize", "per_feature"),

            # Encoder
            "feat_in": pre_cfg.get("features", 80),
            "d_model": enc_cfg.get("d_model", 512),
            "d_ff": enc_cfg.get("d_model", 512) * enc_cfg.get("ff_expansion_factor", 4),
            "n_layers": enc_cfg.get("n_layers", 24),
            "n_heads": enc_cfg.get("n_heads", 8),
            "conv_kernel_size": enc_cfg.get("conv_kernel_size", 31),
            "subsampling_factor": enc_cfg.get("subsampling_factor", 4),
            "subsampling_channels": enc_cfg.get("subsampling_conv_channels", 256),
            "dropout": enc_cfg.get("dropout", 0.1),
            "dropout_att": enc_cfg.get("dropout_att", 0.0),
            "att_context_size": tuple(att_ctx),
            "att_context_style": enc_cfg.get("att_context_style", "regular"),
            "use_bias": enc_cfg.get("use_bias", True),
            "conv_norm_type": enc_cfg.get("conv_norm_type", "batch_norm"),
            "subsampling": enc_cfg.get("subsampling", "dw_striding"),

            # Decoder (Prediction Network)
            "vocab_size": dec_cfg.get("vocab_size", 1024),
            "pred_hidden": dec_cfg.get("pred_hidden", 640),
            "pred_rnn_layers": dec_cfg.get("pred_rnn_layers", 1),

            # Joint
            "joint_hidden": joint_cfg.get("joint_hidden", 640),
            "joint_activation": joint_cfg.get("activation", "relu"),
        }

        return model_config

    def build_weight_map(self) -> Dict[str, str]:
        """
        Build a mapping from NeMo state_dict keys → our model's state_dict keys.

        NeMo convention:
            encoder.pre_encode.conv.{i}.{weight/bias}
            encoder.layers.{i}.self_attn.linear_q.weight
            encoder.layers.{i}.conv_module.depthwise_conv.weight
            encoder.layers.{i}.fc1.fc1.weight (feed-forward)
            decoder.prediction.embed.weight
            decoder.prediction.lstm.weight_ih_l0
            joint.joint_net.{0,2}.weight

        Our convention:
            encoder.subsampling.conv.{i}.{weight/bias}
            encoder.layers.{i}.self_attn.linear_q.weight
            encoder.layers.{i}.conv.depthwise_conv.weight
            encoder.layers.{i}.ffn1.linear1.weight
            decoder.embedding.weight
            decoder.lstm.weight_ih_l0
            joint.enc_proj.weight / joint.pred_proj.weight / joint.output.weight
        """
        nemo_sd = self.get_state_dict()
        weight_map = {}

        for nemo_key in nemo_sd.keys():
            our_key = self._map_single_key(nemo_key)
            if our_key is not None:
                weight_map[nemo_key] = our_key

        return weight_map

    def _map_single_key(self, nemo_key: str) -> Optional[str]:
        """Map a single NeMo state_dict key to our key convention."""
        k = nemo_key

        # Remove common prefixes
        for prefix in ["model.", ""]:
            if k.startswith(prefix) and prefix:
                k = k[len(prefix):]

        # --- Preprocessor (skip - we compute features on the fly) ---
        if k.startswith("preprocessor."):
            return None  # mel filterbank is computed, not learned

        # --- Encoder subsampling ---
        # NeMo: encoder.pre_encode.X → ours: encoder.subsampling.X
        if k.startswith("encoder.pre_encode."):
            return k.replace("encoder.pre_encode.", "encoder.subsampling.")

        # --- Encoder positional encoding ---
        if k.startswith("encoder.pos_emb."):
            return k.replace("encoder.pos_emb.", "encoder.pos_enc.")

        # --- Encoder conformer layers ---
        # NeMo uses various naming for conformer sub-modules
        if k.startswith("encoder.layers."):
            return self._map_encoder_layer_key(k)

        # --- Decoder (prediction network) ---
        if k.startswith("decoder.prediction."):
            rest = k[len("decoder.prediction."):]
            # embed → embedding
            if rest.startswith("embed."):
                return "decoder.embedding." + rest[len("embed."):]
            # lstm layers
            if rest.startswith("lstm.") or rest.startswith("rnn."):
                prefix_len = rest.index(".") + 1
                return "decoder.lstm." + rest[prefix_len:]
            return "decoder." + rest

        # --- Joint network ---
        if k.startswith("joint."):
            return self._map_joint_key(k)

        # Fallback: return as-is
        return k

    def _map_encoder_layer_key(self, k: str) -> str:
        """Map encoder layer keys between NeMo and our naming."""
        # Extract layer index
        m = re.match(r"encoder\.layers\.(\d+)\.(.*)", k)
        if not m:
            return k
        layer_idx = m.group(1)
        rest = m.group(2)
        prefix = f"encoder.layers.{layer_idx}."

        # Self-attention
        if rest.startswith("self_attn.") or rest.startswith("mha."):
            sub = rest.split(".", 1)[1]
            return prefix + "self_attn." + sub

        # Feed-forward 1 (first Macaron FFN)
        if rest.startswith("fc1.") or rest.startswith("feed_forward1."):
            sub = rest.split(".", 1)[1]
            return prefix + "ffn1." + self._map_ffn_key(sub)

        # Feed-forward 2 (second Macaron FFN)
        if rest.startswith("fc2.") or rest.startswith("feed_forward2."):
            sub = rest.split(".", 1)[1]
            return prefix + "ffn2." + self._map_ffn_key(sub)

        # Convolution module
        if rest.startswith("conv_module.") or rest.startswith("conv."):
            sub = rest.split(".", 1)[1]
            return prefix + "conv." + sub

        # Layer norms
        if rest.startswith("norm_"):
            return prefix + rest

        return prefix + rest

    def _map_ffn_key(self, sub: str) -> str:
        """Map FFN sub-module keys."""
        # NeMo: fc1.weight → linear1.weight, fc2.weight → linear2.weight
        if sub.startswith("fc1.") or sub.startswith("linear1."):
            return "linear1." + sub.split(".", 1)[1]
        if sub.startswith("fc2.") or sub.startswith("linear2."):
            return "linear2." + sub.split(".", 1)[1]
        if sub.startswith("layer_norm.") or sub.startswith("norm."):
            return "layer_norm." + sub.split(".", 1)[1]
        return sub

    def _map_joint_key(self, k: str) -> str:
        """Map joint network keys."""
        # NeMo: joint.joint_net.0.weight → enc_proj.weight
        # NeMo: joint.joint_net.1.weight → pred_proj.weight
        # NeMo: joint.joint_net.3.weight → output.weight (after ReLU at index 2)
        if "joint_net.0." in k:
            return k.replace("joint.joint_net.0.", "joint.enc_proj.")
        if "joint_net.1." in k:
            return k.replace("joint.joint_net.1.", "joint.pred_proj.")
        if "joint_net.3." in k:
            return k.replace("joint.joint_net.3.", "joint.output.")

        # NeMo: joint.encoder_proj → joint.enc_proj
        if "encoder_proj" in k:
            return k.replace("encoder_proj", "enc_proj")
        if "decoder_proj" in k or "prediction_proj" in k:
            return k.replace("decoder_proj", "pred_proj").replace("prediction_proj", "pred_proj")

        return k

    def load_weights_into_model(self, model) -> Tuple[list, list]:
        """
        Load converted weights into our model.

        Args:
            model: our StreamingASRModel instance

        Returns:
            (loaded_keys, missing_keys) for diagnostics
        """
        nemo_sd = self.get_state_dict()
        weight_map = self.build_weight_map()

        our_sd = model.state_dict()
        loaded = []
        skipped = []
        shape_mismatched = []

        for nemo_key, our_key in weight_map.items():
            if our_key in our_sd:
                nemo_tensor = nemo_sd[nemo_key]
                our_tensor = our_sd[our_key]

                if nemo_tensor.shape == our_tensor.shape:
                    our_sd[our_key] = nemo_tensor
                    loaded.append((nemo_key, our_key))
                else:
                    shape_mismatched.append(
                        (nemo_key, our_key, nemo_tensor.shape, our_tensor.shape)
                    )
            else:
                skipped.append((nemo_key, our_key))

        # Load with strict=False to handle missing/extra keys
        model.load_state_dict(our_sd, strict=False)

        missing = [k for k in our_sd if k not in {v for _, v in weight_map.items()}]

        print(f"Loaded {len(loaded)} / {len(weight_map)} weight tensors")
        if shape_mismatched:
            print(f"Shape mismatches ({len(shape_mismatched)}):")
            for nk, ok, ns, os_ in shape_mismatched[:10]:
                print(f"  {nk} {ns} → {ok} {os_}")
        if missing:
            print(f"Missing in NeMo ({len(missing)}):")
            for k in missing[:10]:
                print(f"  {k}")

        return [k for _, k in loaded], missing

    def cleanup(self):
        """Remove extracted temp files."""
        if self._tmpdir and os.path.isdir(self._tmpdir):
            shutil.rmtree(self._tmpdir)

    def __del__(self):
        self.cleanup()
