"""
CUDA optimization utilities for streaming ASR inference.
Provides torch.compile wrappers, CUDA graph capture, and memory optimization.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Callable, Tuple
import functools


# ---------------------------------------------------------------------------
# torch.compile optimization wrappers
# ---------------------------------------------------------------------------

def optimize_for_inference(model: nn.Module, backend: str = "inductor") -> nn.Module:
    """
    Apply torch.compile with optimal settings for ASR inference.

    Args:
        model: the model to optimize
        backend: compile backend ("inductor", "cudagraphs", etc.)

    Returns:
        optimized model
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping torch.compile optimization")
        return model

    try:
        optimized = torch.compile(
            model,
            backend=backend,
            mode="reduce-overhead",  # best for inference
            fullgraph=False,         # allow graph breaks for dynamic shapes
        )
        print(f"Model compiled with backend={backend}, mode=reduce-overhead")
        return optimized
    except Exception as e:
        print(f"torch.compile failed: {e}, falling back to eager mode")
        return model


# ---------------------------------------------------------------------------
# CUDA Graph capture for static-shape operations
# ---------------------------------------------------------------------------

class CUDAGraphRunner:
    """
    Captures and replays CUDA graphs for the decoder loop.

    The RNN-T greedy decode inner loop has mostly fixed shapes:
    - decoder step: (1, 1, D_pred) → (1, 1, D_pred)
    - joint step: (1, 1, D_enc) + (1, 1, D_pred) → (1, 1, V+1)

    Capturing this as a CUDA graph eliminates kernel launch overhead.
    """

    def __init__(self):
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_inputs: Dict[str, Tensor] = {}
        self._static_outputs: Dict[str, Tensor] = {}

    def capture(
        self,
        fn: Callable,
        sample_inputs: Dict[str, Tensor],
        warmup_iters: int = 3,
    ) -> "CUDAGraphRunner":
        """
        Capture a CUDA graph from a function.

        Args:
            fn: function to capture
            sample_inputs: dict of {name: tensor} with representative shapes
            warmup_iters: warmup iterations before capture
        """
        if not torch.cuda.is_available():
            return self

        # Allocate static input buffers
        self._static_inputs = {
            k: v.clone() for k, v in sample_inputs.items()
        }

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup_iters):
                fn(**self._static_inputs)
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_outputs = fn(**self._static_inputs)

        return self

    def replay(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Replay the captured graph with new inputs.

        Args:
            inputs: dict matching the keys of sample_inputs

        Returns:
            outputs (references to static buffers, copy if needed)
        """
        if self._graph is None:
            raise RuntimeError("No graph captured. Call capture() first.")

        # Copy new inputs into static buffers
        for k, v in inputs.items():
            self._static_inputs[k].copy_(v)

        # Replay
        self._graph.replay()

        # Return copies to avoid aliasing issues
        return {k: v.clone() for k, v in self._static_outputs.items()}


# ---------------------------------------------------------------------------
# Memory-efficient attention (for when flash attention isn't available)
# ---------------------------------------------------------------------------

@torch.jit.script
def memory_efficient_rel_attention(
    q: Tensor,      # (B, H, T, D)
    k: Tensor,      # (B, H, S, D)
    v: Tensor,      # (B, H, S, D)
    bias_u: Tensor, # (H, D)
    bias_v: Tensor, # (H, D)
    pos: Tensor,    # (1, H, L, D)
    scale: float,
    chunk_size: int = 256,
) -> Tensor:
    """
    Memory-efficient chunked relative-position attention.
    Processes query in chunks to reduce peak memory.
    TorchScript-compiled for best performance.
    """
    B, H, T, D = q.shape
    S = k.shape[2]

    out = torch.zeros_like(q)

    for t_start in range(0, T, chunk_size):
        t_end = min(t_start + chunk_size, T)
        q_chunk = q[:, :, t_start:t_end, :]  # (B, H, chunk, D)

        # Content attention
        q_u = q_chunk + bias_u.unsqueeze(0).unsqueeze(2)
        content = torch.matmul(q_u, k.transpose(-2, -1))  # (B, H, chunk, S)

        # Position attention
        q_v = q_chunk + bias_v.unsqueeze(0).unsqueeze(2)
        pos_score = torch.matmul(q_v, pos.transpose(-2, -1))
        # Simple truncation for position scores
        if pos_score.shape[-1] > S:
            pos_score = pos_score[:, :, :, :S]

        attn = (content + pos_score) * scale
        attn = torch.softmax(attn, dim=-1)
        out[:, :, t_start:t_end, :] = torch.matmul(attn, v)

    return out


# ---------------------------------------------------------------------------
# Mixed precision utilities
# ---------------------------------------------------------------------------

class AutocastInference:
    """
    Context manager for optimal mixed-precision inference.
    Uses float16 for encoder (compute-bound) and float32 for decoder (memory-bound).
    """

    def __init__(self, enabled: bool = True, dtype: torch.dtype = torch.float16):
        self.enabled = enabled and torch.cuda.is_available()
        self.dtype = dtype

    def __enter__(self):
        if self.enabled:
            self._ctx = torch.cuda.amp.autocast(dtype=self.dtype)
            self._ctx.__enter__()
        return self

    def __exit__(self, *args):
        if self.enabled:
            self._ctx.__exit__(*args)


def apply_half_precision(model: nn.Module, exclude_decoder: bool = True) -> nn.Module:
    """
    Convert model to half precision, optionally keeping decoder in fp32.

    The decoder LSTM is memory-bandwidth bound (not compute bound),
    so fp16 gives minimal speedup but can hurt accuracy.
    The encoder is compute-bound, so fp16 gives ~2x speedup on Tensor Cores.
    """
    if not torch.cuda.is_available():
        return model

    model = model.half()

    if exclude_decoder:
        # Keep decoder + joint in fp32
        if hasattr(model, 'decoder'):
            model.decoder = model.decoder.float()
        if hasattr(model, 'joint'):
            model.joint = model.joint.float()

    return model


# ---------------------------------------------------------------------------
# Profiling utilities
# ---------------------------------------------------------------------------

class InferenceProfiler:
    """Lightweight profiler for streaming inference latency."""

    def __init__(self):
        self.timings: Dict[str, list] = {}
        self._start_events: Dict[str, torch.cuda.Event] = {}

    def start(self, name: str):
        if torch.cuda.is_available():
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._start_events[name] = event
        else:
            import time
            self._start_events[name] = time.monotonic()

    def stop(self, name: str):
        if name not in self._start_events:
            return

        if torch.cuda.is_available():
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            start = self._start_events.pop(name)
            elapsed_ms = start.elapsed_time(end)
        else:
            import time
            elapsed_ms = (time.monotonic() - self._start_events.pop(name)) * 1000

        self.timings.setdefault(name, []).append(elapsed_ms)

    def summary(self) -> str:
        lines = ["Inference Timing Summary:"]
        for name, times in self.timings.items():
            avg = sum(times) / len(times)
            mn = min(times)
            mx = max(times)
            lines.append(f"  {name}: avg={avg:.2f}ms min={mn:.2f}ms max={mx:.2f}ms n={len(times)}")
        return "\n".join(lines)
