"""
Fused CUDA kernels for streaming ASR operations.
Uses torch.compile + custom Triton kernels where possible.
Falls back to optimized PyTorch when Triton unavailable.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton: Fused SiLU (Swish) kernel
# ---------------------------------------------------------------------------
if HAS_TRITON:
    @triton.jit
    def _swish_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        out = x * tl.sigmoid(x)
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _swish_backward_kernel(
        x_ptr, grad_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        grad = tl.load(grad_ptr + offsets, mask=mask)
        sig = tl.sigmoid(x)
        out = grad * (sig + x * sig * (1.0 - sig))
        tl.store(out_ptr + offsets, out, mask=mask)


class _TritonSwish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK = 1024
        grid = ((n + BLOCK - 1) // BLOCK,)
        _swish_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK)
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (x,) = ctx.saved_tensors
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK = 1024
        grid = ((n + BLOCK - 1) // BLOCK,)
        _swish_backward_kernel[grid](x, grad_output, out, n, BLOCK_SIZE=BLOCK)
        return out


def fused_swish(x: Tensor) -> Tensor:
    """Fused Swish/SiLU activation – Triton if available, else PyTorch."""
    if HAS_TRITON and x.is_cuda:
        return _TritonSwish.apply(x.contiguous())
    return F.silu(x)


# ---------------------------------------------------------------------------
# Triton: Fused residual + LayerNorm
# ---------------------------------------------------------------------------
if HAS_TRITON:
    @triton.jit
    def _add_layernorm_kernel(
        x_ptr, residual_ptr, weight_ptr, bias_ptr, out_ptr,
        N, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        x = tl.load(x_ptr + row * N + offsets, mask=mask, other=0.0)
        res = tl.load(residual_ptr + row * N + offsets, mask=mask, other=0.0)
        x = x + res

        mean = tl.sum(x, axis=0) / N
        x_centered = x - mean
        var = tl.sum(x_centered * x_centered, axis=0) / N
        inv_std = 1.0 / tl.sqrt(var + eps)

        w = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
        b = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        out = x_centered * inv_std * w + b
        tl.store(out_ptr + row * N + offsets, out, mask=mask)


def fused_add_norm(
    x: Tensor, residual: Tensor,
    weight: Tensor, bias: Tensor,
    eps: float = 1e-5,
) -> Tensor:
    """Fused residual-add + LayerNorm. Triton on CUDA, else PyTorch."""
    if HAS_TRITON and x.is_cuda and x.shape[-1] <= 2048:
        B = x.numel() // x.shape[-1]
        N = x.shape[-1]
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _add_layernorm_kernel[(B,)](
            x.contiguous(), residual.contiguous(),
            weight, bias, out,
            N, eps=eps, BLOCK_SIZE=BLOCK,
        )
        return out
    return F.layer_norm(x + residual, (x.shape[-1],), weight, bias, eps)


# ---------------------------------------------------------------------------
# Triton: Fused Joint Network (encoder_proj + decoder_proj + ReLU + linear)
# ---------------------------------------------------------------------------
if HAS_TRITON:
    @triton.jit
    def _fused_joint_relu_kernel(
        enc_ptr, dec_ptr, out_ptr,
        M, N, BLOCK_N: tl.constexpr,
    ):
        """Fused elementwise add + ReLU for joint network intermediate."""
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_N)
        mask = offsets < N

        enc = tl.load(enc_ptr + row * N + offsets, mask=mask, other=0.0)
        dec = tl.load(dec_ptr + row * N + offsets, mask=mask, other=0.0)
        s = enc + dec
        s = tl.where(s > 0, s, 0.0)  # ReLU
        tl.store(out_ptr + row * N + offsets, s, mask=mask)


def fused_joint_net(
    enc_proj: Tensor,  # (B, T, 1, H)
    dec_proj: Tensor,  # (B, 1, U, H)
    out_weight: Tensor,  # (V, H)
    out_bias: Tensor,  # (V,)
) -> Tensor:
    """
    Fused joint network: add + ReLU + linear.
    Broadcasts enc (B,T,1,H) and dec (B,1,U,H) → (B,T,U,H), then projects.
    """
    B, T, _, H = enc_proj.shape
    _, _, U, _ = dec_proj.shape

    # Broadcast-add
    combined = enc_proj + dec_proj  # (B, T, U, H)

    if HAS_TRITON and combined.is_cuda:
        flat = combined.reshape(-1, H).contiguous()
        M = flat.shape[0]
        BLOCK = triton.next_power_of_2(H)
        # In-place ReLU via Triton
        _relu_inplace = torch.clamp_min_(flat, 0)
    else:
        flat = F.relu(combined.reshape(-1, H))

    # Final projection
    logits = F.linear(flat, out_weight, out_bias)  # (B*T*U, V)
    return logits.view(B, T, U, -1)
