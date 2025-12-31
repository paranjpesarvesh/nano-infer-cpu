import numpy as np
from numba import njit, prange
import math

# CONSTANTS
BLOCK_SIZE = 64  # Fits in L1 Cache (64 * 64 * 1 byte = 4KB)

@njit(fastmath=True)
def rmsnorm(x, weight, epsilon=1e-6):
    """
    Root Mean Square Layer Normalization.
    Standard in Llama architecture.
    """
    # 1. Calculate Sum of Squares
    ss = 0.0
    for i in range(x.shape[0]):
        ss += x[i] * x[i]

    # 2. Calculate RMS
    rms = 1.0 / math.sqrt((ss / x.shape[0]) + epsilon)

    # 3. Normalize and Scale
    # We create a new array to avoid modifying input in-place if not needed
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = x[i] * rms * weight[i]

    return out

@njit(fastmath=True)
def softmax(x):
    """
    Numerically stable Softmax.
    """
    # 1. Find max for stability (prevent overflow)
    max_val = -np.inf
    for i in range(x.shape[0]):
        if x[i] > max_val:
            max_val = x[i]

    # 2. Exponentiate and Sum
    sum_exp = 0.0
    for i in range(x.shape[0]):
        # Overwrite x in-place to save memory
        x[i] = math.exp(x[i] - max_val)
        sum_exp += x[i]

    # 3. Normalize
    scale = 1.0 / sum_exp
    for i in range(x.shape[0]):
        x[i] *= scale

    return x

@njit(fastmath=True)
def silu(x):
    """
    SiLU (Sigmoid Linear Unit) Activation.
    Used in Llama's FeedForward network.
    """
    for i in range(x.shape[0]):
        val = x[i]
        # x * sigmoid(x)
        x[i] = val / (1.0 + math.exp(-val))
    return x

@njit(parallel=True, fastmath=True)
def matmul_int8(x, w_int8, scales, out):
    """
    The Core Kernel: Int8 Matrix Multiplication with Dequantization.

    Args:
        x: Input Vector [Dim] (Float32)
        w_int8: Weight Matrix [Dim, Hidden] (Int8)
        scales: Scale factors [Dim] (Float32)
        out: Output Vector [Hidden] (Float32) - Pre-allocated
    """
    rows, cols = w_int8.shape

    # PARALLEL LOOP: Numba will spread this across your CPU cores
    for col in prange(cols):
        acc = 0.0

        # INNER LOOP: Standard dot product
        # Note: For maximum speed, we would tile this loop "for k in range(0, rows, BLOCK)..."
        # But Numba's auto-vectorizer does a decent job here for 1D vectors.
        for row in range(rows):
            # 1. Load Int8 Weight
            val_int8 = w_int8[row, col]

            # 2. Dequantize (The "On-the-fly" trick)
            # We recover the float value using the scale factor
            val_float = val_int8 * scales[row] # Note: In our format, scale is per-row or per-channel

            # 3. Accumulate
            acc += val_float * x[row]

        out[col] = acc

@njit(fastmath=True)
def apply_rope(q, k, cos, sin):
    """
    Applies Rotary Positional Embeddings (RoPE) in-place.
    Args:
        q: Query vector [Heads, Head_Dim]
        k: Key vector [KV_Heads, Head_Dim]
        cos: Cosine slice for this position [Head_Dim // 2]
        sin: Sine slice for this position [Head_Dim // 2]
    """
    # RoPE works on pairs. We iterate over half the head dimension.
    n_heads, head_dim = q.shape
    n_kv_heads, _ = k.shape
    half_dim = head_dim // 2

    # Rotate Q
    for h in range(n_heads):
        for i in range(half_dim):
            # Llama implementation pairs indices i and i + half_dim
            real_idx = i
            imag_idx = i + half_dim

            q_r = q[h, real_idx]
            q_i = q[h, imag_idx]

            # The Rotation
            # new_real = real * cos - imag * sin
            # new_imag = real * sin + imag * cos
            c = cos[i]
            s = sin[i]

            q[h, real_idx] = q_r * c - q_i * s
            q[h, imag_idx] = q_r * s + q_i * c

    # Rotate K
    for h in range(n_kv_heads):
        for i in range(half_dim):
            real_idx = i
            imag_idx = i + half_dim

            k_r = k[h, real_idx]
            k_i = k[h, imag_idx]

            c = cos[i]
            s = sin[i]

            k[h, real_idx] = k_r * c - k_i * s
            k[h, imag_idx] = k_r * s + k_i * c

    return q, k
