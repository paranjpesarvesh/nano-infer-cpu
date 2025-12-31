import numpy as np

def quantize_tensor(tensor):
    """
    Performs symmetric Int8 quantization on a given tensor.

    Strategy: Per-Channel (Row-wise) Quantization.
    We calculate a separate scale factor for every row in the matrix.
    This preserves accuracy much better than per-tensor quantization.

    Args:
        tensor (np.ndarray): The input float tensor (usually float16 or float32).

    Returns:
        quantized (np.int8): The compressed tensor.
        scales (np.float32): The scaling factors (one per row).
    """

    # 1. Handle 1D Tensors (like Layernorm weights)
    # These are small, so we can just use a single scale (Per-Tensor).
    if len(tensor.shape) == 1:
        max_val = np.max(np.abs(tensor))
        scale = max_val / 127.0 if max_val > 0 else 1.0

        # Quantize
        q = np.round(tensor / scale).astype(np.int8)
        return q, scale

    # 2. Handle 2D Matrices (Linear Layers)
    # Shape: [Rows, Cols]
    # We want a scale factor for EACH ROW.
    rows, cols = tensor.shape

    # Calculate max(abs) for each row independently
    # axis=1 means "across columns" -> results in [Rows]
    max_val = np.max(np.abs(tensor), axis=1)

    # Avoid division by zero
    max_val = np.where(max_val == 0, 1.0, max_val)

    # Calculate scales: [Rows]
    scales = max_val / 127.0

    # 3. Broadcast Division
    # We need to reshape scales to [Rows, 1] so numpy divides correctly
    scales_reshaped = scales.reshape(-1, 1)

    # Perform quantization
    # x_int = round(x_float / scale)
    raw_q = np.round(tensor / scales_reshaped)

    # Clip just in case floating point weirdness pushes it to 128
    raw_q = np.clip(raw_q, -127, 127)

    return raw_q.astype(np.int8), scales.astype(np.float32)
