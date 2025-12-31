import struct
import numpy as np
import os

# CONSTANTS
MAGIC_NUMBER = 0x4E414E4F
HEADER_SIZE = 256
format_string = "<7I"

class ModelHeader:
    def __init__(self, vocab_size, dim, layers, n_heads, n_kv_heads, hidden_dim):
        self.vocab_size = vocab_size
        self.dim = dim
        self.layers = layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.hidden_dim = hidden_dim

    def pack(self):
        data = struct.pack(
            format_string,
            MAGIC_NUMBER,
            self.vocab_size,
            self.dim,
            self.layers,
            self.n_heads,
            self.n_kv_heads,
            self.hidden_dim
        )
        padding = b'\x00' * (HEADER_SIZE - len(data))
        return data + padding

def write_model(output_path, header, weights, scales):
    print(f"Writing to {output_path}...")

    with open(output_path, "wb") as f:
        # 1. Write Header
        f.write(header.pack())

        # 2. Write Weights & Scales
        total_bytes = 0
        for i, w in enumerate(weights):
            # A. Write Weight Block
            w_int8 = w.astype(np.int8)
            f.write(w_int8.tobytes())
            total_bytes += w_int8.nbytes

            # B. Write Scale Block (The Fix)
            current_scale = scales[i]

            if isinstance(current_scale, np.ndarray):
                # CASE 1: Per-Channel (Array of floats)
                # Ensure it's float32 before writing
                s_bytes = current_scale.astype(np.float32).tobytes()
                f.write(s_bytes)
                total_bytes += len(s_bytes)
            else:
                # CASE 2: Per-Tensor (Single float scalar)
                s_float = struct.pack("<f", float(current_scale))
                f.write(s_float)
                total_bytes += 4

    print(f"Success! Wrote {total_bytes / 1024 / 1024:.2f} MB of tensor data.")
