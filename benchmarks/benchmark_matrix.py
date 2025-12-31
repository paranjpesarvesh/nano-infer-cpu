import time
import numpy as np
import os
import sys

# Add parent directory to path so we can import engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.kernels import matmul_int8

def benchmark_compute():
    DIM = 4096
    HIDDEN = 4096 # Square matrix for simplicity
    ITERATIONS = 50

    print(f"Benchmarking Compute: Matrix Multiplication ({DIM}x{HIDDEN})")
    print(f"    Running {ITERATIONS} iterations...")
    print("-" * 60)

    # 1. Setup Data
    # Input vector
    x = np.random.randn(DIM).astype(np.float32)

    # Weights (Simulating the Transposed structure used in our Engine)
    # Shape: [Dim, Hidden]
    w_int8 = np.random.randint(-127, 127, (DIM, HIDDEN)).astype(np.int8)
    scales = np.abs(np.random.randn(DIM)).astype(np.float32) # Per-channel (row) scales

    # Pre-allocate output
    out_custom = np.zeros(HIDDEN, dtype=np.float32)

    # 2. Baseline: NumPy Float32
    # We must dequantize to Float32 to simulate standard PyTorch/NumPy behavior
    # This represents the "Memory Cost" of standard inference
    print("1. Baseline (NumPy Float32):")
    start_dequant = time.time()
    w_float = w_int8.astype(np.float32) * scales[:, None]
    mem_overhead = w_float.nbytes / (1024**2)
    print(f"   [Memory] Expanded Matrix Size: {mem_overhead:.2f} MB")

    # Warmup
    _ = x @ w_float

    start_time = time.time()
    for _ in range(ITERATIONS):
        # Standard Vector-Matrix multiplication
        res = x @ w_float
    avg_numpy = (time.time() - start_time) / ITERATIONS
    print(f"   [Time]   Average Latency: {avg_numpy*1000:.2f} ms")

    # 3. Ours: NanoInfer Int8 Kernel
    print("\n2. NanoInfer (Int8 Kernel):")
    mem_ours = w_int8.nbytes / (1024**2)
    print(f"   [Memory] Compressed Matrix Size: {mem_ours:.2f} MB (4x Smaller)")

    # Warmup (Compiles Numba JIT)
    matmul_int8(x, w_int8, scales, out_custom)

    start_time = time.time()
    for _ in range(ITERATIONS):
        matmul_int8(x, w_int8, scales, out_custom)
    avg_ours = (time.time() - start_time) / ITERATIONS
    print(f"   [Time]   Average Latency: {avg_ours*1000:.2f} ms")

    print("-" * 60)
    print(f"Conclusion:")
    print(f"   Memory Savings: 4.0x")
    if avg_ours < avg_numpy:
        print(f"   Speedup: {avg_numpy / avg_ours:.2f}x Faster")
    else:
        print(f"   Performance: {avg_ours / avg_numpy:.2f}x slower (Expected: NumPy uses AVX-512 BLAS)")
    print("   (Note: Our win is Memory Bandwidth, not raw FLOPs)")

if __name__ == "__main__":
    benchmark_compute()
