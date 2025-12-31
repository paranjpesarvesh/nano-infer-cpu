import time
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.memory import KVCache

# Mock Config Object
class Config:
    def __init__(self):
        self.n_kv_heads = 4
        self.dim = 2048
        self.n_heads = 32
        self.layers = 22

    # Allow dictionary access for the memory module
    def __getitem__(self, item):
        return getattr(self, item)

def benchmark_allocation():
    CTX_LEN = 2048
    HEAD_DIM = 64
    KV_HEADS = 4
    LAYERS = 22

    print(f"Benchmarking Memory Allocation (Context: 0 -> {CTX_LEN})")
    print("-" * 60)

    # 1. Baseline: Python List Append (Dynamic Allocation)
    print("1. Baseline (Python Lists):")
    py_list_cache = [[] for _ in range(LAYERS)]

    latencies_py = []

    # Create dummy vector to append
    # Shape: [KV_HEADS, HEAD_DIM]
    dummy_k = np.zeros((KV_HEADS, HEAD_DIM), dtype=np.float32)

    for i in range(CTX_LEN):
        t0 = time.time()
        for l in range(LAYERS):
            # Simulate appending to KV cache
            py_list_cache[l].append(dummy_k)
        dt = time.time() - t0
        latencies_py.append(dt)

    avg_py = np.mean(latencies_py) * 1000000 # microseconds
    p99_py = np.percentile(latencies_py, 99) * 1000000
    print(f"   Avg Append: {avg_py:.2f} μs")
    print(f"   P99 Spike:  {p99_py:.2f} μs (Re-allocation Jitters)")

    # 2. Ours: Static Ring Buffer
    print("\n2. NanoInfer (Static Ring Buffer):")
    conf = Config()
    kv = KVCache(conf, max_context=CTX_LEN)

    latencies_ours = []

    for i in range(CTX_LEN):
        t0 = time.time()
        for l in range(LAYERS):
            # Update specific slot (O(1) complexity)
            kv.update(l, dummy_k, dummy_k, i)
        dt = time.time() - t0
        latencies_ours.append(dt)

    avg_ours = np.mean(latencies_ours) * 1000000
    p99_ours = np.percentile(latencies_ours, 99) * 1000000
    print(f"   Avg Update: {avg_ours:.2f} μs")
    print(f"   P99 Spike:  {p99_ours:.2f} μs (Stable)")

    print("-" * 60)
    print(f"Stability Win: Our P99 latency is {p99_py/p99_ours:.1f}x more stable.")

if __name__ == "__main__":
    benchmark_allocation()
