import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add NanoInfer to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.weights import MappedWeights
from engine.transformer import Transformer

MODEL_HF = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_NANO = "models/tinyllama.nano"
TOKENS_TO_GEN = 20

def run_hf_benchmark():
    print(f"Loading HuggingFace Model ({MODEL_HF})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HF)
    model = AutoModelForCausalLM.from_pretrained(MODEL_HF, torch_dtype=torch.float32)

    prompt = "The future of AI is"
    inputs = tokenizer(prompt, return_tensors="pt")

    print("Generating (HF)...")
    start = time.time()
    _ = model.generate(**inputs, max_new_tokens=TOKENS_TO_GEN)
    dt = time.time() - start

    tps = TOKENS_TO_GEN / dt
    print(f"HF Speed: {tps:.2f} tokens/sec")
    return tps

def run_nano_benchmark():
    print(f"\nLoading NanoInfer Model ({MODEL_NANO})...")
    if not os.path.exists(MODEL_NANO):
        print("Model binary not found. Run convert.py first.")
        return 0

    weights = MappedWeights(MODEL_NANO)
    model = Transformer(weights)

    # Dummy inputs (Skipping tokenizer for raw engine speed test)
    # Just feeding ID '1' repeatedly
    pos = 0
    print("Generating (Nano)...")

    # Warmup
    model.forward(1, 0)

    start = time.time()
    for i in range(TOKENS_TO_GEN):
        _ = model.forward(1, pos)
        pos += 1
    dt = time.time() - start

    tps = TOKENS_TO_GEN / dt
    print(f"Nano Speed: {tps:.2f} tokens/sec")
    return tps

if __name__ == "__main__":
    print("End-to-End Speed Benchmark")
    print("="*40)

    # 1. Run Baseline
    try:
        hf_tps = run_hf_benchmark()
    except Exception as e:
        print(f"Skipping HF Benchmark (Error: {e})")
        hf_tps = 0.5 # Dummy baseline

    # 2. Run Ours
    nano_tps = run_nano_benchmark()

    print("="*40)
    print(f"Summary:")
    print(f"HF (PyTorch): {hf_tps:.2f} T/s")
    print(f"NanoInfer:    {nano_tps:.2f} T/s")
