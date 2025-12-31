import argparse
import numpy as np
from packer.loader import load_model
from packer.writer import ModelHeader, write_model
from packer.quantize import quantize_tensor  # <--- NEW IMPORT

def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace model to NanoInfer binary")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--out", type=str, default="models/tinyllama.nano")
    args = parser.parse_args()

    # 1. Load
    print(f"Loading {args.model}...")
    config, raw_tensors = load_model(args.model)

    # 2. Quantize (Now using robust logic)
    print("Quantizing weights (Int8 Per-Channel)...")
    quantized_weights = []
    scales_list = []

    for i, tensor in enumerate(raw_tensors):
        # The magic happens here
        q_tensor, scale = quantize_tensor(tensor)

        quantized_weights.append(q_tensor)
        scales_list.append(scale)

        if i % 10 == 0:
            print(f"   Quantized tensor {i}/{len(raw_tensors)}...")

    # 3. Create Header
    header = ModelHeader(
        vocab_size=config["vocab_size"],
        dim=config["dim"],
        layers=config["layers"],
        n_heads=config["n_heads"],
        n_kv_heads=config["n_kv_heads"],
        hidden_dim=config["hidden_dim"]
    )

    # 4. Write
    write_model(args.out, header, quantized_weights, scales_list)
    print(f"Conversion Complete! Model saved to {args.out}")

if __name__ == "__main__":
    main()
