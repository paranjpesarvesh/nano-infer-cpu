import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Downloads the model and flattens the weights into a strict execution order.
    Returns:
        config (dict): {dim, layers, heads, kv_heads, vocab, hidden}
        tensors (list): A list of pure numpy arrays (Float16/Float32)
    """
    print(f"Downloading {model_id} from HuggingFace...")
    # Load model to CPU memory (RAM)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # Load as float32 for accurate quantization later
        low_cpu_mem_usage=True
    )

    # Extract Hyperparameters
    params = hf_model.config
    config = {
        "vocab_size": params.vocab_size,
        "dim": params.hidden_size,           # 2048
        "layers": params.num_hidden_layers,  # 22
        "n_heads": params.num_attention_heads, # 32
        "n_kv_heads": getattr(params, "num_key_value_heads", params.num_attention_heads), # 4 (GQA)
        "hidden_dim": params.intermediate_size # 5632
    }

    print(f"Model Config: {config}")

    # Extract State Dict
    sd = hf_model.state_dict()

    # The Order List: This defines the "Lifecycle of a Token" in our engine
    ordered_tensors = []

    # 1. Token Embeddings
    print("Packing Embedding Layer...")
    ordered_tensors.append(sd["model.embed_tokens.weight"].numpy())

    # 2. Transformer Layers
    for i in range(config["layers"]):
        layer_prefix = f"model.layers.{i}"

        # Attention Block
        # Note: We keep them separate. Some engines merge QKV, we won't for clarity.
        ordered_tensors.append(sd[f"{layer_prefix}.input_layernorm.weight"].numpy())
        ordered_tensors.append(sd[f"{layer_prefix}.self_attn.q_proj.weight"].numpy().T)
        ordered_tensors.append(sd[f"{layer_prefix}.self_attn.k_proj.weight"].numpy().T)
        ordered_tensors.append(sd[f"{layer_prefix}.self_attn.v_proj.weight"].numpy().T)
        ordered_tensors.append(sd[f"{layer_prefix}.self_attn.o_proj.weight"].numpy().T)

        # Feed Forward Block (SwiGLU)
        ordered_tensors.append(sd[f"{layer_prefix}.post_attention_layernorm.weight"].numpy())
        ordered_tensors.append(sd[f"{layer_prefix}.mlp.gate_proj.weight"].numpy().T) # Gate
        ordered_tensors.append(sd[f"{layer_prefix}.mlp.up_proj.weight"].numpy().T)   # Up
        ordered_tensors.append(sd[f"{layer_prefix}.mlp.down_proj.weight"].numpy().T) # Down

        if i % 5 == 0:
            print(f"   Processed Layer {i}/{config['layers']}...")

    # 3. Final Norm & Head
    print("Packing Final Layers...")
    ordered_tensors.append(sd["model.norm.weight"].numpy())
    ordered_tensors.append(sd["lm_head.weight"].numpy().T)

    return config, ordered_tensors

if __name__ == "__main__":
    # Test run
    c, t = load_model()
    print(f"Loaded {len(t)} tensors successfully.")
