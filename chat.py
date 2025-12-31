import time
import argparse
import numpy as np
from transformers import AutoTokenizer
from rich.live import Live

from engine.weights import MappedWeights
from engine.transformer import Transformer
from interface.tui import NanoTUI

def sample(logits, temperature=0.7):
    """
    Basic Temperature Sampling.
    """
    if temperature == 0.0:
        return np.argmax(logits)

    # Softmax with temp
    logits = logits / temperature
    max_logit = np.max(logits)
    probs = np.exp(logits - max_logit)
    probs /= np.sum(probs)

    # Sample
    # Note: simple weighted random choice
    token = np.random.choice(len(logits), p=probs)
    return token

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/tinyllama.nano")
    parser.add_argument("--hf_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    args = parser.parse_args()

    # 1. Init TUI
    tui = NanoTUI()

    # 2. Load Resources (with visual feedback)
    with Live(tui.render_layout(), refresh_per_second=4) as live:
        tui.history.append("System: Initializing Engine...\n", style="dim")

        # Load Weights
        tui.history.append(f"System: Mapping {args.model}...\n", style="dim")
        weights = MappedWeights(args.model)

        # Init Engine
        tui.history.append("System: Allocating KV Cache...\n", style="dim")
        model = Transformer(weights)

        # Load Tokenizer
        tui.history.append("System: Loading Tokenizer...\n", style="dim")
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model)

        tui.history.append("System: Ready. CTRL+C to exit.\n", style="bold green")

    # 3. Interaction Loop
    # We break out of 'Live' context to take input, then go back in.
    while True:
        try:
            user_input = input("\nUser > ")
            if user_input.lower() in ["exit", "quit"]:
                break

            # Prepare Prompt (TinyLlama Chat Format)
            prompt = f"<|system|>\nYou are a helpful assistant.\n</s>\n<|user|>\n{user_input}\n</s>\n<|assistant|>\n"
            tokens = tokenizer.encode(prompt)

            # Start Generation UI
            with Live(tui.render_layout(), refresh_per_second=10) as live:
                tui.history.append(f"\nUser: {user_input}\n", style="bold white")
                tui.history.append("AI: ", style="bold green")

                start_time = time.time()
                pos = 0

                # --- PREFILL PHASE ---
                # Process the prompt tokens
                for i, token in enumerate(tokens):
                    logits = model.forward(token, pos)
                    pos += 1

                # --- DECODE PHASE ---
                # Generate new tokens
                new_tokens = []
                next_token = sample(logits)

                while next_token != tokenizer.eos_token_id and len(new_tokens) < 512:
                    # 1. Print Token
                    word = tokenizer.decode([next_token])
                    tui.stream_token(word)
                    new_tokens.append(next_token)

                    # 2. Update Stats
                    dt = time.time() - start_time
                    tps = (len(new_tokens)) / dt if dt > 0 else 0
                    tui.update_stats(tps, pos)
                    live.update(tui.render_layout())

                    # 3. Forward Pass
                    logits = model.forward(next_token, pos)
                    next_token = sample(logits)
                    pos += 1

                tui.reset_input()

        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
