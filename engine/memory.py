import numpy as np

class KVCache:
    def __init__(self, config, max_context=2048):
        """
        Statically allocates memory for the entire conversation history.

        Shape: [Layers, 2, KV_Heads, Context_Len, Head_Dim]
        """
        self.max_context = max_context

        # FIX: Use dictionary access ['key'] instead of dot notation .key
        self.n_kv_heads = config["n_kv_heads"]
        self.head_dim = config["dim"] // config["n_heads"]
        self.layers = config["layers"]

        print(f"Allocating KV Cache: {self.layers} layers, {max_context} tokens...")

        # WE ALLOCATE ONCE. NO RE-ALLOCATION DURING CHAT.
        # Format: (Layer, Key/Val, Head, Seq, Dim)
        self.cache = np.zeros(
            (self.layers, 2, self.n_kv_heads, max_context, self.head_dim),
            dtype=np.float32
        )

        # Usage tracking
        self.current_pos = 0

    def update(self, layer_idx, k_vector, v_vector, pos):
        """
        Writes the new Key/Value vectors into the pre-allocated slot.
        """
        if pos >= self.max_context:
            pos = pos % self.max_context

        # Write directly into the reserved memory slot
        self.cache[layer_idx, 0, :, pos, :] = k_vector
        self.cache[layer_idx, 1, :, pos, :] = v_vector

    def get_context(self, layer_idx, pos):
        """
        Retrieves the history for Attention computation.
        """
        # We slice from 0 to pos+1.
        k_hist = self.cache[layer_idx, 0, :, :pos+1, :]
        v_hist = self.cache[layer_idx, 1, :, :pos+1, :]
        return k_hist, v_hist

    def reset(self):
        """Clears the cache for a new conversation."""
        self.current_pos = 0
