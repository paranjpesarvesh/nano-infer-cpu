import numpy as np
from engine.kernels import rmsnorm, matmul_int8, silu, softmax, apply_rope
from engine.memory import KVCache

class Transformer:
    def __init__(self, weights):
        self.w = weights
        self.c = weights.config

        # Initialize Memory (KV Cache)
        self.kv = KVCache(self.c)

        # Pointers to specific weight indices
        self.w_embed = self.w.weights[0]
        self.w_final_norm = self.w.weights[-2]
        self.w_head = self.w.weights[-1]

    def forward(self, token_id, pos):
        """
        Runs one forward pass for a single token.
        """
        dim = self.c["dim"]

        # --- 1. EMBEDDING (FIXED) ---
        # We select the Int8 vector for the token
        # AND the specific float32 scale for this token (scalar)
        token_scale = self.w.scales[0][token_id]
        x = self.w_embed[token_id].astype(np.float32) * token_scale

        # --- 2. LAYERS LOOP ---
        tensor_idx = 1 # Start after embedding

        for i in range(self.c["layers"]):
            # --- ATTENTION BLOCK ---
            resid = x.copy()

            # A. Norm (Dequantize first)
            # Norm weights are 1D, so scale is a scalar. This works fine.
            w_norm_att = self.w.weights[tensor_idx].astype(np.float32) * self.w.scales[tensor_idx]
            x = rmsnorm(x, w_norm_att)
            tensor_idx += 1

            # B. QKV Projections
            q = np.zeros(dim, dtype=np.float32)
            k = np.zeros(self.c["n_kv_heads"] * (dim // self.c["n_heads"]), dtype=np.float32)
            v = np.zeros_like(k)

            matmul_int8(x, self.w.weights[tensor_idx], self.w.scales[tensor_idx], q)
            matmul_int8(x, self.w.weights[tensor_idx+1], self.w.scales[tensor_idx+1], k)
            matmul_int8(x, self.w.weights[tensor_idx+2], self.w.scales[tensor_idx+2], v)
            tensor_idx += 3

            # C. RoPE & KV Update
            head_dim = dim // self.c["n_heads"]
            q = q.reshape(self.c["n_heads"], head_dim)
            k = k.reshape(self.c["n_kv_heads"], head_dim)
            v = v.reshape(self.c["n_kv_heads"], head_dim)

            q, k = apply_rope(q, k, pos)
            self.kv.update(i, k, v, pos)

            # D. Attention
            k_hist, v_hist = self.kv.get_context(i, pos)

            att_out = np.zeros(dim, dtype=np.float32)

            for h in range(self.c["n_heads"]):
                kv_h = h // (self.c["n_heads"] // self.c["n_kv_heads"])

                score = np.dot(k_hist[kv_h], q[h])
                score /= np.sqrt(head_dim)
                score = softmax(score)

                head_out = np.dot(score, v_hist[kv_h])
                att_out[h*head_dim : (h+1)*head_dim] = head_out

            # E. Output Projection
            x = np.zeros(dim, dtype=np.float32)
            matmul_int8(att_out, self.w.weights[tensor_idx], self.w.scales[tensor_idx], x)
            tensor_idx += 1

            x += resid

            # --- FEED FORWARD BLOCK ---
            resid = x.copy()

            # Norm
            w_norm_ffn = self.w.weights[tensor_idx].astype(np.float32) * self.w.scales[tensor_idx]
            x = rmsnorm(x, w_norm_ffn)
            tensor_idx += 1

            # Gate/Up
            hidden_dim = self.c["hidden_dim"]
            gate = np.zeros(hidden_dim, dtype=np.float32)
            up = np.zeros(hidden_dim, dtype=np.float32)

            matmul_int8(x, self.w.weights[tensor_idx], self.w.scales[tensor_idx], gate)
            matmul_int8(x, self.w.weights[tensor_idx+1], self.w.scales[tensor_idx+1], up)

            gate = silu(gate)
            inter = gate * up

            # Down
            x = np.zeros(dim, dtype=np.float32)
            matmul_int8(inter, self.w.weights[tensor_idx+2], self.w.scales[tensor_idx+2], x)
            tensor_idx += 3

            x += resid

        # --- 3. FINAL OUTPUT ---
        w_norm_final = self.w_final_norm.astype(np.float32) * self.w.scales[-2]
        x = rmsnorm(x, w_norm_final)

        logits = np.zeros(self.c["vocab_size"], dtype=np.float32)
        matmul_int8(x, self.w_head, self.w.scales[-1], logits)

        return logits
