import mmap
import struct
import numpy as np
import os

class MappedWeights:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.f = open(model_path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

        # 1. Read Header
        header_data = self.mm[:28]
        values = struct.unpack("<7I", header_data)

        self.config = {
            "magic": values[0],
            "vocab_size": values[1],
            "dim": values[2],
            "layers": values[3],
            "n_heads": values[4],
            "n_kv_heads": values[5],
            "hidden_dim": values[6]
        }

        print(f"Model Config Loaded: {self.config}")

        self.offset = 256
        self.weights = []
        self.scales = []

        # 2. Map Tensors
        layer_shapes = self._get_layer_shapes()
        print("Mapping tensors...")
        for shape in layer_shapes:
            self._map_tensor(shape)

    def _get_layer_shapes(self):
        c = self.config
        dim = c["dim"]
        hidden = c["hidden_dim"]
        head_dim = dim // c["n_heads"]
        shapes = []
        shapes.append((c["vocab_size"], dim)) # Embed
        for _ in range(c["layers"]):
            shapes.append((dim,)) # Attn Norm

            # SWAPPED SHAPES (In -> Out)
            shapes.append((dim, dim)) # Q
            shapes.append((dim, c["n_kv_heads"] * head_dim)) # K [Dim -> Heads]
            shapes.append((dim, c["n_kv_heads"] * head_dim)) # V [Dim -> Heads]
            shapes.append((dim, dim)) # O

            shapes.append((dim,)) # FFN Norm

            # SWAPPED SHAPES
            shapes.append((dim, hidden)) # Gate [Dim -> Hidden]
            shapes.append((dim, hidden)) # Up   [Dim -> Hidden]
            shapes.append((hidden, dim)) # Down [Hidden -> Dim]

        shapes.append((dim,)) # Final Norm
        shapes.append((dim, c["vocab_size"])) # Head [Dim -> Vocab]

        return shapes

    def _map_tensor(self, shape):
        # 1. Read Int8 Weights
        n_elements = np.prod(shape)
        weight_view = np.frombuffer(self.mm, dtype=np.int8, count=n_elements, offset=self.offset)
        if len(shape) > 1:
            weight_view = weight_view.reshape(shape)

        self.weights.append(weight_view)
        self.offset += n_elements

        # 2. Read Float32 Scales (THE FIX)
        # Determine how many scales to read based on shape
        if len(shape) == 1:
            n_scales = 1 # 1D tensor = 1 scale
        else:
            n_scales = shape[0] # 2D tensor = 1 scale per row

        scale_view = np.frombuffer(self.mm, dtype=np.float32, count=n_scales, offset=self.offset)

        # If scalar, store as float. If array, store as array.
        if n_scales == 1:
            self.scales.append(scale_view[0])
        else:
            self.scales.append(scale_view)

        self.offset += n_scales * 4 # 4 bytes per float

    def close(self):
        self.mm.close()
        self.f.close()
