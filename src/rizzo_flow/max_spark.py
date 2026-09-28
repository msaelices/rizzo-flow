"""Spark2.5 as MAX graphs: prefill a shared prefix, then read letter logits after short suffixes.

Two compiled graphs with symbolic shapes, both reading the same weights:

- `prefill`: one sequence of new tokens after `P` cached ones; returns the K/V of all `P + S`
  positions for every layer. Called in chunks, so attention intermediates stay bounded.
- `score`: `B` right-padded suffixes that all continue the same cached prefix; returns the
  logits of the requested token ids at each suffix's last real position. The prefix K/V is
  broadcast to the batch, never copied per branch, and nothing is written back.

Attention is written with plain graph ops (fp32 scores and softmax, like MLX's fused kernel), so
the same graphs compile for the CPU and for GPUs. RoPE angles are computed in fp32 from the
positions, as MLX does, rather than read from a BF16 table.
"""

from dataclasses import dataclass

import numpy as np

SLIDING = "sliding_attention"


@dataclass(frozen=True)
class SparkConfig:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    sliding_window: int
    rms_norm_eps: float
    layer_types: tuple[str, ...]
    rope: dict  # layer type -> (theta, rotary dims)

    @classmethod
    def from_hf(cls, config: dict):
        if config.get("model_type") != "spark2_5":
            raise ValueError("Only the Spark2.5 architecture is supported")
        if not config.get("tie_word_embeddings", True):
            raise ValueError("Spark2.5 without tied embeddings is not supported")
        if config.get("hidden_act", "gelu") != "gelu" or config.get("attention_bias"):
            raise ValueError("Unsupported Spark2.5 variant (activation or attention bias)")
        head_dim = config["head_dim"]
        rope = {}
        for kind, params in config["rope_parameters"].items():
            dims = int(head_dim * params.get("partial_rotary_factor", 1.0))
            rope[kind] = (float(params.get("rope_theta", 10000.0)), dims)
        return cls(
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_layers=config["num_hidden_layers"],
            num_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            head_dim=head_dim,
            vocab_size=config["vocab_size"],
            sliding_window=config["sliding_window"],
            rms_norm_eps=config.get("rms_norm_eps", 1e-6),
            layer_types=tuple(config["layer_types"]),
            rope=rope,
        )


class SparkGraphs:
    """Builds the graphs. Weights are named as in the checkpoint, so the session binds them
    straight from the safetensors files; each graph gets its own `Weight` objects."""

    def __init__(self, config: SparkConfig, dtype, device):
        self.config = config
        self.dtype = dtype
        self.device = device

    # Short role -> (checkpoint name suffix, shape) for each block.
    def _block_weights(self):
        c = self.config
        qkv = (c.num_heads + 2 * c.num_kv_heads) * c.head_dim
        return {
            "input_norm": ("input_layernorm.weight", (c.hidden_size,)),
            "post_norm": ("post_attention_layernorm.weight", (c.hidden_size,)),
            "qkv": ("self_attn.q_k_v_proj.weight", (qkv, c.hidden_size)),
            "gate": ("self_attn.g_proj.weight", (c.num_heads, c.hidden_size)),
            "out": ("self_attn.out_proj.weight", (c.hidden_size, c.num_heads * c.head_dim)),
            "mlp_gate": ("mlp.gate_proj.weight", (c.intermediate_size, c.hidden_size)),
            "mlp_up": ("mlp.up_proj.weight", (c.intermediate_size, c.hidden_size)),
            "mlp_down": ("mlp.down_proj.weight", (c.hidden_size, c.intermediate_size)),
        }

    def weight_shapes(self) -> dict[str, tuple[int, ...]]:
        """Every tensor the graphs read, by checkpoint name."""
        c = self.config
        shapes = {
            "model.embedding.weight": (c.vocab_size, c.hidden_size),
            "model.norm.weight": (c.hidden_size,),
        }
        for i in range(c.num_layers):
            for name, shape in self._block_weights().values():
                shapes[f"model.layers.{i}.{name}"] = shape
        return shapes

    def _declare_weights(self):
        from max.graph import Weight

        def weight(name, shape):
            return Weight(name, self.dtype, shape, device=self.device)

        c = self.config
        self.embedding = weight("model.embedding.weight", (c.vocab_size, c.hidden_size))
        self.norm = weight("model.norm.weight", (c.hidden_size,))
        self.layers = [
            {
                role: weight(f"model.layers.{i}.{name}", shape)
                for role, (name, shape) in self._block_weights().items()
            }
            for i in range(c.num_layers)
        ]

    # -- building blocks ---------------------------------------------------------------------

    def _linear(self, x, w):
        from max.graph import ops

        return ops.matmul(x, ops.transpose(w, 0, 1))

    def _rms_norm(self, x, w):
        from max.graph import ops

        return ops.rms_norm(x, w, self.config.rms_norm_eps)

    def _rope(self, x, positions, kind):
        """Rotate the first `dims` channels of `x` [..., S, heads, head_dim] (half-split pairs),
        pass the rest through. `positions` is int64 [S]."""
        from max.dtype import DType
        from max.graph import ops

        theta, dims = self.config.rope[kind]
        half = dims // 2
        inv_freq = (1.0 / theta ** (np.arange(0, dims, 2, dtype=np.float64) / dims)).astype(
            np.float32
        )
        freq = ops.constant(inv_freq, DType.float32, device=self.device)
        angles = ops.outer(ops.cast(positions, DType.float32), freq)  # [S, half]
        cos = ops.unsqueeze(ops.cos(angles), 1)  # [S, 1, half]
        sin = ops.unsqueeze(ops.sin(angles), 1)
        xf = ops.cast(x, DType.float32)
        x1 = xf[..., :half]
        x2 = xf[..., half:dims]
        parts = [x1 * cos - x2 * sin, x2 * cos + x1 * sin]
        if dims < self.config.head_dim:
            parts.append(xf[..., dims:])
        return ops.cast(ops.concat(parts, axis=-1), x.dtype)

    def _attention(self, q, k, v, query_positions, key_positions, sliding):
        """q [B, S, H, D]; k, v [B, T, Hkv, D]; grouped-query attention with a causal (and, for
        sliding layers, windowed) mask built from absolute positions. Returns [B, S, H, D]."""
        from max.dtype import DType
        from max.graph import ops

        c = self.config
        group = c.num_heads // c.num_kv_heads
        b, s = q.shape[0], q.shape[1]
        t = k.shape[1]
        # Query head h reads KV head h // group, as in MLX and transformers.
        qg = ops.reshape(q, [b, s, c.num_kv_heads, group, c.head_dim])
        qg = ops.permute(qg, [0, 2, 3, 1, 4])  # [B, Hkv, G, S, D]
        qg = ops.reshape(ops.cast(qg, DType.float32), [b, c.num_kv_heads, group * s, c.head_dim])
        kt = ops.cast(ops.permute(k, [0, 2, 3, 1]), DType.float32)  # [B, Hkv, D, T]
        scores = ops.matmul(qg, kt) * (c.head_dim**-0.5)  # [B, Hkv, G*S, T]
        scores = ops.reshape(scores, [b, c.num_kv_heads, group, s, t])
        qp = ops.unsqueeze(query_positions, 1)  # [S, 1]
        kp = ops.unsqueeze(key_positions, 0)  # [1, T]
        allowed = ops.greater_equal(qp, kp)
        if sliding:
            window = ops.constant(c.sliding_window, DType.int64, device=self.device)
            allowed = ops.logical_and(allowed, ops.greater(kp + window, qp))
        neg = ops.constant(-np.inf, DType.float32, device=self.device)
        zero = ops.constant(0.0, DType.float32, device=self.device)
        scores = scores + ops.where(allowed, zero, neg)
        probs = ops.softmax(scores)
        probs = ops.reshape(probs, [b, c.num_kv_heads, group * s, t])
        vt = ops.cast(ops.permute(v, [0, 2, 1, 3]), DType.float32)  # [B, Hkv, T, D]
        out = ops.matmul(probs, vt)  # [B, Hkv, G*S, D]
        out = ops.reshape(out, [b, c.num_kv_heads, group, s, c.head_dim])
        out = ops.permute(out, [0, 3, 1, 2, 4])  # [B, S, Hkv, G, D]
        return ops.cast(ops.reshape(out, [b, s, c.num_heads, c.head_dim]), q.dtype)

    def _layer(self, i, h, positions, key_positions, past_k, past_v):
        """One block on h [B, S, hidden]; past K/V [P, Hkv, D] shared by the batch.
        Returns (h, k, v) with k, v the full [B, P + S, Hkv, D] keys and values."""
        from max.dtype import DType
        from max.graph import ops

        c = self.config
        w = self.layers[i]
        kind = c.layer_types[i]
        b, s = h.shape[0], h.shape[1]
        x = self._rms_norm(h, w["input_norm"])
        qkv = self._linear(x, w["qkv"])
        q_size = c.num_heads * c.head_dim
        kv_size = c.num_kv_heads * c.head_dim
        q = ops.reshape(qkv[..., :q_size], [b, s, c.num_heads, c.head_dim])
        k = ops.reshape(qkv[..., q_size : q_size + kv_size], [b, s, c.num_kv_heads, c.head_dim])
        v = ops.reshape(qkv[..., q_size + kv_size :], [b, s, c.num_kv_heads, c.head_dim])
        q = self._rope(q, positions, kind)
        k = self._rope(k, positions, kind)
        p = past_k.shape[0]
        shared = [b, p, c.num_kv_heads, c.head_dim]
        k = ops.concat([ops.broadcast_to(ops.unsqueeze(past_k, 0), shared), k], axis=1)
        v = ops.concat([ops.broadcast_to(ops.unsqueeze(past_v, 0), shared), v], axis=1)
        attn = self._attention(q, k, v, positions, key_positions, kind == SLIDING)
        gate = ops.cast(ops.sigmoid(ops.cast(self._linear(x, w["gate"]), DType.float32)), h.dtype)
        attn = attn * ops.unsqueeze(gate, -1)
        h = h + self._linear(ops.reshape(attn, [b, s, c.num_heads * c.head_dim]), w["out"])
        x = self._rms_norm(h, w["post_norm"])
        mlp = ops.gelu(self._linear(x, w["mlp_gate"])) * self._linear(x, w["mlp_up"])
        return h + self._linear(mlp, w["mlp_down"]), k, v

    def _cache_types(self):
        from max.graph import TensorType

        c = self.config
        kv = TensorType(self.dtype, ["P", c.num_kv_heads, c.head_dim], device=self.device)
        return [kv] * (2 * c.num_layers)

    # -- graphs ------------------------------------------------------------------------------

    def prefill(self):
        """Inputs: tokens [S], positions [S] (P..P+S-1), past positions [P], then past K and V
        per layer. Outputs: K and V per layer over all P + S positions."""
        from max.dtype import DType
        from max.graph import Graph, TensorType, ops

        inputs = [
            TensorType(DType.int64, ["S"], device=self.device),
            TensorType(DType.int64, ["S"], device=self.device),
            TensorType(DType.int64, ["P"], device=self.device),
            *self._cache_types(),
        ]
        with Graph("spark_prefill", input_types=inputs) as graph:
            self._declare_weights()
            tokens, positions, past_positions, *cache = (v.tensor for v in graph.inputs)
            key_positions = ops.concat([past_positions, positions], axis=0)
            h = ops.unsqueeze(ops.gather(self.embedding, tokens, axis=0), 0)
            outputs = []
            for i in range(self.config.num_layers):
                h, k, v = self._layer(
                    i, h, positions, key_positions, cache[2 * i], cache[2 * i + 1]
                )
                outputs += [ops.squeeze(k, 0), ops.squeeze(v, 0)]
            graph.output(*outputs)
        return graph

    def score(self):
        """Inputs: tokens [B, S] right-padded, positions [S], each row's last real token as a
        flat index `r * S + last` [B], candidate token ids [N], past positions [P], past K and
        V per layer.
        Output: fp32 logits [B, N] from the tied embedding rows of the candidates only."""
        from max.dtype import DType
        from max.graph import Graph, TensorType, ops

        inputs = [
            TensorType(DType.int64, ["B", "S"], device=self.device),
            TensorType(DType.int64, ["S"], device=self.device),
            TensorType(DType.int64, ["B"], device=self.device),
            TensorType(DType.int64, ["N"], device=self.device),
            TensorType(DType.int64, ["P"], device=self.device),
            *self._cache_types(),
        ]
        with Graph("spark_score", input_types=inputs) as graph:
            self._declare_weights()
            tokens, positions, last, slots, past_positions, *cache = (
                v.tensor for v in graph.inputs
            )
            key_positions = ops.concat([past_positions, positions], axis=0)
            h = ops.gather(self.embedding, tokens, axis=0)  # [B, S, hidden]
            for i in range(self.config.num_layers):
                h, _, _ = self._layer(
                    i, h, positions, key_positions, cache[2 * i], cache[2 * i + 1]
                )
            flat = ops.reshape(h, [h.shape[0] * h.shape[1], self.config.hidden_size])
            final = self._rms_norm(ops.gather(flat, last, axis=0), self.norm)  # [B, hidden]
            head = ops.gather(self.embedding, slots, axis=0)  # [N, hidden]
            logits = ops.matmul(final, ops.transpose(head, 0, 1))
            graph.output(ops.cast(logits, DType.float32))
        return graph
