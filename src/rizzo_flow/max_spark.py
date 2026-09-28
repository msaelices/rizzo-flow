"""Spark2.5 as MAX graphs: prefill a shared prefix, then read letter logits after short suffixes.

Two compiled graphs with symbolic shapes. The weights are trailing graph inputs, not constants,
so both graphs read the same device buffers (as constants, each compiled model would hold its
own copy on the device):

- `prefill`: one sequence of new tokens after `P` cached ones; returns the K/V of all `P + S`
  positions for every layer. Called in chunks, so attention intermediates stay bounded.
- `score`: `B` right-padded suffixes that all continue the same cached prefix; returns the
  logits of the requested token ids at each suffix's last real position. The prefix K/V is
  broadcast to the batch, never copied per branch, and nothing is written back.

Attention is written with 2-D matmuls and an fp32 mask and softmax (like MLX's fused kernel),
split between the shared prefix and each row's own new tokens. RoPE angles are computed in fp32 from the
positions, as MLX does, rather than read from a BF16 table.
"""

from dataclasses import dataclass

import numpy as np

SLIDING = "sliding_attention"
EMBEDDING = "model.embedding.weight"
# Graph weight role -> the checkpoint roles concatenated (along rows) to build it.
FUSED = {"qkvg": ("qkv", "gate"), "mlp_gate_up": ("mlp_gate", "mlp_up")}


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
    """Builds the graphs. Both take the weights as trailing inputs, in `input_shapes()` order,
    after their own inputs and the per-layer K/V. Projections that read the same input are
    fused into one matmul: Q, K, V and the output gate; the MLP's gate and up."""

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

    def input_shapes(self) -> dict[str, tuple[list[str], tuple[int, ...]]]:
        """The graphs' weight inputs, in order: key -> (checkpoint tensors to concatenate along
        rows, resulting shape)."""
        c = self.config
        parts = self._block_weights()
        fused = {part for group in FUSED.values() for part in group}
        inputs = {
            "model.embedding.weight": (["model.embedding.weight"], (c.vocab_size, c.hidden_size)),
            "model.norm.weight": (["model.norm.weight"], (c.hidden_size,)),
        }
        for i in range(c.num_layers):
            prefix = f"model.layers.{i}."
            for role, (name, shape) in parts.items():
                if role not in fused:
                    inputs[prefix + role] = ([prefix + name], shape)
            for role, group in FUSED.items():
                rows = sum(parts[part][1][0] for part in group)
                names = [prefix + parts[part][0] for part in group]
                inputs[prefix + role] = (names, (rows, c.hidden_size))
        return inputs

    def weight_device(self, key):
        """The embedding table stays in host memory: the graphs only need the rows of the
        input tokens and of the candidate letters, gathered there and moved to the device.
        On a small GPU that is 0.5 GiB (1.7B) or 0.6 GiB (4B) for the activations instead."""
        from max.graph import DeviceRef

        return DeviceRef.CPU() if key == EMBEDDING else self.device

    def _weight_types(self):
        from max.graph import TensorType

        return [
            TensorType(self.dtype, shape, device=self.weight_device(key))
            for key, (_, shape) in self.input_shapes().items()
        ]

    def _embed(self, ids):
        """Rows of the host-side embedding table for int64 ids (on the host), on the device."""
        from max.graph import ops

        return ops.gather(self.embedding, ids, axis=0).to(self.device)

    def _bind_weights(self, values):
        named = dict(zip(self.input_shapes(), values, strict=True))
        self.embedding = named[EMBEDDING]
        self.norm = named["model.norm.weight"]
        fused = {part for group in FUSED.values() for part in group}
        roles = [role for role in self._block_weights() if role not in fused]
        self.layers = [
            {role: named[f"model.layers.{i}.{role}"] for role in [*roles, *FUSED]}
            for i in range(self.config.num_layers)
        ]

    # -- building blocks ---------------------------------------------------------------------

    def _linear(self, x, w):
        from max.graph import ops

        return ops.matmul(x, ops.transpose(w, 0, 1))

    def _rms_norm(self, x, w):
        from max.graph import ops

        # Scale by the weight in fp32 before rounding to BF16, as MLX's fused kernel does.
        return ops.rms_norm(x, w, self.config.rms_norm_eps, multiply_before_cast=True)

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

    def _masks(self, rows, positions, past_positions):
        """Additive fp32 masks [rows, 1, S, P + rows * S] per layer kind, built once per graph,
        for `rows` sequences of S new tokens that all continue the same P cached ones. Keys are
        the P prefix tokens, which precede every new one (only the sliding window masks them),
        then all rows' new tokens, of which a query sees its own row's, causally."""
        from max.dtype import DType
        from max.graph import ops

        window = ops.constant(self.config.sliding_window, DType.int64, device=self.device)
        zero = ops.constant(0.0, DType.float32, device=self.device)
        neg = ops.constant(-np.inf, DType.float32, device=self.device)
        s = positions.shape[0]
        qpos = ops.unsqueeze(positions, 1)  # [S, 1]
        prefix = ops.where(ops.greater(ops.unsqueeze(past_positions, 0) + window, qpos), zero, neg)
        row_of = ops.constant(np.arange(rows).reshape(rows, 1), DType.int64, device=self.device)
        key_row = ops.reshape(ops.broadcast_to(row_of, [rows, s]), [rows * s])
        key_pos = ops.reshape(ops.broadcast_to(ops.unsqueeze(positions, 0), [rows, s]), [rows * s])
        same_row = ops.equal(ops.unsqueeze(ops.unsqueeze(row_of, 2), 3), key_row)  # [R,1,1,RS]
        causal = ops.greater_equal(ops.unsqueeze(qpos, 0), key_pos)  # [1, S, RS]
        in_window = ops.greater(key_pos + window, ops.unsqueeze(qpos, 0))
        full = ops.logical_and(same_row, ops.unsqueeze(causal, 1))  # [R, 1, S, RS]
        sliding = ops.logical_and(full, ops.unsqueeze(in_window, 1))
        shape = [rows, 1, s, past_positions.shape[0]]
        return {
            SLIDING: ops.concat(
                [ops.broadcast_to(prefix, shape), ops.where(sliding, zero, neg)], axis=-1
            ),
            "full_attention": ops.concat(
                [ops.broadcast_to(zero, shape), ops.where(full, zero, neg)], axis=-1
            ),
        }

    def _attention(self, q, k, v, past_k, past_v, mask):
        """Grouped-query attention of q [R, S, H, D] over the shared prefix (past K/V
        [P, Hkv, D]) and the rows' own new keys k, v [R, S, Hkv, D], with an additive `mask`
        [R, 1, S, P + R*S] from `_masks`. Returns [R, S, H, D].

        Per KV head, one matmul pair for all rows' queries at once, against the prefix keys
        followed by every row's new keys, block-masked to each row: the kernel count does not
        grow with the batch and the prefix is never copied per row. 2-D matmuls in the model dtype
        (tensor cores on GPUs; fp32 on the CPU, where MAX's BF16 matmul is slow), fp32 mask
        and softmax. With symbolic lengths MAX's batched matmul, and its flash kernel with
        256-wide heads, were 5-11x slower on an RTX 3050 Ti."""
        from max.dtype import DType
        from max.graph import ops

        c = self.config
        group = c.num_heads // c.num_kv_heads
        rows, s = int(q.shape[0]), q.shape[1]
        p = past_k.shape[0]
        compute = q.dtype if self.device.is_gpu() else DType.float32
        scale = c.head_dim**-0.5
        heads = []
        for j in range(c.num_kv_heads):
            # Query heads j*G .. j*G+G-1 read KV head j, as in MLX and transformers.
            qj = ops.permute(q[:, :, j * group : (j + 1) * group, :], [0, 2, 1, 3])  # [R,G,S,D]
            qj = ops.cast(ops.reshape(qj, [rows * group * s, c.head_dim]), compute)
            # Keys and values [P + R*S, D]: the prefix, then every row's new tokens.
            kj, vj = (
                ops.cast(
                    ops.concat(
                        [past[:, j, :], ops.reshape(x[:, :, j, :], [rows * s, c.head_dim])], axis=0
                    ),
                    compute,
                )
                for past, x in ((past_k, k), (past_v, v))
            )
            # Scores in fp32, as in MLX: BF16 scores moved probabilities by up to 0.03.
            qf, kf = ops.cast(qj, DType.float32), ops.cast(kj, DType.float32)
            scores = ops.matmul(qf, ops.transpose(kf, 0, 1)) * scale
            scores = ops.reshape(scores, [rows, group, s, p + rows * s])
            probs = ops.softmax(scores + mask)
            probs = ops.cast(ops.reshape(probs, [rows * group * s, p + rows * s]), compute)
            out = ops.matmul(probs, vj)  # [R*G*S, D]
            out = ops.permute(ops.reshape(out, [rows, group, s, c.head_dim]), [0, 2, 1, 3])
            heads.append(out)  # [R, S, G, D]
        return ops.cast(ops.concat(heads, axis=2), q.dtype)

    def _layer(self, i, h, positions, past_k, past_v, masks):
        """One block on h [R, S, hidden]; past K/V [P, Hkv, D] shared by the rows.
        Returns (h, k, v) with k, v the rows' new keys and values [R, S, Hkv, D]."""
        from max.dtype import DType
        from max.graph import ops

        c = self.config
        w = self.layers[i]
        kind = c.layer_types[i]
        b, s = h.shape[0], h.shape[1]
        x = self._rms_norm(h, w["input_norm"])
        qkvg = self._linear(x, w["qkvg"])
        q_size = c.num_heads * c.head_dim
        kv_size = c.num_kv_heads * c.head_dim
        end = q_size + 2 * kv_size
        q = ops.reshape(qkvg[..., :q_size], [b, s, c.num_heads, c.head_dim])
        k = ops.reshape(qkvg[..., q_size : q_size + kv_size], [b, s, c.num_kv_heads, c.head_dim])
        v = ops.reshape(qkvg[..., q_size + kv_size : end], [b, s, c.num_kv_heads, c.head_dim])
        q = self._rope(q, positions, kind)
        k = self._rope(k, positions, kind)
        attn = self._attention(q, k, v, past_k, past_v, masks[kind])
        gate = ops.cast(ops.sigmoid(ops.cast(qkvg[..., end:], DType.float32)), h.dtype)
        attn = attn * ops.unsqueeze(gate, -1)
        h = h + self._linear(ops.reshape(attn, [b, s, c.num_heads * c.head_dim]), w["out"])
        x = self._rms_norm(h, w["post_norm"])
        gate_up = self._linear(x, w["mlp_gate_up"])
        mlp = ops.gelu(gate_up[..., : c.intermediate_size]) * gate_up[..., c.intermediate_size :]
        return h + self._linear(mlp, w["mlp_down"]), k, v

    def _cache_types(self):
        from max.graph import TensorType

        c = self.config
        kv = TensorType(self.dtype, ["P", c.num_kv_heads, c.head_dim], device=self.device)
        return [kv] * (2 * c.num_layers)

    # -- graphs ------------------------------------------------------------------------------

    def prefill(self):
        """Inputs: tokens [S], positions [S] (P..P+S-1), past positions [P], past K and V per
        layer, weights. Outputs: K and V per layer over all P + S positions."""
        from max.dtype import DType
        from max.graph import DeviceRef, Graph, TensorType, ops

        inputs = [
            TensorType(DType.int64, ["S"], device=DeviceRef.CPU()),
            TensorType(DType.int64, ["S"], device=self.device),
            TensorType(DType.int64, ["P"], device=self.device),
            *self._cache_types(),
            *self._weight_types(),
        ]
        with Graph("spark_prefill", input_types=inputs) as graph:
            values = [v.tensor for v in graph.inputs]
            tokens, positions, past_positions = values[:3]
            cache = values[3 : 3 + 2 * self.config.num_layers]
            self._bind_weights(values[3 + 2 * self.config.num_layers :])
            masks = self._masks(1, positions, past_positions)
            h = ops.unsqueeze(self._embed(tokens), 0)
            outputs = []
            for i in range(self.config.num_layers):
                past_k, past_v = cache[2 * i], cache[2 * i + 1]
                h, k, v = self._layer(i, h, positions, past_k, past_v, masks)
                outputs += [
                    ops.concat([past_k, ops.squeeze(k, 0)], axis=0),
                    ops.concat([past_v, ops.squeeze(v, 0)], axis=0),
                ]
            graph.output(*outputs)
        return graph

    def score(self, batch: int):
        """Inputs: tokens [batch, S] right-padded, positions [S], each row's last real token as a
        flat index `r * S + last` [B], candidate token ids [N], past positions [P], past K and
        V per layer, weights.
        Output: fp32 logits [batch, N] from the tied embedding rows of the candidates only."""
        from max.dtype import DType
        from max.graph import DeviceRef, Graph, TensorType, ops

        inputs = [
            TensorType(DType.int64, [batch, "S"], device=DeviceRef.CPU()),
            TensorType(DType.int64, ["S"], device=self.device),
            TensorType(DType.int64, [batch], device=self.device),
            TensorType(DType.int64, ["N"], device=DeviceRef.CPU()),
            TensorType(DType.int64, ["P"], device=self.device),
            *self._cache_types(),
            *self._weight_types(),
        ]
        with Graph("spark_score", input_types=inputs) as graph:
            values = [v.tensor for v in graph.inputs]
            tokens, positions, last, slots, past_positions = values[:5]
            cache = values[5 : 5 + 2 * self.config.num_layers]
            self._bind_weights(values[5 + 2 * self.config.num_layers :])
            masks = self._masks(batch, positions, past_positions)
            h = self._embed(tokens)  # [batch, S, hidden]
            for i in range(self.config.num_layers):
                h, _, _ = self._layer(i, h, positions, cache[2 * i], cache[2 * i + 1], masks)
            flat = ops.reshape(h, [h.shape[0] * h.shape[1], self.config.hidden_size])
            final = self._rms_norm(ops.gather(flat, last, axis=0), self.norm)  # [B, hidden]
            head = self._embed(slots)  # [N, hidden]
            logits = ops.matmul(final, ops.transpose(head, 0, 1))
            graph.output(ops.cast(logits, DType.float32))
        return graph
