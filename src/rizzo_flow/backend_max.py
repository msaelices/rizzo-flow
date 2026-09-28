"""MAX scoring: Spark2.5 as compiled MAX graphs, no generation, candidate rows only.

Same contract as `backend.SparkBackend` and `backend_llama.LlamaBackend` (tokenizer, `metadata`,
`score`). The weights are the BF16 safetensors checkpoint that MLX loads, bound by name from the
files. The prefix K/V stays on the device between calls; branches read it, never write it.
"""

import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np

from .chat_template import compile_template
from .config import FLOW_CHECKPOINTS, checkpoint_hashes, identify
from .max_spark import SparkConfig, SparkGraphs
from .prompts import PROMPT_VERSION, Compiled, canonical


class MaxTokenizer:
    """The two tokenizer calls `prompts.compile_request` makes: the checkpoint's chat template
    rendered as transformers renders it, and its `tokenizer.json` for encoding."""

    def __init__(self, path: Path):
        from tokenizers import Tokenizer

        config = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
        # transformers prefers the standalone template file over the config entry.
        template_file = path / "chat_template.jinja"
        if template_file.is_file():
            template = template_file.read_text(encoding="utf-8")
        else:
            template = config.get("chat_template")
        if not template:
            raise ValueError("The checkpoint carries no chat template")
        self.render = compile_template(template)
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.pad_token_id = self._special(config.get("pad_token"))
        self.eos_token_id = self._special(config.get("eos_token"))

    def _special(self, token):
        if isinstance(token, dict):
            token = token.get("content")
        return self.tokenizer.token_to_id(token) if token else None

    def apply_chat_template(self, messages, tokenize=False, **variables) -> str:
        if tokenize:
            raise ValueError("Render the text, then call encode()")
        return self.render(messages=messages, **variables)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids


def pick_device(device="auto"):
    from max.driver import CPU, Accelerator, accelerator_count

    if device not in ("auto", "gpu", "cpu"):
        raise ValueError("The MAX backend takes --device auto, gpu or cpu")
    if device == "cpu":
        return CPU()
    if accelerator_count():
        return Accelerator()
    if device == "gpu":
        raise ValueError("MAX sees no GPU on this machine; use --device cpu")
    return CPU()


class SparkRunner:
    """The two compiled graphs and the device buffers they exchange."""

    def __init__(self, config: SparkConfig, weights, device, dtype):
        from max.engine import InferenceSession
        from max.graph import DeviceRef

        self.config = config
        self.device = device
        self.dtype = dtype
        graphs = SparkGraphs(config, dtype, DeviceRef.from_device(device))
        missing = set(graphs.weight_shapes()) - set(weights)
        if missing:
            raise ValueError(f"Checkpoint lacks {len(missing)} tensors, e.g. {min(missing)}")
        session = InferenceSession(devices=[device])
        self._prefill = session.load(graphs.prefill(), weights_registry=weights)
        self._score = session.load(graphs.score(), weights_registry=weights)

    def _buffer(self, array):
        from max.driver import Buffer

        return Buffer.from_numpy(np.ascontiguousarray(array)).to(self.device)

    def empty_cache(self):
        from max.driver import Buffer

        c = self.config
        empty = Buffer(self.dtype, [0, c.num_kv_heads, c.head_dim], device=self.device)
        return [empty] * (2 * c.num_layers)

    def prefill(self, tokens, chunk):
        """K/V for `tokens`, fed `chunk` at a time so attention scores stay bounded."""
        cache = self.empty_cache()
        for start in range(0, len(tokens), chunk):
            piece = tokens[start : start + chunk]
            cache = self._prefill.execute(
                self._buffer(np.array(piece, dtype=np.int64)),
                self._buffer(np.arange(start, start + len(piece), dtype=np.int64)),
                self._buffer(np.arange(start, dtype=np.int64)),
                *cache,
            )
        return cache

    def score(self, cache, prefix_length, suffixes, slots, pad):
        """fp32 logits [len(suffixes), len(slots)] after each suffix, all continuing `cache`."""
        width = max(map(len, suffixes))
        tokens = np.full((len(suffixes), width), pad, dtype=np.int64)
        for row, suffix in enumerate(suffixes):
            tokens[row, : len(suffix)] = suffix
        # Right padding is causal future context: each row reads its own last real position.
        last = np.array([r * width + len(s) - 1 for r, s in enumerate(suffixes)], dtype=np.int64)
        (logits,) = self._score.execute(
            self._buffer(tokens),
            self._buffer(np.arange(prefix_length, prefix_length + width, dtype=np.int64)),
            self._buffer(last),
            self._buffer(np.array(slots, dtype=np.int64)),
            self._buffer(np.arange(prefix_length, dtype=np.int64)),
            *cache,
        )
        from max.driver import CPU

        return logits.to(CPU()).to_numpy()

    def free_bytes(self):
        if self.device.is_host:
            return None
        return self.device.stats["free_memory"]


class MaxBackend:
    def __init__(
        self, runner, tokenizer, metadata, batch_size=4, prefill_chunk=512, idle_free=None
    ):
        if not 1 <= batch_size <= 16 or not 1 <= prefill_chunk <= 2048:
            raise ValueError("batch_size must be 1–16 and prefill_chunk 1–2048")
        self.runner = runner
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.batch_size = batch_size
        self.prefill_chunk = prefill_chunk
        self._idle_free = idle_free  # free device memory before the weights were loaded
        self._lowest_free = None

    @classmethod
    def load(cls, path, device="auto", ctx=8192, batch_size=4, prefill_chunk=512):
        from max.dtype import DType
        from max.graph.weights import load_weights

        path = Path(path).resolve()
        if not path.is_dir():
            raise ValueError(f"Model not found at {path}. Run `rizzo download --backend max`.")
        if ctx < 1:
            raise ValueError("ctx must be positive")
        started = time.perf_counter()
        target = pick_device(device)
        idle_free = None if target.is_host else target.stats["free_memory"]
        hashes = checkpoint_hashes(path)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        spark = SparkConfig.from_hf(config)
        spec = identify(config)
        weights = load_weights(sorted(path.glob("*.safetensors")))
        registry = {name: weight.data() for name, weight in weights.items()}
        runner = SparkRunner(spark, registry, target, DType.bfloat16)
        identity = {
            "source": spec.repo,
            "requested_revision": spec.revision,
            "source_files": hashes,
            "precision": "bf16",
            "device": "cpu" if target.is_host else "gpu",
            "backend": "cpu" if target.is_host else target.api,
            "runtime": "max",
            "max": importlib.metadata.version("max"),
            "prompt_version": PROMPT_VERSION,
        }
        # Only the fine-tune adds a key, as with the other backends.
        flow = FLOW_CHECKPOINTS.get(spec.size)
        if flow and flow.weights and all(hashes.get(k) == v for k, v in flow.weights.items()):
            identity["weights"] = "flow"
        metadata = {
            **identity,
            "fingerprint": hashlib.sha256(canonical(identity).encode()).hexdigest(),
            "device_name": None if target.is_host else target.label,
            "load_seconds": time.perf_counter() - started,
        }
        return cls(runner, MaxTokenizer(path), metadata, batch_size, prefill_chunk, idle_free)

    def _track_memory(self):
        free = self.runner.free_bytes()
        if free is not None:
            self._lowest_free = free if self._lowest_free is None else min(self._lowest_free, free)

    def peak_device_bytes(self) -> int | None:
        """Largest drop in free device memory since before the load (weights included). Other
        processes using the same GPU are counted too; None on the CPU."""
        if self._idle_free is None or self._lowest_free is None:
            return None
        return max(self._idle_free - self._lowest_free, 0)

    def score(self, prefix: list[int], jobs: list[Compiled], mode="shared"):
        if mode not in ("shared", "direct"):
            raise ValueError("Unknown execution mode")
        if not jobs:
            raise ValueError("No decisions supplied")
        if any(
            job.tokens[: len(prefix)] != prefix or len(job.tokens) <= len(prefix) for job in jobs
        ):
            raise ValueError("Invalid shared prefix")
        runner = self.runner
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id or 0
        started = time.perf_counter()
        result = {}
        prefix_seconds = 0.0
        evaluated_tokens = 0
        batches = 0
        # A lone question has nobody to share the prefix with: one pass, as with llama.cpp.
        reuse = mode == "shared" and bool(prefix) and len(jobs) > 1
        if not reuse:
            for job in jobs:
                cache = runner.prefill(job.tokens[:-1], self.prefill_chunk)
                logits = runner.score(cache, len(job.tokens) - 1, [job.tokens[-1:]], job.slots, pad)
                result[job.id] = logits[0].tolist()
                evaluated_tokens += len(job.tokens)
                batches += 1
                del cache
        else:
            mark = time.perf_counter()
            cache = runner.prefill(prefix, self.prefill_chunk)
            prefix_seconds = time.perf_counter() - mark
            evaluated_tokens += len(prefix)
            ordered = sorted(jobs, key=lambda j: len(j.tokens))
            for offset in range(0, len(ordered), self.batch_size):
                group = ordered[offset : offset + self.batch_size]
                suffixes = [job.tokens[len(prefix) :] for job in group]
                union = sorted({slot for job in group for slot in job.slots})
                logits = runner.score(cache, len(prefix), suffixes, union, pad)
                for job, row in zip(group, logits, strict=True):
                    result[job.id] = [float(row[union.index(slot)]) for slot in job.slots]
                evaluated_tokens += len(group) * max(map(len, suffixes))
                batches += 1
            del cache
        self._track_memory()
        timing = {
            "inference_seconds": time.perf_counter() - started,
            "prefill_seconds": prefix_seconds,
            "shared_prefix_tokens": len(prefix) if reuse else 0,
            "evaluated_tokens_including_padding": evaluated_tokens,
            "logical_input_tokens": sum(len(j.tokens) for j in jobs),
            "batches": batches,
            "generated_tokens": 0,
        }
        if self._lowest_free is not None:
            timing["peak_device_bytes"] = self.peak_device_bytes()
        return result, timing
