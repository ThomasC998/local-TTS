"""Assemble the Breeze main model and fill it from the original checkpoint.

The module tree here is built so that every parameter name matches the target
names ``breeze_tts_mlx.checkpoint`` already maps the Hugging Face tensors onto.
That is not a coincidence to preserve casually: it is what lets one mapping
table serve both backends, so a checkpoint that loads on the Mac cannot quietly
load *differently* here.

What this does not do is read MLX's INT8 artifact. Those weights are packed in
MLX's affine quantization format -- scales and biases per group of 64, in MLX's
own layout -- and nothing in PyTorch reads it. Windows has its own quantization,
described below; MODELS.md says which checkpoints exist and how to fetch them.

Two checkpoint formats
----------------------
*Safetensors*, ``model.safetensors.index.json`` plus shards: the original BF16
weights. Read one tensor at a time, so host memory never holds more than what is
being copied to the GPU.

*Pickle*, ``pytorch_model.bin.index.json`` plus ``.bin`` shards: how a torchao
INT8 conversion ships, because transformers cannot currently round-trip an INT8
torchao checkpoint through safetensors. The quantized weights are not tensors
but tensor *subclasses* -- ``torch.load`` reconstructs them, and they behave as
weights because torchao intercepts ``F.linear`` on them. Two consequences the
code below has to respect: a subclass must not be cast or transposed like a
plain tensor, and a whole shard is materialized at once rather than tensor by
tensor, so loading briefly needs about 5 GB of host RAM.

Which layers are quantized is the checkpoint's decision, not ours. In the
published INT8 conversion it is the backbone and depth-decoder linears; the text
encoder, the embeddings, the LM head and every norm stay BF16. Nothing here
needs to know that -- each tensor arrives already in whatever form it was saved.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from safetensors import safe_open
from torch import nn

from ._shared import BreezeConfig, map_source_tensor

logger = logging.getLogger("breeze.torch.model")
from .backbone import Qwen3Backbone
from .depth_decoder import BreezeDepthDecoder
from .text_encoder import T5GemmaTextEncoder

DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_dtype(name: str) -> torch.dtype:
    key = str(name).lower()
    if key not in DTYPES:
        raise ValueError(
            f"Unsupported dtype {name!r}; choose one of {sorted(set(DTYPES))}"
        )
    return DTYPES[key]


# --------------------------------------------------------------------------
# Reading a checkpoint, in either of the two formats it ships in
# --------------------------------------------------------------------------
SAFETENSORS_INDEX = "model.safetensors.index.json"
PICKLE_INDEX = "pytorch_model.bin.index.json"


def find_index(checkpoint_dir: Path) -> tuple[Path, str]:
    """The shard index and which format it describes."""
    for name, kind in ((SAFETENSORS_INDEX, "safetensors"), (PICKLE_INDEX, "pickle")):
        path = checkpoint_dir / name
        if path.is_file():
            return path, kind
    raise FileNotFoundError(
        f"{checkpoint_dir} has neither {SAFETENSORS_INDEX} nor {PICKLE_INDEX}, so it "
        "is not a complete Hugging Face checkpoint. MLX's INT8 artifact cannot be "
        "loaded by the CUDA backend -- see MODELS.md."
    )


def read_weight_map(index_path: Path) -> dict[str, str]:
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid or empty weight_map in {index_path}")
    return {str(name): str(shard) for name, shard in weight_map.items()}


# The two types a dense weight can arrive as. Anything else is a tensor
# subclass carrying packed data and its own scales.
_DENSE_TYPES = (torch.Tensor, nn.Parameter)


def is_plain(tensor: Any) -> bool:
    """Whether this is an ordinary tensor rather than a quantized subclass.

    ``isinstance`` would be true for both -- torchao's tensors *are* Tensors.
    What matters is the exact type: a subclass carries packed data and its own
    scales, and casting or transposing it as if it were a dense array either
    fails or silently produces something that is no longer the saved weight.
    """
    return type(tensor) in _DENSE_TYPES


def require_torchao(checkpoint_dir: Path) -> None:
    """Import torchao, which is what makes a quantized checkpoint loadable.

    Importing it registers its tensor classes as safe globals for
    ``torch.load``, and installs the ``F.linear`` handling that makes the
    loaded weights usable. Without it the load fails inside pickle with a
    message about an unsupported global, which names neither torchao nor this.
    """
    try:
        import torchao  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"{checkpoint_dir.name} is a torchao INT8 checkpoint, and torchao is "
            "not installed. Run `pip install torchao`, or download the BF16 "
            "checkpoint instead:\n"
            "    python download_model.py --variant torch-bf16"
        ) from exc


@contextmanager
def open_shard(path: Path, kind: str) -> Iterator[Callable[[str], Any]]:
    """Yield a ``get(name) -> tensor`` for one shard, in either format."""
    if kind == "safetensors":
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            yield handle.get_tensor
        return

    # weights_only=True still reconstructs torchao's tensors, because importing
    # torchao allowlists them. It is kept on: these files are several gigabytes
    # downloaded from a model host, and unpickling them arbitrarily is not
    # something to do by default.
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - re-raised with what to do about it
        raise RuntimeError(
            f"Could not read {path.name}: {exc}\n"
            "This is usually a torchao version mismatch -- the checkpoint was "
            "written with 0.17 and its tensor classes have to be the ones "
            "torch.load can reconstruct. Try `pip install torchao==0.17.0`."
        ) from exc
    try:
        yield loaded.__getitem__
    finally:
        loaded.clear()


class SharedAudioEmbedding(nn.Module):
    """One physical table for all sixteen codebooks, offset per codebook."""

    def __init__(self, num_embeddings: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, hidden_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class BreezeTorchModel(nn.Module):
    """Inference-only active Breeze main model on PyTorch.

    As in the MLX port, the legacy direct text embedding and the embedded Mimi
    codec are absent: neither is reachable from this project's pipeline, and
    together they are most of the checkpoint's size.
    """

    def __init__(self, config: BreezeConfig) -> None:
        super().__init__()
        self.breeze_config = config
        raw = config.model
        self.text_encoder = T5GemmaTextEncoder(config.text)
        self.text_encoder_proj = nn.Linear(
            int(config.text["hidden_size"]), int(raw["hidden_size"]), bias=False
        )
        self.backbone = Qwen3Backbone(config.backbone)
        self.depth_decoder = BreezeDepthDecoder(config.depth)
        self.audio_embedding = SharedAudioEmbedding(
            int(raw["num_codebooks"]) * int(raw["vocab_size"]),
            int(raw["hidden_size"]),
        )
        self.lm_head = nn.Linear(
            int(raw["hidden_size"]), int(raw["vocab_size"]) + 1, bias=False
        )
        self.compute_dtype = torch.bfloat16

    def set_compute_dtype(self, dtype: torch.dtype) -> None:
        self.compute_dtype = dtype
        self.text_encoder.set_compute_dtype(dtype)

    def embed_audio_frames(self, code_ids: torch.Tensor) -> torch.Tensor:
        if code_ids.shape[-1] != int(self.breeze_config.model["num_codebooks"]):
            raise ValueError("audio frames must contain exactly 16 codebook IDs")
        offsets = torch.arange(
            code_ids.shape[-1], dtype=code_ids.dtype, device=code_ids.device
        ) * int(self.breeze_config.model["vocab_size"])
        return self.audio_embedding(code_ids + offsets).sum(dim=-2)

    def embed_depth_code(
        self, code_ids: torch.Tensor, *, codebook_index: int
    ) -> torch.Tensor:
        return self.audio_embedding(
            code_ids + codebook_index * int(self.breeze_config.model["vocab_size"])
        )

    # -- loading -----------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> BreezeTorchModel:
        checkpoint_dir = Path(checkpoint_dir)
        config_path = checkpoint_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing {config_path}")
        index_path, kind = find_index(checkpoint_dir)

        with config_path.open("r", encoding="utf-8") as handle:
            raw_config = json.load(handle)
        quantized = bool(raw_config.get("quantization_config"))
        if quantized:
            require_torchao(checkpoint_dir)
            # The BF16 parts of a quantized checkpoint have to stay the dtype
            # the quantized parts dequantize into, or every matmul that mixes
            # them fails. So the checkpoint's own dtype wins over the setting.
            declared = resolve_dtype(
                str(raw_config.get("dtype") or raw_config.get("torch_dtype") or "bfloat16")
            )
            if declared != dtype:
                logger.info(
                    "%s is quantized and stores its unquantized weights as %s; "
                    "using that instead of the configured %s",
                    checkpoint_dir.name, declared, dtype,
                )
            dtype = declared

        # Built on the meta device so the several gigabytes of randomly
        # initialized weights PyTorch would otherwise allocate, and immediately
        # overwrite, are never allocated at all. On an 8 GB card that difference
        # is load-or-fail.
        with torch.device("meta"):
            model = cls(BreezeConfig.from_file(config_path))
        model.set_compute_dtype(dtype)
        # Before loading, not after: assign=True wraps each incoming tensor in a
        # Parameter inheriting the placeholder's requires_grad, and a quantized
        # subclass asked to require gradients raises.
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        weight_map = read_weight_map(index_path)
        by_shard: dict[str, list[str]] = defaultdict(list)
        for source_name in weight_map:
            if map_source_tensor(source_name) is not None:
                by_shard[weight_map[source_name]].append(source_name)

        state: dict[str, torch.Tensor] = {}
        prefixes = {
            "text_encoder": "text_encoder.",
            "text_encoder_proj": "text_encoder_proj.",
            "backbone": "backbone.",
            "depth_decoder": "depth_decoder.",
            "audio_embedding": "audio_embedding.",
            "lm_head": "lm_head.",
        }
        for shard_name, names in sorted(by_shard.items()):
            shard_path = checkpoint_dir / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(
                    f"Missing checkpoint shard {shard_path}. Download the complete "
                    "snapshot -- see MODELS.md."
                )
            with open_shard(shard_path, kind) as get_tensor:
                for source_name in sorted(names):
                    target = map_source_tensor(source_name)
                    assert target is not None
                    tensor = get_tensor(source_name)
                    plain = is_plain(tensor)
                    if plain and tensor.is_floating_point():
                        tensor = tensor.to(dtype)
                    prefix = prefixes[target.component]
                    if target.transform == "split_codebook_heads":
                        if not plain:
                            raise RuntimeError(
                                f"{source_name} is quantized, but it has to be "
                                "split into fifteen separate heads, which needs a "
                                "dense tensor. A checkpoint that quantizes "
                                "depth_decoder.codebooks_head is not supported."
                            )
                        if tensor.ndim != 3 or tensor.shape[0] != 15:
                            raise ValueError(
                                "depth codebook head must have shape "
                                f"[15, hidden, vocab], got {tuple(tensor.shape)}"
                            )
                        for index in range(tensor.shape[0]):
                            key = f"{prefix}{target.target_name}.{index}.weight"
                            state[key] = tensor[index].T.contiguous().to(device)
                    else:
                        state[f"{prefix}{target.target_name}"] = tensor.to(device)
                    del tensor

        # assign=True is what makes the meta construction work: parameters are
        # replaced by the loaded tensors rather than copied into storage that
        # does not exist. It is also what preserves a quantized subclass, which
        # a copy into a dense placeholder would flatten.
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        if missing or unexpected:
            # A parameter left on the meta device would fail much later, inside
            # a matmul, with an error naming neither the tensor nor the file.
            raise RuntimeError(
                "Checkpoint does not match the model.\n"
                f"  missing: {list(missing)[:8]}\n"
                f"  unexpected: {list(unexpected)[:8]}"
            )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        logger.info(
            "Loaded %s (%s, %s%s)",
            checkpoint_dir.name, kind, dtype,
            " + torchao int8" if quantized else "",
        )
        return model
