"""The decode loop: prompt in, 24 kHz audio out, a chunk at a time.

Structurally identical to ``breeze_tts_mlx.runtime``. The differences are all
mechanical -- ``mx.array`` becomes ``torch.Tensor`` on the GPU, and the prompt
collator's tensors no longer need copying between two array libraries because
they were already PyTorch.

One thing is deliberately *not* moved to the GPU: sampling. It runs on the CPU
in NumPy, on the same ``NumpySampler`` the Mac uses, so a given seed produces
the same token sequence on both platforms.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from ._shared import AudioTokenizer, NumpySampler, SamplingConfig
from .model import BreezeTorchModel, resolve_dtype


@dataclass(frozen=True)
class TorchRuntimeConfig:
    max_new_tokens: int = 1500
    max_seq_len: int = 2048
    repetition_penalty: float = 1.1
    codec_chunk_frames: int = 2
    backbone_sampling: SamplingConfig = field(default_factory=SamplingConfig)
    depth_sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be greater than zero")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be greater than zero")
        if self.codec_chunk_frames <= 0:
            raise ValueError("codec_chunk_frames must be greater than zero")
        self.backbone_sampling.validate()
        self.depth_sampling.validate()


@dataclass(frozen=True)
class TorchAudioChunk:
    audio: np.ndarray
    sample_rate: int
    codec_frames: int
    is_final: bool
    timing: dict[str, float | int | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class _BranchBatch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    batch_size: int
    guidance_scale: float


def _left_pad(tensor: torch.Tensor, width: int, value: int | bool = 0) -> torch.Tensor:
    amount = width - tensor.shape[1]
    if amount <= 0:
        return tensor
    return torch.nn.functional.pad(tensor, (amount, 0), value=value)


def _left_pad_embeds(tensor: torch.Tensor, width: int) -> torch.Tensor:
    amount = width - tensor.shape[1]
    if amount <= 0:
        return tensor
    padding = torch.zeros(
        (tensor.shape[0], amount, *tensor.shape[2:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([padding, tensor], dim=1)


def resolve_device(requested: str) -> torch.device:
    """Pick the compute device, refusing to silently land on the CPU.

    A CPU fallback is not a degraded mode for this model, it is an unusable
    one: a paragraph would take minutes, and the streaming player it feeds
    would underrun continuously. Failing at load with an actionable message
    beats a server that starts and then never keeps up.
    """
    if requested in ("", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError(
            "No CUDA device was found. The PyTorch backend needs an NVIDIA GPU; "
            "see README.md for the supported hardware and how to check the "
            "driver with `nvidia-smi`. Set BREEZE_TORCH_DEVICE=cpu only to "
            "debug loading -- generation at CPU speed cannot stream."
        )
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"{requested} was requested but PyTorch reports no CUDA device. "
            "The usual cause is a CPU-only torch wheel: reinstall with "
            "`install.ps1 -Device cuda`."
        )
    return device


class BreezeTorchRuntime:
    """One loaded model, reused for the life of the process."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
        audio_device: str | None = None,
        config: TorchRuntimeConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.artifact_dir = Path(checkpoint_dir)
        self.runtime_config = config or TorchRuntimeConfig()
        self.runtime_config.validate()
        self.torch_device = resolve_device(device)
        self.torch_dtype = resolve_dtype(dtype)
        self.model = BreezeTorchModel.from_checkpoint(
            self.artifact_dir, device=self.torch_device, dtype=self.torch_dtype
        )
        self.config = self.model.breeze_config.model
        # The prompt collator builds its tensors here before anything reaches
        # the GPU. Keeping it on the CPU matches the Mac path exactly, and the
        # tensors are a few kilobytes.
        self.device = "cpu"
        # This checkpoint uses a plain space Split pre-tokenizer, not the
        # affected Mistral regex, so the transformers warning does not apply and
        # its suggested flag crashes on a non-sequence pre-tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.artifact_dir, fix_mistral_regex=False
        )
        # The Qwen codec is a separate model and runs in FP32 exactly as on the
        # Mac. It is small, and dropping it to FP16 audibly roughens the output.
        self.audio_tokenizer = AudioTokenizer(
            self.artifact_dir / "audio_tokenizer",
            device=audio_device or str(self.torch_device),
            dtype="float32",
        )
        self.sampler = NumpySampler(seed)
        self._depth_caches: dict[int, list[Any]] = {}

    @property
    def sample_rate(self) -> int:
        return self.audio_tokenizer.sample_rate

    # -- prompt --------------------------------------------------------------
    def _encode_text_segments(self, segments: list[np.ndarray]) -> list[torch.Tensor]:
        if not segments:
            return []
        lengths = [int(segment.size) for segment in segments]
        max_length = max(lengths)
        ids = np.zeros((len(segments), max_length), dtype=np.int64)
        mask = np.zeros_like(ids)
        positions = np.zeros_like(ids)
        for row, segment in enumerate(segments):
            length = lengths[row]
            ids[row, :length] = segment
            mask[row, :length] = 1
            positions[row, :length] = np.arange(length, dtype=np.int64)
        device = self.torch_device
        hidden = self.model.text_encoder(
            torch.as_tensor(ids, device=device),
            attention_mask=torch.as_tensor(mask, device=device),
            position_ids=torch.as_tensor(positions, device=device),
        )
        projected = self.model.text_encoder_proj(hidden.to(self.model.compute_dtype))
        return [projected[row, :length] for row, length in enumerate(lengths)]

    def _merge_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text_ids_mask: torch.Tensor,
        text_ids_len: torch.Tensor,
        input_values: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids_np = input_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        text_mask_np = text_ids_mask.detach().cpu().numpy().astype(bool, copy=False)
        lengths = [
            int(value) for value in text_ids_len.detach().cpu().numpy().reshape(-1)
        ]
        length_index = 0
        segments: list[np.ndarray] = []
        row_segment_counts: list[int] = []
        for row in range(ids_np.shape[0]):
            text_ids = ids_np[row, text_mask_np[row]]
            consumed = 0
            count = 0
            while consumed < text_ids.size:
                if length_index >= len(lengths):
                    raise ValueError("text_ids_len ended before all text tokens")
                length = lengths[length_index]
                length_index += 1
                if length <= 0:
                    continue
                segments.append(text_ids[consumed : consumed + length])
                consumed += length
                count += 1
            if consumed != text_ids.size:
                raise ValueError("text segment lengths do not match text_ids_mask")
            row_segment_counts.append(count)
        if length_index != len(lengths):
            raise ValueError("unused text_ids_len values remain after segment parsing")

        encoded_segments = self._encode_text_segments(segments)
        hidden_size = int(self.config["hidden_size"])
        device = self.torch_device
        inputs_embeds = torch.zeros(
            (ids_np.shape[0], ids_np.shape[1], hidden_size),
            dtype=self.model.compute_dtype,
            device=device,
        )
        segment_index = 0
        for row, segment_count in enumerate(row_segment_counts):
            positions = np.flatnonzero(text_mask_np[row])
            if segment_count:
                row_text = torch.cat(
                    encoded_segments[segment_index : segment_index + segment_count],
                    dim=0,
                )
                inputs_embeds[row, torch.as_tensor(positions, device=device), :] = (
                    row_text.to(inputs_embeds.dtype)
                )
            segment_index += segment_count

        if input_values is not None:
            codes_np = input_values.detach().cpu().numpy().astype(np.int64, copy=False)
            codes = torch.as_tensor(codes_np, device=device)
            audio_embeds = self.model.embed_audio_frames(codes)
            audio_token_id = int(self.config["audio_token_id"])
            audio_eos_id = int(self.config["audio_eos_token_id"])
            eos_frame = torch.zeros(
                (1, int(self.config["num_codebooks"])), dtype=torch.long, device=device
            )
            eos_embed = self.model.embed_audio_frames(eos_frame)[0]
            for row in range(ids_np.shape[0]):
                audio_positions = np.flatnonzero(ids_np[row] == audio_token_id)
                if audio_positions.size != codes_np.shape[1]:
                    raise ValueError(
                        "audio placeholder count does not match encoded reference frames"
                    )
                inputs_embeds[row, torch.as_tensor(audio_positions, device=device), :] = (
                    audio_embeds[row].to(inputs_embeds.dtype)
                )
                eos_positions = np.flatnonzero(ids_np[row] == audio_eos_id)
                if eos_positions.size:
                    inputs_embeds[
                        row, torch.as_tensor(eos_positions, device=device), :
                    ] = eos_embed.to(inputs_embeds.dtype).expand(
                        eos_positions.size, hidden_size
                    )
        return inputs_embeds, attention_mask.to(device=device, dtype=torch.int32)

    def _build_branches(self, inputs: dict[str, Any]) -> _BranchBatch:
        dual_keys = [key for key in inputs if key.startswith("cfg_uncond_")]
        if dual_keys:
            raise ValueError("This backend supports no CFG or single CFG")
        guidance_scale = float(inputs.get("cfg_scale", 1.0))
        has_negative = inputs.get("cfg_negative_prompt_ids") is not None
        if guidance_scale == 0.0 and has_negative:
            embeds, mask = self._merge_inputs(
                input_ids=inputs["cfg_negative_prompt_ids"],
                attention_mask=inputs["cfg_negative_prompt_attention_mask"],
                text_ids_mask=inputs["cfg_negative_text_ids_mask"],
                text_ids_len=inputs["cfg_negative_text_ids_len"],
                input_values=inputs.get("cfg_negative_input_values"),
            )
            return _BranchBatch(embeds, mask, 1, 1.0)
        if guidance_scale == 1.0:
            embeds, mask = self._merge_inputs(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                text_ids_mask=inputs["text_ids_mask"],
                text_ids_len=inputs["text_ids_len"],
                input_values=inputs.get("input_values"),
            )
            return _BranchBatch(embeds, mask, 1, 1.0)
        if not has_negative:
            raise ValueError("cfg_scale != 1 requires a negative prompt")

        cond_ids = inputs["input_ids"]
        uncond_ids = inputs["cfg_negative_prompt_ids"]
        width = max(cond_ids.shape[1], uncond_ids.shape[1])
        cond_values = inputs.get("input_values")
        uncond_values = inputs.get("cfg_negative_input_values")
        values_compatible = (cond_values is None) == (uncond_values is None)
        if values_compatible and cond_values is not None:
            values_compatible = cond_values.shape[1:] == uncond_values.shape[1:]
        if values_compatible:
            joint_values = (
                None
                if cond_values is None
                else torch.cat([cond_values, uncond_values], dim=0)
            )
            embeds, mask = self._merge_inputs(
                input_ids=torch.cat(
                    [_left_pad(cond_ids, width), _left_pad(uncond_ids, width)], dim=0
                ),
                attention_mask=torch.cat(
                    [
                        _left_pad(inputs["attention_mask"], width),
                        _left_pad(inputs["cfg_negative_prompt_attention_mask"], width),
                    ],
                    dim=0,
                ),
                text_ids_mask=torch.cat(
                    [
                        _left_pad(inputs["text_ids_mask"], width, False),
                        _left_pad(inputs["cfg_negative_text_ids_mask"], width, False),
                    ],
                    dim=0,
                ),
                text_ids_len=torch.cat(
                    [inputs["text_ids_len"], inputs["cfg_negative_text_ids_len"]]
                ),
                input_values=joint_values,
            )
            return _BranchBatch(embeds, mask, 2, guidance_scale)

        cond_embeds, cond_mask = self._merge_inputs(
            input_ids=cond_ids,
            attention_mask=inputs["attention_mask"],
            text_ids_mask=inputs["text_ids_mask"],
            text_ids_len=inputs["text_ids_len"],
            input_values=cond_values,
        )
        uncond_embeds, uncond_mask = self._merge_inputs(
            input_ids=uncond_ids,
            attention_mask=inputs["cfg_negative_prompt_attention_mask"],
            text_ids_mask=inputs["cfg_negative_text_ids_mask"],
            text_ids_len=inputs["cfg_negative_text_ids_len"],
            input_values=uncond_values,
        )
        width = max(cond_embeds.shape[1], uncond_embeds.shape[1])
        return _BranchBatch(
            torch.cat(
                [_left_pad_embeds(cond_embeds, width), _left_pad_embeds(uncond_embeds, width)]
            ),
            torch.cat([_left_pad(cond_mask, width), _left_pad(uncond_mask, width)]),
            2,
            guidance_scale,
        )

    # -- generation ----------------------------------------------------------
    @staticmethod
    def _guided_logits(
        logits: torch.Tensor, batch_size: int, scale: float
    ) -> np.ndarray:
        # Sampling is FP32 on the CPU by design, so this is the one boundary
        # where a tensor crosses back. It is 2052 floats per step.
        values = logits.float().cpu().numpy()
        if batch_size == 1:
            return values[0]
        return values[1] + scale * (values[0] - values[1])

    def _depth_frame(
        self,
        backbone_hidden: torch.Tensor,
        first_token: int,
        *,
        branch_batch_size: int,
        guidance_scale: float,
    ) -> np.ndarray:
        device = self.torch_device
        token_batch = torch.full(
            (branch_batch_size,), first_token, dtype=torch.long, device=device
        )
        first_embedding = self.model.embed_depth_code(token_batch, codebook_index=0)
        cache = self._depth_caches.get(branch_batch_size)
        if cache is None:
            cache = self.model.depth_decoder.make_cache()
            self._depth_caches[branch_batch_size] = cache
        else:
            for layer_cache in cache:
                layer_cache.reset()
        logits = self.model.depth_decoder.begin_frame(
            backbone_hidden, first_embedding, cache
        )
        frame = [first_token]
        codebook_size = int(self.config["codec_config"]["codebook_size"])
        vocab_size = int(self.config["vocab_size"])
        num_codebooks = int(self.config["num_codebooks"])
        for predicted_index in range(1, num_codebooks):
            guided = self._guided_logits(logits, branch_batch_size, guidance_scale)
            token = self.sampler.sample(
                guided,
                self.runtime_config.depth_sampling,
                suppress_from=codebook_size,
                suppress_to=vocab_size,
            )
            frame.append(token)
            if predicted_index < num_codebooks - 1:
                token_batch = torch.full(
                    (branch_batch_size,), token, dtype=torch.long, device=device
                )
                embedding = self.model.embed_depth_code(
                    token_batch, codebook_index=predicted_index
                )
                logits = self.model.depth_decoder.step_frame(
                    embedding, codebook_index=predicted_index, cache=cache
                )
        return np.asarray(frame, dtype=np.int64)

    @torch.inference_mode()
    def iter_audio_chunks(
        self,
        inputs: dict[str, Any],
        *,
        request_id: str | None = None,
    ) -> Iterator[TorchAudioChunk]:
        branch = self._build_branches(inputs)
        if branch.inputs_embeds.shape[1] >= self.runtime_config.max_seq_len:
            raise ValueError("prompt length reaches or exceeds max_seq_len")
        request_id = request_id or f"torch-{uuid.uuid4().hex}"
        codec = self.audio_tokenizer.stream_runtime(
            self.runtime_config.codec_chunk_frames
        )
        codec.open_request(request_id, reset=True, is_first_decode=True)
        first_codec_decode = True
        frame_buffer: list[np.ndarray] = []
        total_frames = 0
        chunk_index = 0
        started = time.perf_counter()
        device = self.torch_device

        attention_mask = branch.attention_mask
        attention_np = attention_mask.cpu().numpy()
        position_np = np.cumsum(attention_np, axis=-1, dtype=np.int64) - 1
        position_np[attention_np == 0] = 1
        valid_lengths = attention_np.sum(axis=-1, dtype=np.int64)
        cache = self.model.backbone.make_cache()
        hidden = self.model.backbone(
            branch.inputs_embeds,
            attention_mask=attention_mask,
            position_ids=torch.as_tensor(position_np, device=device),
            cache=cache,
        )
        last_hidden = hidden[:, -1, :]
        logits = self.model.lm_head(last_hidden)
        token = self.sampler.sample(
            self._guided_logits(logits, branch.batch_size, branch.guidance_scale),
            self.runtime_config.backbone_sampling,
            suppress_from=int(self.config["codec_config"]["codebook_size"]),
            suppress_to=int(self.config["vocab_size"]),
        )
        token_history: list[int] = []
        eos_token = int(self.config["vocab_size"])

        def decode_buffer(*, is_final: bool) -> TorchAudioChunk:
            nonlocal first_codec_decode, frame_buffer, total_frames, chunk_index
            frames = frame_buffer
            frame_buffer = []
            decode_started = time.perf_counter()
            audio = self.audio_tokenizer.decode_chunk(
                codec, request_id, frames, reset=first_codec_decode
            )
            total_frames += len(frames)
            chunk = TorchAudioChunk(
                audio=audio,
                sample_rate=self.sample_rate,
                codec_frames=len(frames),
                is_final=is_final,
                timing={
                    "chunk_index": chunk_index,
                    "codec_frames": len(frames),
                    "total_frames": total_frames,
                    "codec_ms": (time.perf_counter() - decode_started) * 1000.0,
                    "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                },
            )
            first_codec_decode = False
            chunk_index += 1
            return chunk

        try:
            for step_index in range(self.runtime_config.max_new_tokens):
                if token == eos_token:
                    break
                frame = self._depth_frame(
                    last_hidden,
                    token,
                    branch_batch_size=branch.batch_size,
                    guidance_scale=branch.guidance_scale,
                )
                frame_buffer.append(frame)
                reached_limit = step_index == self.runtime_config.max_new_tokens - 1
                reached_cache_limit = (
                    cache[0].offset >= self.runtime_config.max_seq_len - 1
                )
                if len(frame_buffer) >= self.runtime_config.codec_chunk_frames:
                    yield decode_buffer(is_final=reached_limit or reached_cache_limit)
                if reached_limit or reached_cache_limit:
                    break

                frame_tensor = torch.as_tensor(frame, dtype=torch.long, device=device)[
                    None, None, :
                ]
                frame_embeds = self.model.embed_audio_frames(frame_tensor)
                if branch.batch_size == 2:
                    frame_embeds = frame_embeds.expand(
                        2, frame_embeds.shape[1], frame_embeds.shape[2]
                    )
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (branch.batch_size, 1),
                            dtype=attention_mask.dtype,
                            device=device,
                        ),
                    ],
                    dim=1,
                )
                positions = torch.as_tensor(valid_lengths[:, None], device=device)
                valid_lengths = valid_lengths + 1
                hidden = self.model.backbone(
                    frame_embeds,
                    attention_mask=attention_mask,
                    position_ids=positions,
                    cache=cache,
                )
                last_hidden = hidden[:, -1, :]
                logits = self.model.lm_head(last_hidden)
                token_history.append(token)
                token = self.sampler.sample(
                    self._guided_logits(
                        logits, branch.batch_size, branch.guidance_scale
                    ),
                    self.runtime_config.backbone_sampling,
                    suppress_from=int(self.config["codec_config"]["codebook_size"]),
                    suppress_to=int(self.config["vocab_size"]),
                    token_history=token_history,
                    repetition_penalty=self.runtime_config.repetition_penalty,
                )
            if frame_buffer:
                yield decode_buffer(is_final=True)
        finally:
            codec.close_request(request_id)
