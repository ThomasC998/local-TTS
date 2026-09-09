from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
from transformers import AutoTokenizer

from .audio_codec import AudioTokenizer
from .model import BreezeMLXModel
from .sampling import NumpySampler, SamplingConfig


@dataclass(frozen=True)
class MLXRuntimeConfig:
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
class MLXAudioChunk:
    audio: np.ndarray
    sample_rate: int
    codec_frames: int
    is_final: bool
    timing: dict[str, float | int | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class _BranchBatch:
    inputs_embeds: mx.array
    attention_mask: mx.array
    batch_size: int
    guidance_scale: float


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _torch_left_pad(
    tensor: torch.Tensor, width: int, value: int | bool
) -> torch.Tensor:
    amount = width - tensor.shape[1]
    if amount <= 0:
        return tensor
    return torch.nn.functional.pad(tensor, (amount, 0), value=value)


def _mlx_left_pad(tensor: mx.array, width: int) -> mx.array:
    amount = width - tensor.shape[1]
    if amount <= 0:
        return tensor
    padding = mx.zeros((tensor.shape[0], amount, *tensor.shape[2:]), dtype=tensor.dtype)
    return mx.concatenate([padding, tensor], axis=1)


class BreezeMLXRuntime:
    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        audio_device: str = "auto",
        config: MLXRuntimeConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.runtime_config = config or MLXRuntimeConfig()
        self.runtime_config.validate()
        self.model = BreezeMLXModel.from_artifact(self.artifact_dir)
        self.config = self.model.breeze_config.model
        # The existing template collator creates Torch tensors on this device.
        # Main-model tensors are copied to MLX only after collation.
        self.device = "cpu"
        # This checkpoint uses a plain space Split pre-tokenizer, not the
        # affected Mistral regex. Transformers 4.57 otherwise warns, while its
        # suggested True flag crashes because Split is not a sequence.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.artifact_dir, fix_mistral_regex=False
        )
        self.audio_tokenizer = AudioTokenizer(
            self.artifact_dir / "audio_tokenizer",
            device=audio_device,
            dtype="float32",
        )
        self.sampler = NumpySampler(seed)
        self._depth_caches: dict[int, list[Any]] = {}

    @property
    def sample_rate(self) -> int:
        return self.audio_tokenizer.sample_rate

    def _encode_text_segments(self, segments: list[np.ndarray]) -> list[mx.array]:
        if not segments:
            return []
        lengths = [int(segment.size) for segment in segments]
        max_length = max(lengths)
        ids = np.zeros((len(segments), max_length), dtype=np.int32)
        mask = np.zeros_like(ids)
        positions = np.zeros_like(ids)
        for row, segment in enumerate(segments):
            length = lengths[row]
            ids[row, :length] = segment
            mask[row, :length] = 1
            positions[row, :length] = np.arange(length, dtype=np.int32)
        hidden = self.model.text_encoder(
            mx.array(ids),
            attention_mask=mx.array(mask),
            position_ids=mx.array(positions),
        )
        projected = self.model.text_encoder_proj(
            hidden.astype(self.model.compute_dtype)
        )
        return [projected[row, :length] for row, length in enumerate(lengths)]

    def _merge_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text_ids_mask: torch.Tensor,
        text_ids_len: torch.Tensor,
        input_values: torch.Tensor | None,
    ) -> tuple[mx.array, mx.array]:
        ids_np = _to_numpy(input_ids).astype(np.int32, copy=False)
        text_mask_np = _to_numpy(text_ids_mask).astype(bool, copy=False)
        lengths = [int(value) for value in _to_numpy(text_ids_len).reshape(-1)]
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
        inputs_embeds = mx.zeros(
            (ids_np.shape[0], ids_np.shape[1], hidden_size),
            dtype=self.model.compute_dtype,
        )
        segment_index = 0
        for row, segment_count in enumerate(row_segment_counts):
            positions = np.flatnonzero(text_mask_np[row])
            if segment_count:
                row_text = mx.concatenate(
                    encoded_segments[segment_index : segment_index + segment_count],
                    axis=0,
                )
                inputs_embeds[row, mx.array(positions, dtype=mx.int32), :] = row_text
            segment_index += segment_count

        if input_values is not None:
            codes_np = _to_numpy(input_values).astype(np.int32, copy=False)
            codes = mx.array(codes_np)
            audio_embeds = self.model.embed_audio_frames(codes)
            audio_token_id = int(self.config["audio_token_id"])
            audio_eos_id = int(self.config["audio_eos_token_id"])
            eos_frame = mx.zeros((1, int(self.config["num_codebooks"])), dtype=mx.int32)
            eos_embed = self.model.embed_audio_frames(eos_frame)[0]
            for row in range(ids_np.shape[0]):
                audio_positions = np.flatnonzero(ids_np[row] == audio_token_id)
                if audio_positions.size != codes_np.shape[1]:
                    raise ValueError(
                        "audio placeholder count does not match encoded reference frames"
                    )
                inputs_embeds[row, mx.array(audio_positions, dtype=mx.int32), :] = (
                    audio_embeds[row]
                )
                eos_positions = np.flatnonzero(ids_np[row] == audio_eos_id)
                if eos_positions.size:
                    inputs_embeds[row, mx.array(eos_positions, dtype=mx.int32), :] = (
                        mx.broadcast_to(eos_embed, (eos_positions.size, hidden_size))
                    )
        return inputs_embeds, mx.array(_to_numpy(attention_mask).astype(np.int32))

    def _build_branches(self, inputs: dict[str, Any]) -> _BranchBatch:
        dual_keys = [key for key in inputs if key.startswith("cfg_uncond_")]
        if dual_keys:
            raise ValueError("The MLX CLI currently supports no CFG or single CFG")
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
                    [
                        _torch_left_pad(cond_ids, width, 0),
                        _torch_left_pad(uncond_ids, width, 0),
                    ],
                    dim=0,
                ),
                attention_mask=torch.cat(
                    [
                        _torch_left_pad(inputs["attention_mask"], width, 0),
                        _torch_left_pad(
                            inputs["cfg_negative_prompt_attention_mask"], width, 0
                        ),
                    ],
                    dim=0,
                ),
                text_ids_mask=torch.cat(
                    [
                        _torch_left_pad(inputs["text_ids_mask"], width, False),
                        _torch_left_pad(
                            inputs["cfg_negative_text_ids_mask"], width, False
                        ),
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
            mx.concatenate(
                [_mlx_left_pad(cond_embeds, width), _mlx_left_pad(uncond_embeds, width)]
            ),
            mx.concatenate(
                [_mlx_left_pad(cond_mask, width), _mlx_left_pad(uncond_mask, width)]
            ),
            2,
            guidance_scale,
        )

    @staticmethod
    def _guided_logits(logits: mx.array, batch_size: int, scale: float) -> np.ndarray:
        # NumPy has no BF16 buffer format. Sampling is intentionally FP32 on
        # the CPU, so cast only this small logits tensor at the boundary.
        values = np.asarray(logits.astype(mx.float32))
        if batch_size == 1:
            return values[0]
        return values[1] + scale * (values[0] - values[1])

    def _depth_frame(
        self,
        backbone_hidden: mx.array,
        first_token: int,
        *,
        branch_batch_size: int,
        guidance_scale: float,
    ) -> np.ndarray:
        token_batch = mx.full((branch_batch_size,), first_token, dtype=mx.int32)
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
                token_batch = mx.full((branch_batch_size,), token, dtype=mx.int32)
                embedding = self.model.embed_depth_code(
                    token_batch, codebook_index=predicted_index
                )
                logits = self.model.depth_decoder.step_frame(
                    embedding,
                    codebook_index=predicted_index,
                    cache=cache,
                )
        return np.asarray(frame, dtype=np.int64)

    def iter_audio_chunks(
        self,
        inputs: dict[str, Any],
        *,
        request_id: str | None = None,
    ) -> Iterator[MLXAudioChunk]:
        branch = self._build_branches(inputs)
        if branch.inputs_embeds.shape[1] >= self.runtime_config.max_seq_len:
            raise ValueError("prompt length reaches or exceeds max_seq_len")
        request_id = request_id or f"mlx-{uuid.uuid4().hex}"
        codec = self.audio_tokenizer.stream_runtime(
            self.runtime_config.codec_chunk_frames
        )
        codec.open_request(request_id, reset=True, is_first_decode=True)
        first_codec_decode = True
        frame_buffer: list[np.ndarray] = []
        total_frames = 0
        chunk_index = 0
        started = time.perf_counter()

        attention_mask = branch.attention_mask
        attention_np = np.asarray(attention_mask)
        position_np = np.cumsum(attention_np, axis=-1, dtype=np.int32) - 1
        position_np[attention_np == 0] = 1
        valid_lengths = attention_np.sum(axis=-1, dtype=np.int32)
        cache = self.model.backbone.make_cache()
        hidden = self.model.backbone(
            branch.inputs_embeds,
            attention_mask=attention_mask,
            position_ids=mx.array(position_np),
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

        def decode_buffer(*, is_final: bool) -> MLXAudioChunk:
            nonlocal first_codec_decode, frame_buffer, total_frames, chunk_index
            frames = frame_buffer
            frame_buffer = []
            decode_started = time.perf_counter()
            audio = self.audio_tokenizer.decode_chunk(
                codec,
                request_id,
                frames,
                reset=first_codec_decode,
            )
            total_frames += len(frames)
            chunk = MLXAudioChunk(
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

                frame_mx = mx.array(frame, dtype=mx.int32)[None, None, :]
                frame_embeds = self.model.embed_audio_frames(frame_mx)
                if branch.batch_size == 2:
                    frame_embeds = mx.broadcast_to(
                        frame_embeds,
                        (2, frame_embeds.shape[1], frame_embeds.shape[2]),
                    )
                attention_mask = mx.concatenate(
                    [
                        attention_mask,
                        mx.ones((branch.batch_size, 1), dtype=attention_mask.dtype),
                    ],
                    axis=1,
                )
                positions = mx.array(valid_lengths[:, None])
                valid_lengths += 1
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
