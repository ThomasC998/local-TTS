# Breeze TTS 2: ONNX, INT8, CPU, and macOS optimization plan

This document is an implementation plan for converting the current Breeze TTS 2 inference stack into:

1. a portable, componentized ONNX Runtime CPU backend with INT8 weights; and
2. an Apple Silicon backend optimized for macOS with MLX and/or direct Core ML.

It is based on a full audit of the repository's inference, model, generation, CUDA graph, streaming codec, configuration, and test code, plus the serialized tensor headers described by [`chkpt/config.json`](chkpt/config.json) and [`chkpt/model.safetensors.index.json`](chkpt/model.safetensors.index.json). Upstream checkpoint metadata and file sizes were checked at Hugging Face revision `c1c8ca18b70b30822735633991d9ebf4898e47d4` on 2026-08-27.

The main conclusion is that a single monolithic `model.onnx` is the wrong target. Breeze is a pipeline of two different autoregressive transformers, a bidirectional text encoder, prompt/audio assembly, stochastic sampling, and a stateful neural audio codec. The correct design is several tensor-only graphs under one backend-neutral generation engine.

## Executive decision

Build and ship different artifacts for the two deployment goals:

| Target | Recommended implementation | Weight format | Why |
| --- | --- | --- | --- |
| Generic x86-64/Arm64 CPU | ONNX Runtime CPU EP | Dynamic U8S8 INT8 for transformer `MatMul` and `Gather`; calibrated S8S8 QDQ for supported codec operations; sensitive operations remain float | Portable, mature CPU kernels, explicit control over caches and memory |
| Apple Silicon, fastest research path | MLX for text encoder/backbone/depth; direct Core ML or FP16 MLX for codec | Start with 8-bit affine Linear/Embedding, then test 4-bit selected transformer weights; codec FP16 first | MLX has packed quantized Linear and Embedding kernels and uses Apple unified memory |
| Apple Silicon, native packaged app | Direct PyTorch-to-Core ML ML Program | FP16 compute; INT8 weight-only first; macOS 15 INT4 per-block/palettization experiments; stateful KV cache | Best integration with Core ML, GPU, and Neural Engine; no ONNX intermediary needed |
| macOS portability fallback | ONNX Runtime CPU EP | Same CPU INT8 artifact | Predictable fallback on Apple Silicon and Intel Macs |

Do **not** assume that an INT8 ONNX graph passed to ONNX Runtime's CoreML Execution Provider becomes an INT8 Core ML model. The CoreML EP's supported graph surface and provider partitioning can cause quantized nodes to fall back to CPU. Direct Core ML conversion is the Apple compression path.

### Expected weight-size result

The upstream weight files occupy approximately:

- main BF16 checkpoint: **6.966 GB** (6.488 GiB) of tensor payload;
- separate Qwen audio tokenizer: **0.682 GB** (0.635 GiB), stored entirely as FP32 tensors;
- combined weights: **7.649 GB** (7.124 GiB), excluding tokenizer JSON and small metadata files.

Two main-checkpoint allocations are not used by this repository's CLI/API path and can be omitted from a purpose-built inference artifact:

- `embed_text_tokens.weight`: **1.074 GB**;
- main checkpoint `codec_model.*`: **0.192 GB**.

After omitting those weights, the active main-model parameters contain about **2.850 GB at one byte per parameter**. Theoretical raw lower bounds, before scales, buffers, alignment, float exceptions, or duplicated initializers, are:

| Artifact strategy | Approximate raw weight bytes | Reduction from 7.649 GB |
| --- | ---: | ---: |
| Active main INT8 + current audio tokenizer FP32 | 3.532 GB | 53.8% |
| Active main INT8 + audio tokenizer FP16 | 3.191 GB | 58.3% |
| Active main INT8 + audio tokenizer INT8 | 3.021 GB | 60.5% |

A default MatMul-only dynamic quantization is not enough by itself. After dead-weight removal, quantizing the active ordinary dense matrices while leaving embeddings and heads BF16 produces approximately **3.251 GB for the main artifact**; with the current FP32 audio tokenizer, the combined payload is about **3.933 GB plus quantization metadata**, slightly larger than half of the original 7.649 GB. Reaching the half-size goal therefore requires at least one of:

- quantized text/audio embedding Gather paths;
- an INT8/refactored depth codebook head;
- converting the external codec to FP16/INT8; or
- a carefully validated lower-bit Apple artifact.

A realistic first production package target is **3.6–3.9 GB** with the codec kept conservatively mixed precision, or **3.1–3.5 GB** after codec quantization. Peak resident memory will be higher than file size because ONNX Runtime may prepack weights and must also hold activations, KV caches, codec state, allocator arenas, and output buffers.

These are engineering targets, not promises. Each target must pass the quality and performance gates later in this document on the actual CPU or Mac model being shipped.

## 1. What the current model actually does

The checkpoint is not a standard text LLM with a vocoder attached. Its inference sequence is:

```text
request text / instruction / optional reference WAV
                      |
                      v
tokenizer + template segment construction
                      |
         +------------+-------------+
         |                          |
         v                          v
26-layer T5Gemma2 text       Qwen audio encoder
encoder + 1152->2048 proj    (reference modes only)
         |                          |
         +------------+-------------+
                      v
host prompt assembly: projected text + reference audio-code embeddings
                      |
                      v
28-layer Qwen3 backbone prefill -> KV cache -> sample codebook 0
                      |
                      v
12-layer depth decoder, 15 autoregressive steps -> codebooks 1..15
                      |
                      v
one complete 16-codebook frame at 12.5 frames/s
              |                       |
              v                       v
stateful Qwen codec decoder       feed frame back to backbone
              |
              v
stream 24 kHz mono PCM
```

The important dimensions from the checkpoint are:

| Component | Architecture | Key dimensions |
| --- | --- | --- |
| Text encoder | T5Gemma2-compatible bidirectional encoder | 26 layers, hidden 1152, FFN 6912, 4 query heads, 1 KV head, vocabulary 262,158 |
| Text projection | Linear | 1152 to 2048 |
| Backbone | Qwen3 decoder | 28 layers, hidden 2048, FFN 6144, 16 query heads, 8 KV heads, head dimension 128 |
| Depth decoder | Breeze/Llama-like decoder | 12 layers, hidden 1024, FFN 8192, 8 query heads, 2 KV heads, head dimension 128 |
| Acoustic frame | Residual codebooks | 16 codebooks, valid codec IDs 0–2047 |
| Codec | Bundled Qwen audio tokenizer | 24 kHz output, 12.5 code frames/s |

The backbone predicts the first token in every acoustic frame. The depth decoder then predicts the remaining 15 tokens autoregressively. At 12.5 audio frames/s, steady-state generation requires roughly 12.5 backbone steps and **187.5 depth-transformer steps per second of audio**, before codec work. This is why the depth decoder's invocation and memory-bandwidth cost is as important as the larger backbone.

Classifier-free guidance (CFG) uses a conditional and unconditional branch packed as batch 2. It does not duplicate weights, but it doubles relevant activations and KV cache rows. The existing fast profile explicitly prepares both batch 1 (`cfg_scale=1`) and batch 2 (`cfg_scale=4`) shapes.

Token semantics that must remain identical across backends are:

- valid codec IDs: `0..2047`;
- reserved codec IDs that must be suppressed: `2048..2050`;
- codebook padding ID: `2050`;
- backbone-only EOS class: `2051`;
- audio frame width: 16 codebooks.

## 2. Exact serialized-weight audit

The following values were computed from the two upstream safetensors headers. The index's `metadata.total_size` is authoritative for serialized payload size. The checkpoint has 3,466,363,713 unique parameters plus 16,842,784 persistent Mimi codebook-state elements, for 3,483,206,497 serialized state elements. The 32 scalar initialization flags are FP32; the remaining main-checkpoint tensors are BF16.

### Main checkpoint by top-level component

| Prefix | Serialized state elements | Payload | Role | Recommendation |
| --- | ---: | ---: | --- | --- |
| `backbone_model` | 1,409,410,048 | 2.819 GB | 28-layer Qwen3 backbone | INT8 MatMul weights; float norms/RoPE/softmax; explicit KV |
| `text_encoder` | 999,903,232 | 2.000 GB | 26-layer T5Gemma2 text encoder | INT8 MatMul and embedding/Gather; fold projection into graph |
| `embed_text_tokens` | 536,899,584 | 1.074 GB | Legacy/fallback direct text embedding | **Do not load or export for this checkpoint** |
| `depth_decoder` | 434,280,448 | 0.869 GB | 12-layer, 15-step codebook decoder | INT8 MatMul; keep tiny cache; rescue sensitive head if needed |
| `codec_model` | 96,151,393 (79,308,609 parameters + 16,842,784 buffers) | 0.192 GB | Mimi model embedded in main checkpoint | **Do not load for current CLI/API path** |
| `lm_head` | 4,202,496 | 0.008 GB | Backbone logits 0..2051 | Keep FP32 initially; size is negligible |
| `text_encoder_proj` | 2,359,296 | 0.005 GB | 1152-to-2048 projection | Fold into text graph; keep FP16/FP32 initially |
| **Total tensor payload** | **3,483,206,497** | **6.966 GB** |  |  |

The largest compute weights inside the active components are:

| Subcomponent | Parameters | BF16 payload |
| --- | ---: | ---: |
| Backbone MLP projections | 1,056,964,608 | 2.114 GB |
| Text encoder MLP projections | 621,084,672 | 1.242 GB |
| Backbone attention projections | 352,328,704 | 0.705 GB |
| Text encoder embedding | 302,007,168 | 0.604 GB |
| Depth decoder MLP projections | 301,989,888 | 0.604 GB |
| Shared audio-code embedding | 67,207,168 | 0.134 GB |
| Text encoder attention projections | 76,690,432 | 0.153 GB |
| Depth codebook heads | 31,503,360 | 0.063 GB |
| Depth attention projections | 31,457,280 | 0.063 GB |

This distribution suggests the following optimization order:

1. omit dead weights;
2. quantize MLP projections;
3. quantize attention projections;
4. quantize the large text and audio embedding tables;
5. tune or selectively restore sensitive heads/layers;
6. optimize the codec separately.

### Separate bundled audio tokenizer

The separate `audio_tokenizer/model.safetensors` contains 170,557,441 serialized FP32 state elements and 682,229,764 data bytes. Some of these elements are quantizer/codebook statistics or buffers, so the table deliberately reports state rather than claiming they are all trainable parameters:

| Prefix | State elements | FP32 payload | Runtime use |
| --- | ---: | ---: | --- |
| `decoder.*` | 114,323,137 | 457.293 MB | Required for every synthesis request |
| `encoder.*` | 56,234,304 | 224.937 MB | Required only for voice clone/direction reference audio |

Split these into separately loadable encoder and decoder artifacts. Voice-design requests should never allocate the reference encoder. A service supporting all modes can lazy-load it on the first reference request. A voice-design-only product variant can omit the encoder from its package entirely, provided the feature restriction is explicit.

Converting the bundled codec from its current FP32 storage to FP16 alone halves its weight payload. This is an attractive first macOS step even before codec INT8 is proven safe.

## 3. Repository findings that block CPU and macOS today

### 3.1 The current CLI/API cannot run on CPU

[`breeze_infer/runtime.py`](breeze_infer/runtime.py#L22-L29) selects CUDA when available and otherwise returns `cpu`; it never selects MPS. Both [`infer.py`](infer.py#L65-L91) and [`breeze_infer/api.py`](breeze_infer/api.py#L75-L103) then always construct `FastBreezeStreamingRuntime`, even when every `fast_*` flag is false.

[`models/fast_streaming.py`](models/fast_streaming.py#L156-L195) unconditionally raises when the device is not CUDA. Consequently, the documented eager mode is not a CPU eager path in practice.

Do not fix this by merely deleting the guard. The runtime owns CUDA-specific graph classes and assumptions. Add a backend router:

```text
cuda + requested fast stages -> existing FastBreezeStreamingRuntime
cuda + eager                 -> existing/eager PyTorch backend
cpu                          -> new OrtCpuBackend
mps                          -> temporary PyTorchMpsBackend baseline
mlx                          -> new MlxBackend
coreml                       -> new CoreMlBackend
```

For a short-lived parity baseline only, the non-CUDA guard may be changed to reject non-CUDA devices only when at least one CUDA-fast stage is enabled, while forcing exportable eager/SDPA attention. That is useful to obtain CPU reference traces; it is not the final optimized backend.

### 3.2 Dtype and attention selection are CUDA-centric

The loader hard-codes BF16 in [`breeze_infer/runtime.py`](breeze_infer/runtime.py#L86-L92). The text encoder separately prefers `flash_attention_2` in [`models/breeze.py`](models/breeze.py#L970-L988). A portable policy should be:

| Backend | Baseline activation/cache dtype | Weight target |
| --- | --- | --- |
| Generic CPU correctness | FP32 | FP32 before quantization |
| Generic CPU optimized | FP32 first; test FP16/BF16 only on supported hardware | INT8 |
| Apple MLX/Core ML | FP16, with norms/softmax accumulation as required | INT8 or mixed 4/8-bit |
| CUDA existing path | BF16 | Existing BF16 |

Export wrappers must force tensor-only eager or decomposed SDPA behavior. Do not export FlashAttention imports, CUDA graphs, or backend-dependent branches.

### 3.3 The runtime loads two codec models

`BreezeForConditionalGeneration.__init__` always constructs `model.codec_model` in [`models/breeze.py`](models/breeze.py#L923-L931). The loader then also constructs `Qwen3TTSTokenizer` from the bundled `audio_tokenizer` directory.

The current CLI/API uses the bundled Qwen tokenizer for:

- reference encoding through [`breeze_infer/audio.py`](breeze_infer/audio.py#L13-L22); and
- streaming decode through [`models/fast_streaming.py`](models/fast_streaming.py#L281-L305).

The main `codec_model` is only relevant to legacy/fallback generation paths. An inference-only loader supporting the current CLI/API behavior should keep `config.codec_config` metadata but skip every `codec_model.*` tensor.

Deleting `model.codec_model` after `from_pretrained()` reduces steady-state memory but does not reduce peak loading memory. Selective loading or direct safetensors-to-artifact conversion is required.

### 3.4 A 1.074 GB embedding is bypassed

The checkpoint contains a text encoder. In [`models/breeze.py`](models/breeze.py#L1534-L1569), `_merge_input_ids_with_input_values` selects `convert_input_ids_to_embeds()` whenever `self.text_encoder` exists; the direct `embed_text_tokens` table is only the fallback `else` branch.

Create an inference-only model variant that asserts `text_encoder is not None` and has no `embed_text_tokens` parameter. Add a regression test that all supported templates and modes work with the parameter absent before permanently excluding it from packaged artifacts.

### 3.5 Shared weights are easy to duplicate during graph splitting

The audio-code embedding is tied between the backbone and depth decoder. The index serializes it as `depth_decoder.model.embed_tokens.weight`, while `_tie_weights` connects it to the backbone embedding.

Naively exporting separate prefill and decode graphs can duplicate the entire 28-layer backbone. Naively exporting backbone and depth graphs can duplicate the 134 MB BF16 audio embedding. Prevent both:

- use one physical backbone session for both prefill and decode; or
- make prefill/decode graph descriptors point to one shared external-data weight blob; and
- move the shared quantized audio embedding into one small host/native module or one shared artifact.

The host audio-embedding module performs a row Gather/dequantization for every codebook offset and either sums all 16 rows for a backbone frame or returns the row needed by a depth step. This is small work and guarantees one physical copy of the table.

### 3.6 Current tests do not cover an end-to-end non-CUDA runtime

The current tests cover templates, helper logic, CFG selection, token masks, API shape, and warmup configuration. They do not instantiate a complete CPU pipeline, exercise cache parity, or compare audio across backends. New stage and end-to-end tests are mandatory before quantization.

## 4. Why not export `BreezeForConditionalGeneration.generate()`

The full model/generation path contains export-hostile behavior:

- branches on input rank;
- Python lists, loops, splits, `.item()`, `.tolist()`, and shape-dependent decisions;
- boolean indexed writes while assembling prompt embeddings;
- Hugging Face `DynamicCache` and `StaticCache` Python objects with mutation;
- token-dependent Python generation loops and dynamic concatenation;
- `torch.multinomial` and sampling configuration objects;
- a position-specific depth head implemented as a Python list of linear calls;
- CUDA graph state and CUDA streams;
- stateful codec dataclasses/dictionaries and in-place `copy_` updates.

A monolithic trace will either fail, specialize incorrectly to one example, duplicate state, or contain expensive cache copies. Keep orchestration in a generation engine and export only stable tensor functions.

## 5. Target runtime architecture and graph contracts

Create inference-only modules. Do not add export conditionals throughout the training `PreTrainedModel` hierarchy.

Suggested source boundaries are:

```text
optimization/
  analyze_checkpoint.py
  capture_reference_traces.py
  export_onnx.py
  quantize_onnx.py
  validate_artifacts.py
  benchmark.py

breeze_infer/backends/
  base.py
  ort_cpu.py
  mlx_macos.py
  coreml_macos.py

breeze_infer/engine.py
breeze_infer/sampling.py
```

The final names may differ, but keep the responsibilities separate.

### 5.1 Host-only preprocessing and orchestration

Keep these operations out of ONNX/Core ML/MLX graphs:

- tokenizer JSON execution and template rendering;
- WAV read, downmix, and resampling;
- segment boundaries and prompt layout;
- CFG branch selection and branch packing;
- scatter of independently encoded text segments into the prompt;
- stopping conditions and maximum-length enforcement;
- invalid codec-ID masking;
- top-k, top-p, temperature, repetition penalty, and RNG ownership;
- streaming response and PCM conversion.

Use one shared sampler implementation for every backend. The sampler should accept logits and an explicit RNG state or random uniform values. This makes seed behavior testable and avoids relying on provider-specific random operators.

### 5.2 `text_encoder.onnx`

Inputs:

- `input_ids`: `int64[N, S]`;
- `attention_mask`: `int32/int64[N, S]`;
- `position_ids`: `int64[N, S]`.

Output:

- `projected_text`: float `[N, S, 2048]`.

Requirements:

- each independent logical text segment is one padded row;
- fold `text_encoder_proj` into the graph;
- use last hidden state only, because the current configuration uses `text_encoder_feature_layer_idx=-1` and linear projection;
- do not export prompt assembly or `embed_text_tokens`;
- support `N=1` and `N=2` initially, then the exact number of independent segments required by reference templates;
- use length buckets matching actual traffic. The CUDA profile's 32-token granularity and 32–512 range are a reasonable starting point, not a CPU requirement.

### 5.3 Shared quantized audio embedding

Store the tied table once. Its logical shape is `[16 * 2051, 2048]`.

Functions:

```text
embed_frame(code_ids[B,T,16]) -> frame_embedding[B,T,2048]
embed_depth_token(code_id[B], codebook_position) -> row_embedding[B,2048]
```

Use row-wise or group-wise INT8 scales. Gather only selected rows, convert those rows to the activation dtype, and sum/project. Avoid dequantizing the full table at session initialization.

### 5.4 One `backbone.onnx` session

Use one physical set of backbone weights for prompt prefill and token decode.

Inputs:

- `inputs_embeds`: float `[B, T, 2048]`, where `T` is prompt length or 1;
- `attention_mask` or an additive causal mask with an explicit contract;
- `position_ids`: `int64[B, T]`;
- explicit stacked `past_key` and `past_value`, logically `[28, B, 8, S, 128]`, or equivalent per-layer tensors;
- `cache_position`/valid-length tensors where required.

Outputs:

- final hidden states needed by the depth decoder;
- backbone logits `[B, T, 2052]` or only final-position logits;
- **new K/V deltas** `[28, B, 8, T, 128]`, not a fully concatenated cache.

The host owns a preallocated append-only KV cache. Returning `past + present` at every one-token step can copy hundreds of MiB per step and erase all INT8 gains.

If the exporter/runtime cannot express zero-length past tensors reliably, use a masked dummy past row or separate logical entry points that share the same external data. Do not create two independent copies of the weights.

Preserve two different concepts for left-padded CFG prompts:

- the physical cache slot used by both rows; and
- each row's logical RoPE position based on its number of valid tokens.

The current CUDA runtime intentionally tracks both. Collapsing them changes CFG results.

### 5.5 `depth_step.onnx` MVP and `depth_frame` optimized path

MVP inputs:

- backbone final hidden `[B, 2048]` for position zero;
- first codebook ID `[B]`;
- previous depth-token embedding or IDs;
- depth cache, logically `[12, B, 2, S, 128]` for K and V;
- codebook position/cache position.

MVP outputs:

- logits for the current residual codebook;
- new depth K/V delta.

The host calls this graph 15 times and samples after each call. This is the easiest correctness milestone, but approximately 187.5 Python-to-ORT calls per generated second can become a large overhead.

For production, use one of these approaches:

1. implement the 15-step loop in the C++ generation engine while reusing one ORT session and bound buffers;
2. export a fixed 15-step `depth_frame` function and pass explicit random uniforms so sampling is deterministic; or
3. add a narrowly scoped ORT custom operator that owns the fixed depth loop and sampler.

The depth cache resets for every audio frame and is tiny. Optimize this stage for dispatch, weight bandwidth, and fusion rather than cache memory.

Represent CFG combination without separate graph weights. For batch 2, combine conditional/unconditional logits with a small weight vector equivalent to:

```text
guided = scale * conditional + (1 - scale) * unconditional
```

### 5.6 `codec_decoder_step.onnx`

Base the functional wrapper on the semantics of `_LaneCore`, `StaticShiftKVCache`, and the current convolution-state builders, but replace every Python dataclass/dictionary and in-place mutation with tensor inputs and outputs.

Inputs should include:

- codes `[1, 16, chunk_frames]`;
- absolute decode position;
- flattened causal-convolution states;
- flattened transposed-convolution states;
- fixed-window transformer K/V state;
- first/tail state flags where needed.

Outputs should include:

- waveform samples;
- updated persistent states.

Workspace tensors that are overwritten within one call are temporaries, not part of the public state ABI.

Retain the current low-latency scheduling decision: once one complete acoustic frame exists, decode and emit it before computing the next backbone token.

### 5.7 `reference_audio_encoder.onnx`

This is optional at process startup and mandatory only for reference modes.

Inputs:

- normalized/resampled mono PCM and valid length.

Outputs:

- audio codes `[1, frames, 16]` with exactly the same orientation and trimming rules as `Qwen3TTSTokenizer.encode`.

Keep WAV I/O and resampling in the host. Export the neural encoder/quantizer only. Validate frame boundaries, final-frame behavior, and reference prompt code equality before accepting this graph.

## 6. KV-cache memory plan

INT8 weights do not automatically quantize KV caches. For the backbone at maximum length 2048:

```text
bytes = layers * K/V * batch * KV_heads * max_length * head_dim * bytes_per_element
      = 28 * 2 * B * 8 * 2048 * 128 * element_bytes
```

| Cache dtype | No CFG, B=1 | CFG, B=2 |
| --- | ---: | ---: |
| FP16/BF16 | 224 MiB | 448 MiB |
| FP32 | 448 MiB | 896 MiB |

The depth cache at 17 positions is only about 0.2 MiB FP16 for B=1 and resets per acoustic frame.

Recommendations:

- keep backbone KV at FP16 on Apple backends;
- use FP32 for the first portable ORT CPU implementation unless the target's FP16/BF16 attention kernels are verified faster;
- expose request profiles such as 512, 1024, and 2048 maximum total tokens rather than always allocating 2048;
- enforce `prompt_length + max_new_tokens <= cache_capacity` before inference;
- use past/present buffer sharing or host-owned K/V deltas to avoid allocation/copy on every step;
- do not quantize KV to INT8 until the weight-only model is stable and measured. KV quantization is a separate accuracy/performance experiment.

Because the codec frame rate is 12.5 Hz, 1500 generated frames represent up to roughly 120 seconds of output. Many applications can use a smaller request limit and save cache memory without changing weights.

## 7. Step-by-step implementation

### Phase 0 — Pin inputs and build a reference corpus

1. Create a dedicated conversion environment. Start with the repository's tested model versions and add the exporter/runtime tools; do not silently upgrade `transformers` or `qwen-tts` while establishing parity:

   ```bash
   python3.11 -m venv .venv-onnx
   source .venv-onnx/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install onnx onnxscript onnxruntime safetensors psutil
   python -m pip freeze > optimization-toolchain.lock.txt
   ```

   Once export succeeds, replace unbounded tool names with the exact verified versions. Use a separate macOS conversion environment for `coremltools` and MLX so a Core ML dependency change does not perturb the portable ONNX build.

2. Download the complete model snapshot, not only `config.json` and the index:

   ```bash
   huggingface-cli download BreezeBlue/Breeze-TTS-2 \
     --revision c1c8ca18b70b30822735633991d9ebf4898e47d4 \
     --local-dir chkpt-full
   ```

3. Verify that the directory contains:

   ```text
   model-00001-of-00002.safetensors
   model-00002-of-00002.safetensors
   model.safetensors.index.json
   tokenizer.json
   tokenizer_config.json
   generation_config.json
   audio_tokenizer/model.safetensors
   audio_tokenizer/config.json
   audio_tokenizer/preprocessor_config.json
   ```

4. Record SHA-256 hashes, the Hugging Face revision, repository commit, Python version, package lock, OS, CPU model, RAM size, and—on macOS—the chip and OS version.

5. Build a fixed validation corpus containing at least:

   - short and long English voice design;
   - short and long Chinese voice design;
   - English and Chinese reference cloning;
   - voice direction with CFG 4;
   - CFG 1 and CFG 4;
   - punctuation, numbers, vocal events, and long pauses;
   - short, medium, and long clean references;
   - prompt lengths around every planned bucket boundary;
   - seeds 0, 1, 42, and several random seeds.

6. Capture reference tensors from the current CUDA/PyTorch implementation:

   - projected text segments;
   - assembled prompt embeddings and masks;
   - prefill K/V and final logits/hidden state;
   - at least 32 forced backbone decode steps;
   - logits for all 15 depth positions across many frames;
   - generated acoustic codes;
   - codec waveform and every persistent state across chunk boundaries.

7. Benchmark the existing path. Record cold load, warm load, peak RSS/GPU memory, text encoding, reference encoding, backbone prefill, first depth frame, first codec chunk, TTFA, total synthesis time, and real-time factor.

If no CUDA machine is available, obtain and version these traces on a machine that can run the upstream code before changing model logic. Stochastic end-to-end output alone is not sufficient as a reference.

### Phase 1 — Add a backend-neutral generation engine

1. Extract template/tokenization behavior without modifying it.

2. Define protocols resembling:

   ```python
   class TextEncoderBackend:
       def encode_segments(self, input_ids, attention_mask, position_ids): ...

   class AcousticBackend:
       def prefill(self, prompt_embeddings, attention_mask, position_ids): ...
       def backbone_step(self, frame, state): ...
       def depth_frame(self, backbone_hidden, first_code, state, rng): ...

   class CodecBackend:
       def encode_reference(self, pcm, sample_rate): ...
       def decode_frames(self, codes, state): ...
   ```

3. Move all sampling to one backend-independent module. Match current order exactly:

   - mask reserved codec IDs;
   - apply repetition penalty only to generated backbone-token history;
   - temperature;
   - top-k;
   - top-p;
   - sample/greedy decision.

4. Make RNG explicit. Do not let PyTorch, ONNX, MLX, and Core ML each own unrelated sampling behavior.

5. Keep the existing CUDA runtime behind the same interface. Run its tests and compare the new engine with the old engine before adding ONNX.

6. Add end-to-end CPU-safe tests using small fake stage backends. Verify CFG branch order, EOS handling, codebook order, frame scheduling, codec tail flushing, and stream finalization.

### Phase 2 — Build an inference-only selective loader

1. Parse the safetensors index and create a manifest mapping every retained tensor to one of:

   - text encoder;
   - text projection;
   - shared audio embedding;
   - backbone;
   - backbone head;
   - depth decoder;
   - depth heads;
   - audio tokenizer encoder;
   - audio tokenizer decoder.

2. Explicitly reject or omit:

   - `embed_text_tokens.*` when the text encoder is enabled;
   - main `codec_model.*` for the current external-Qwen-codec pipeline;
   - training-only buffers and optimizer states, if any.

3. Load one component at a time on meta tensors or directly from safetensors. Do not instantiate the full model in deployment and then delete modules.

4. Preserve the audio embedding tie in the artifact manifest. Assert its checksum once and reference it from both consumers.

5. Split the Qwen audio tokenizer encoder and decoder by prefix so they can be loaded independently.

6. Add a manifest field documenting the exact unsupported path caused by omissions. For example, the inference-only artifact does not support legacy `model.generate(output_audio=True, audio_tokenizer=None)` because that path used the omitted Mimi model.

7. Run the full corpus with the selective BF16/FP32 loader and require parity before exporting.

### Phase 3 — Write tensor-only wrappers

For each graph:

1. Remove `ModelOutput` objects; return tuples of named tensors.

2. Replace Transformers cache objects with explicit tensors.

3. Replace in-place cache mutation with returned deltas or functional slice/concat operations.

4. Replace Python shape branches with separate entry points or explicit tensor control.

5. Replace `create_causal_mask` with a small tensor-only mask builder whose semantics are covered by tests.

6. Replace depth codebook Python-list heads with a stacked weight tensor and batched Gather/MatMul.

7. Force eval mode, disable dropout, disable FlashAttention/CUDA graph code, and avoid `.float()` mutations of stored weights.

8. Test every wrapper in PyTorch against the reference traces before invoking the exporter.

### Phase 4 — Export float ONNX baselines

Use the modern `torch.export`-based ONNX exporter (`torch.onnx.export(..., dynamo=True)`), which is the current recommended PyTorch path. Use external tensor data because the unquantized components exceed the 2 GB protobuf limit.

Conceptual export settings:

```python
program = torch.onnx.export(
    wrapper.eval(),
    args=example_args,
    kwargs=example_kwargs,
    dynamo=True,
    external_data=True,
    dynamic_shapes=dynamic_shapes,
    report=True,
    verify=True,
)
program.save(output_path, external_data=True)
```

Implementation rules:

1. Pin PyTorch, ONNX, ONNX Script, and ONNX Runtime versions in an optimization lock file. Exporter behavior is version-sensitive.

2. Choose the ONNX opset supported by the pinned runtime. Do not select an opset merely because it is newest.

3. Export one component at a time to keep conversion peak memory bounded.

4. Mark only necessary axes dynamic. Prefer fixed B=1/2 and bounded sequence buckets for hot decode.

5. Keep initializers out of graph inputs so constant folding and fusions remain possible.

6. Save large tensor data externally and validate the model by path, not by loading a >2 GB protobuf into the checker.

7. Run `onnx.checker.check_model(path)` and ONNX Runtime inference for every reference fixture.

8. Generate an operator histogram and list all provider assignments/fallbacks.

9. Compare one physical `backbone.onnx` used at T=prompt and T=1 against separate logical entry points. Prefer one session unless shape specialization wins enough to justify shared-weight graph descriptors.

ONNX Runtime recommends symbolic shape inference, graph optimization, and ONNX shape inference before quantization. However, its documented preprocessing optimizer cannot emit models larger than 2 GB. The unquantized text encoder and backbone exceed that limit. For them:

- attempt the Qwen3 optimizer only on a compatible, separately exported backbone graph and inspect actual fusion counts;
- otherwise skip the incompatible offline optimization pass and use standard runtime graph optimization;
- never assume a fusion occurred—count fused `Attention`, `QAttention`, `MatMul`, and fallback nodes;
- do not split layers into many runtime sessions solely to satisfy the preprocessing tool, because inter-session dispatch and cache transfer may cost more than the fusion saves.

The full custom Breeze graph is not a standard Qwen3 model. Only the isolated backbone is a candidate for the Qwen3 transformer optimizer, and even it requires verification.

### Phase 5 — Establish float ONNX parity

Test in this order:

1. text encoder last hidden and projected output;
2. shared audio embeddings;
3. backbone prefill logits, hidden state, and every K/V layer;
4. forced-token backbone decode for 32 or more steps;
5. all depth positions under forced tokens;
6. codec reference encoder codes;
7. codec decoder waveform and state over one-frame, two-frame, and irregular final chunks;
8. end-to-end greedy generation;
9. end-to-end sampled generation using host-provided random values.

Suggested initial float parity gates are:

- no NaN/Inf;
- shape, mask, and cache-position equality;
- per-layer cosine similarity near 1.0;
- top-k overlap and top-1 token agreement rather than only max absolute error;
- acoustic-code equality for greedy fixtures;
- codec waveform error measured before PCM16 clipping.

Choose numerical tolerances empirically from FP32-vs-BF16 PyTorch variation. Do not reuse one absolute tolerance for logits, normalized hidden states, K/V, and audio.

### Phase 6 — Quantize transformer graphs to INT8

For generic CPU, start with ONNX Runtime dynamic quantization:

- activation type: dynamic unsigned INT8;
- weight type: signed INT8;
- per-channel weights where supported;
- quantize constant-weight `MatMul` projections;
- include eligible `Gather` embedding paths or replace them with the explicit row-wise embedding implementation;
- keep norms, RoPE, softmax, residual arithmetic, masks, sampling, and cache tensors float.

The current ONNX Runtime dynamic quantizer defaults primarily to constant-B `MatMul`. `Gemm` is not in its IntegerOps registry. Ensure the exporter preserves `Linear` as `MatMul` plus optional `Add`, or use static QDQ for unavoidable `Gemm`. Inspect the quantized node histogram—do not infer success from the output filename.

Conceptual command code:

```python
from onnxruntime.quantization import QuantType, quantize_dynamic

quantize_dynamic(
    model_input="backbone.fp32.onnx",
    model_output="backbone.int8.onnx",
    per_channel=True,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gather"],
    use_external_data_format=True,
)
```

Confirm the exact signature against the pinned ONNX Runtime version.

Quantize in this risk order and validate after every step:

1. backbone MLPs;
2. text encoder MLPs;
3. depth MLPs;
4. backbone attention Q/K/V/O projections;
5. text encoder attention projections;
6. depth attention projections;
7. text embedding and shared audio embedding;
8. depth codebook heads;
9. text projection and backbone LM head only if useful.

Keep these float initially:

- every RMSNorm/LayerNorm parameter and computation;
- softmax and attention-score accumulation;
- RoPE sin/cos and application;
- residual adds and CFG combination;
- `lm_head` (only about 8 MB BF16);
- text projection (only about 5 MB BF16);
- final depth/codebook heads if quantization changes acoustic-token distributions;
- logits and sampler.

Use ONNX Runtime's quantization debugging tools to compare float and quantized activations. When quality fails, restore individual layers based on measured error, not broad guesses. Common rescue candidates are the first/last text layers, the last few backbone layers, depth input/output projections, and codebook heads.

Do not enable 7-bit `reduce_range` automatically. The documented saturation issue is specific to some x86 U8S8 instructions; it does not apply in the same way to VNNI or Arm. Benchmark the exact target CPU.

### Phase 7 — Quantize and optimize the codec separately

The codec is structurally different from the transformers and needs calibration.

1. First split encoder and decoder artifacts.

2. Establish FP32 ONNX parity.

3. On Apple, test FP16 before INT8. The source weights are FP32, so this already halves codec storage.

4. For CPU, collect representative calibration data:

   - real reference WAVs for the encoder;
   - real acoustic-code sequences generated by the float Breeze model for the decoder;
   - state tensors from start, middle, and tail chunks;
   - silence, voiced, unvoiced, high-energy, bilingual, and vocal-event samples.

5. Use static S8S8 QDQ with per-channel weights for supported `Conv`, `MatMul`, `Gather`, and other verified nodes.

6. Treat `ConvTranspose` carefully. It exists in ONNX Runtime's QDQ registry but not in its IntegerOps/QLinear registries. QDQ can compress weights around a float `ConvTranspose`; that does **not** prove native INT8 compute or a speedup.

7. Initially keep these float/FP16:

   - final waveform convolution;
   - normalization layers;
   - quantizer/codebook tables if code parity drops;
   - unsupported transposed-convolution kernels.

8. Measure encoder code equality and decoder audio quality independently. A small waveform error can become audible even when model-level token metrics look good.

### Phase 8 — Implement the hot loop without Python allocation

A Python prototype is appropriate for correctness, but the production CPU loop should use the ONNX Runtime C++ API or a small native extension.

Requirements:

- create sessions once and keep them resident;
- preallocate input embeddings, masks, logits, frame buffers, depth cache, and codec state;
- bind reusable outputs with `Ort::IoBinding`/OrtValues;
- append only K/V deltas into preallocated cache storage;
- never allocate/copy a full present cache per token;
- keep the 15-step depth loop native;
- centralize RNG and sampling;
- emit one codec frame before starting the next backbone step;
- share one thread pool or explicitly avoid oversubscription across sessions;
- keep service concurrency at one until memory and thread behavior are characterized.

Start with these ONNX Runtime settings, then sweep them:

```text
graph_optimization_level = ORT_ENABLE_ALL
execution_mode = ORT_SEQUENTIAL
inter_op_num_threads = 1
intra_op_num_threads = physical performance cores (then benchmark alternatives)
```

Mostly linear transformer graphs rarely benefit from parallel graph execution. Thread spinning can lower latency but consumes CPU and power; benchmark both latency and energy. On heterogeneous Apple cores, test several intra-op counts rather than assuming all logical cores are faster.

Use offline ORT-format conversion only after the ONNX artifact is correct. ORT format may reduce runtime binary/startup cost; it is not another weight-quantization scheme.

### Phase 9 — Avoid cold-load and resident-memory traps

1. Ship only optimized artifacts. Do not package the original BF16 shards beside INT8 deployment weights unless explicitly offering both modes.

2. Use external data with page-aligned offsets and deterministic tensor ordering. Place each component's layer weights adjacently to make memory mapping/page faults sequential.

3. Record both on-disk bytes and peak RSS during session creation. Some EPs prepack quantized matrices into a second layout.

4. Provide two runtime profiles:

   - **resident/fast:** text encoder, backbone, depth, and codec decoder stay loaded;
   - **memory constrained:** reference encoder is lazy; optionally release text encoder after prompt encoding for one-shot CLI use.

5. Cache text-encoder outputs for repeated normalized instruction/reference text when semantically safe.

6. Prewarm only shapes actually supported by the deployment profile. Do not replicate the CUDA warmup configuration blindly on CPU.

7. Save hardware-specific offline-optimized artifacts under hardware-specific names. Extended/layout optimizations can be execution-provider specific.

## 8. macOS implementation plan

Choose and document a minimum deployment target before converting artifacts:

| Target floor | Relevant capability |
| --- | --- |
| macOS 13 | Core ML ML Program INT8 weight-only compression and palettization support |
| macOS 14 | Core ML activation quantization/W8A8 format support; current MLX requires Apple Silicon and macOS 14+ |
| macOS 15 | Core ML state inputs for KV/cache state, INT4 per-block weights, and newer grouped palettization |
| M4-class Mac | Primary Mac target for a potentially faster W8A8 Neural Engine path |

Older hardware may still benefit from smaller stored weights, but it must not inherit M4 latency claims.

### 8.1 First establish an FP16 PyTorch MPS baseline

Add explicit MPS selection and use this only as a comparison point:

```python
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```

Use FP16 where supported, keep sensitive reductions float, and profile fallback. `PYTORCH_ENABLE_MPS_FALLBACK=1` can preserve correctness for unsupported operations, but repeated CPU/GPU fallback can erase the speed gain. PyTorch MPS is a float Metal backend, not the primary INT8 solution.

Treat TorchAO's documented Apple quantization as an experimental Arm CPU path unless measurements prove otherwise; do not describe it as native MPS INT8.

### 8.2 Recommended transformer path: MLX

MLX is the strongest ground-up Apple Silicon research path for the text encoder, backbone, and depth decoder because it provides quantized Linear and Embedding modules backed by packed quantized matrix operations and shares memory between CPU and GPU.

Implementation sequence:

1. Reimplement tensor-only T5Gemma2, Qwen3, Breeze depth, RMSNorm, RoPE, GQA, and SwiGLU modules in MLX.

2. Write a direct safetensors converter using the selective manifest. Do not instantiate PyTorch during normal MLX loading.

3. Validate FP16 MLX stage parity.

4. Quantize eligible `Linear` and `Embedding` modules:

   ```python
   import mlx.nn as nn

   nn.quantize(
       model,
       group_size=64,
       bits=8,
       mode="affine",
       class_predicate=predicate,
   )
   ```

5. Exclude norms, RoPE, softmax, text projection, LM head, and codebook heads initially.

6. Use fixed/preallocated FP16 KV caches and update slices in place from the generation engine.

7. Compile bucketed text/prefill functions and fixed decode functions with `mx.compile`. Account for first-call compilation and avoid accidental recompilation from changing shapes.

8. After 8-bit acceptance, test 4-bit affine group-64 or group-32 on large MLPs first. Keep attention and embeddings at 6/8-bit until quality is proven.

9. Compare mixed policies, for example:

   ```text
   MLP middle layers: 4-bit
   attention projections: 8-bit
   text/audio embeddings: 8-bit
   first/last layers and heads: FP16 or 8-bit
   norms/softmax/RoPE/cache: FP16/FP32 accumulation as required
   ```

MLX automatic quantization covers supported Linear and Embedding modules, not the codec's Conv/ConvTranspose stack. Keep the codec FP16 initially or implement and benchmark its kernels separately.

### 8.3 Native packaged path: direct Core ML ML Program

Convert tensor-only PyTorch wrappers directly to Core ML; Apple recommends a direct PyTorch conversion rather than using ONNX as an intermediary.

1. Target `mlprogram` with FP16 compute.

2. For macOS 12–14 compatibility, start with bounded/fixed shapes and INT8 weight-only compression where available.

3. For macOS 15+, use state inputs for backbone KV, depth KV, and codec state. Stateful prediction avoids copying large cache inputs/outputs and is specifically suitable for autoregressive transformers.

4. Use enumerated text/prefill shapes, based on measured prompt distribution. Core ML can optimize enumerated shapes at compile time; unbounded dynamic ML Program shapes are not a good target.

5. Start with 8-bit per-channel weight-only quantization. Then test:

   - INT4 per-block weights for large transformer MLPs on the Mac GPU;
   - grouped-channel palettization on macOS 15;
   - 6/8-bit palettization for sensitive layers;
   - W8A8 only with representative calibration and primarily on M4 Neural Engine targets.

6. Benchmark compute-unit configurations (`CPU_AND_GPU`, `ALL`, and other supported combinations) per component. The best setting for a transformer token step may differ from the convolutional codec.

7. Compile and cache the model ahead of normal requests. Large Core ML compilation can take substantial time.

Critical caveat: Core ML weight-only INT8/INT4 compression changes stored constants, but runtime computation remains float. Depending on hardware and compute unit, weights may be decompressed ahead of runtime or on the fly. Smaller disk size therefore does not guarantee lower steady-state memory or faster inference. Measure the actual Mac model.

Activation quantization can slow CPU/GPU because of runtime conversion. Use W8A8 only when the model is substantially assigned to a Neural Engine with appropriate INT8 compute, especially M4-class hardware.

### 8.4 Hybrid Apple backend

A practical first high-performance Mac backend is:

```text
MLX 8-bit text encoder
MLX 8-bit/selected 4-bit backbone and depth decoder
Core ML FP16 stateful codec decoder
Core ML FP16 optional reference encoder
host tokenizer/sampler/streaming
```

The boundaries exchange small text embeddings, acoustic codes, and PCM rather than large intermediate feature maps. Apple unified memory reduces transfer pressure, but provider calls and synchronization must still be measured.

### 8.5 ONNX Runtime CoreML EP is an experiment, not the default Apple plan

Official macOS ONNX Runtime wheels include the CoreML EP. If it is evaluated:

- request ML Program model format;
- cache the compiled Core ML model;
- prefer static inputs;
- enable profiling/provider assignment diagnostics;
- set CPU EP as an explicit fallback;
- test the float ONNX graph separately from the INT8 CPU graph.

The published CoreML EP ML Program operator list does not cover all quantized transformer operators such as Q/DQ, MatMulInteger/QLinear variants, QAttention, or MatMulNBits. A graph split between Core ML and CPU can be slower than either backend alone. Direct Core ML is the compression route.

### 8.6 Intel Macs

MLX and Apple Neural Engine optimizations are Apple-Silicon-specific. On Intel Macs, use the generic ONNX Runtime CPU artifact and benchmark x86 quantization kernels. Core ML float execution may still be tested, but do not use Apple Silicon performance expectations.

## 9. Quality-validation plan

### 9.1 Why waveform equality is not enough

The default model samples both backbone and depth logits. Small floating-point differences can select different early tokens, after which waveforms legitimately diverge. Validation needs two modes:

1. **teacher-forced/greedy stage parity**, which isolates numerical errors; and
2. **sampled distribution and perceptual evaluation**, which evaluates final quality.

### 9.2 Stage metrics

| Stage | Metrics |
| --- | --- |
| Text encoder | max/mean error, cosine similarity, per-token outliers |
| Backbone | per-layer hidden/KV cosine, logit KL/JS divergence, top-1 agreement, top-5 overlap |
| Depth decoder | same metrics for each of 15 positions, split by codebook index |
| Reference encoder | exact code agreement, per-codebook agreement, frame-count equality |
| Codec decoder | waveform L1/L2, SI-SDR where applicable, spectral convergence, mel error, chunk-boundary discontinuity |
| End-to-end | output duration, ASR WER/CER, speaker similarity, F0/prosody statistics, intelligibility, human A/B listening |

Suggested initial acceptance gates—not universal truths—are:

- float ONNX greedy acoustic-code agreement effectively exact on the fixed corpus;
- INT8 top-1 and top-k agreement reported per stage and codebook, with no systematic late-codebook collapse;
- no material relative WER/CER regression;
- speaker-similarity drop no larger than a predeclared small margin;
- duration/prosody distributions remain within predeclared margins;
- blinded listening shows no consistent preference for the float reference;
- no clicks at streaming chunk boundaries;
- identical EOS/reserved-token behavior and maximum-length termination.

Set exact thresholds after measuring natural variation across PyTorch devices/dtypes. Store the thresholds and corpus version in the artifact manifest.

### 9.3 Quantization rescue process

When a fully quantized component fails:

1. run ONNX Runtime activation/weight error debugging;
2. rank nodes by normalized output error and downstream token sensitivity;
3. restore the smallest set of nodes to float;
4. re-run teacher-forced stage tests;
5. re-run the entire end-to-end corpus;
6. record the final exclusion list by stable node/module name.

Do not restore an entire component to float when one output head or two layers cause the regression.

## 10. Performance benchmark plan

Benchmark on every actual target class, not only the conversion machine.

### Required measurements

- artifact bytes on disk;
- cold process peak RSS;
- resident RSS after all sessions initialize;
- weight prepacking/compilation time;
- first request TTFA;
- warm p50/p95 TTFA;
- text encoder latency by token bucket;
- optional reference encoder latency by audio duration;
- backbone prefill latency by prompt length and CFG mode;
- backbone token-step latency;
- one complete 15-token depth-frame latency;
- codec latency per frame and chunk;
- end-to-end real-time factor;
- CPU utilization, thread count, and power/energy where available;
- throughput under the supported concurrency, initially one.

Use ONNX Runtime JSON profiling to obtain node-level time and provider placement. Separate model compute from host sampling, cache copies, PCM transfer, and Python/native call overhead.

### Benchmark matrix

| Variable | Values |
| --- | --- |
| Mode | voice design, clone, direction |
| CFG | 1, 4 |
| Prompt bucket | 32, 64, 128, 256, 512 or measured equivalents |
| Output | short, 10 s, 30 s, maximum supported |
| State | cold process, cold session, warm |
| Precision | float baseline, INT8, selected mixed INT4/INT8 on Mac |
| Threads | 1, performance-core count, all physical cores, selected alternatives |
| Mac compute | CPU, GPU, Neural Engine/ALL where supported |

Report machine-identifying details with every result. “Mac” is not one target: M1, M2, M3, and M4 have different bandwidth and available INT8 paths.

## 11. Packaging and manifest

One possible CPU package is:

```text
breeze-tts-2-ort-int8/
  manifest.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  generation_config.json
  text_encoder.onnx
  text_encoder.weights.int8.data
  shared_audio_embedding.int8
  shared_audio_embedding.scales
  backbone.onnx
  backbone.weights.int8.data
  depth_step.onnx
  depth.weights.int8.data
  codec_decoder.onnx
  codec_decoder.weights.data
  codec_encoder.onnx                 # optional/lazy
  codec_encoder.weights.data         # optional/lazy
  LICENSE
  MODEL_LICENSE
```

`manifest.json` should include:

- schema version;
- source model ID and exact revision;
- repository commit;
- converter/package versions;
- ONNX opset and IR versions;
- quantization scheme per module/node group;
- float exclusion list;
- graph and weight SHA-256 hashes;
- supported language/features/modes;
- sample rate, frame rate, codebook count, token IDs;
- supported CFG modes and batches;
- min/max/bucket shapes;
- activation and cache dtypes;
- validation corpus version and results;
- benchmark hardware and results;
- required runtime/provider version;
- license and derivative-model notice.

The model license governs derivative/quantized checkpoints and is research/non-commercial unless separate authorization is obtained. Ship both `MODEL_LICENSE` and source-license notices with converted artifacts.

## 12. Milestones and exit criteria

### Milestone A — Correct backend separation

- existing CUDA behavior unchanged;
- non-CUDA backend is not routed through CUDA graph classes;
- central sampler matches current greedy/seeded fixtures;
- fake-backend end-to-end tests pass.

### Milestone B — Selective float loader

- `embed_text_tokens` and main `codec_model` are absent from the deployment model;
- all supported CLI/API modes pass reference tests;
- peak loading memory no longer includes omitted weights;
- audio embedding exists once.

### Milestone C — Float ONNX CPU

- every component passes stage parity;
- cache deltas and positions match across prefill/decode;
- greedy end-to-end codes and codec output meet float gates;
- one physical copy of each heavyweight component is resident.

### Milestone D — Transformer INT8

- actual quantized MatMul/Gather node counts are recorded;
- on-disk combined weights are below half the original or a documented codec milestone remains;
- INT8 is faster than float on target CPUs;
- quality gates pass.

### Milestone E — Codec optimization

- encoder/decoder are independently loadable;
- decoder steady state is faster or materially smaller with no audible regression;
- reference encoder code behavior passes gates;
- ConvTranspose fallback/precision is documented.

### Milestone F — Apple backend

- FP16 MPS, MLX, direct Core ML, and ORT CPU baselines are measured on target Macs;
- chosen backend beats ORT CPU or offers a clearly documented packaging/power benefit;
- model compilation is cached;
- provider fallback is absent or quantified;
- 8-bit passes before 4-bit is considered production.

## 13. Approaches to avoid

- Do not export the complete `generate()` path as one trace.
- Do not remove the CUDA-only guard and call that a CPU port.
- Do not load BF16 weights and quantize them at every process start.
- Do not ship original BF16 shards alongside optimized artifacts by default.
- Do not duplicate the backbone between prefill and decode sessions.
- Do not duplicate the tied audio embedding between graphs.
- Do not return/copy the entire present KV cache at each token.
- Do not quantize norms, softmax, RoPE, or sampling just to increase an “INT8 percentage.”
- Do not assume `.onnx` alone reduces weight size.
- Do not assume Core ML weight-only compression means INT8 arithmetic.
- Do not assume an ORT INT8 graph will stay on the CoreML EP.
- Do not use `PYTORCH_ENABLE_MPS_FALLBACK=1` without profiling provider transitions.
- Do not compare sampled waveforms sample-for-sample as the only quality test.
- Do not use random synthetic codec calibration codes as the only calibration set.
- Do not move to 4-bit before the 8-bit component baselines and validation harness are reliable.
- Do not use GGUF/llama.cpp as a shortcut unless custom kernels are implemented for the two-level acoustic generation and stateful codec; this is not a standard LLM architecture.

## 14. Recommended order of work

The shortest path to a useful result is:

1. create reference traces and benchmark fixtures;
2. add a provider-neutral generation engine and central sampler;
3. implement the inference-only selective loader and realize the 1.27 GB dead-weight saving;
4. export and validate text encoder, shared embedding, backbone, depth step, and codec graphs in float;
5. implement explicit host-owned K/V caches;
6. dynamically quantize transformer MatMul/Gather weights to INT8;
7. move the 15-step depth loop into native code;
8. split and optimize the separate codec, using calibration;
9. tune ONNX Runtime sessions and packaging on target CPUs;
10. port transformer components to MLX for Apple Silicon;
11. convert the codec—or the complete tensor-only pipeline—to stateful Core ML for the selected macOS floor;
12. evaluate mixed 4/8-bit Apple weights only after the INT8 path passes.

This order produces measurable memory savings early, preserves the existing CUDA backend, and avoids tying correctness work to the most difficult codec and Apple deployment decisions.

## 15. Primary technical references

### Breeze checkpoint

- [BreezeBlue/Breeze-TTS-2 model](https://huggingface.co/BreezeBlue/Breeze-TTS-2)
- [Pinned model tree at revision c1c8ca18](https://huggingface.co/BreezeBlue/Breeze-TTS-2/tree/c1c8ca18b70b30822735633991d9ebf4898e47d4)

### PyTorch and ONNX

- [PyTorch `torch.onnx` exporter](https://docs.pytorch.org/docs/stable/onnx)
- [ONNX external tensor data](https://onnx.ai/onnx/repo-docs/ExternalData.html)

### ONNX Runtime

- [Quantize ONNX models](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [Transformer graph optimization](https://onnxruntime.ai/docs/performance/transformers-optimization.html)
- [Graph optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
- [Past/present shared KV buffer](https://onnxruntime.ai/docs/genai/howto/past-present-share-buffer.html)
- [I/O binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)
- [Thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
- [Profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)
- [CoreML Execution Provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [Current quantizer operator registry](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/registry.py)

### Apple Core ML

- [Direct PyTorch conversion](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html)
- [ML Program conversion and FP16 behavior](https://apple.github.io/coremltools/docs-guides/source/convert-to-ml-program.html)
- [Optimization overview and hardware guidance](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html)
- [Compression feature availability by OS](https://apple.github.io/coremltools/docs-guides/source/opt-whats-new.html)
- [Core ML weight quantization API](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.quantization.html)
- [Stateful models and transformer KV cache](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)
- [Flexible/enumerated inputs](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)

### MLX and MPS

- [MLX quantization operation and supported formats](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
- [MLX model quantization](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.quantize.html)
- [MLX unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
- [MLX compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
