# Breeze TTS 2 MLX backend

This backend targets Apple-Silicon Macs, including the M1 Pro. It runs the
untouched source checkpoint as `original`, can download a ready-to-run INT8
artifact, or creates INT8 and INT4 artifacts locally for lower memory use.
There are no separate FP16 or FP32 artifacts. It uses:

- MLX 8-bit or 4-bit affine weight-only quantization for the active text
  encoder, Qwen3 backbone, depth decoder, text/audio embeddings, and depth
  codebook heads;
- the source checkpoint's original floating-point dtypes in `chkpt-full`;
- FP16 attention/MLP compute, backbone/depth residuals, and KV caches for
  INT8/INT4;
- an FP32 text-encoder residual stream to preserve exponent range while its
  expensive quantized operations stay FP16;
- FP16 for the small text projection and backbone LM head in INT8/INT4; and
- the bundled Qwen audio tokenizer unchanged in FP32 for every model.

The converted package omits `embed_text_tokens.*` and the main checkpoint's
`codec_model.*`. Those weights are inactive in the repository's supported
CLI/API pipeline. The separate `audio_tokenizer/` is copied byte-for-byte into
INT8/INT4 artifacts.

## 1. Apple-Silicon `uv` environment

This repository's `pyproject.toml` is intentionally limited to the MLX inference
application. It resolves only for macOS on ARM64, pins Python 3.12, and does not
install this repository's CUDA, API-server, or training dependency sets. The
Qwen tokenizer's own declared transitive dependencies are retained so its FP32
codec remains supported without patching third-party package metadata.

```bash
brew install uv sox
sox --version
uv sync --locked
```

`uv` creates `.venv/` automatically. You do not need to activate it; run every
command below through `uv run`. SoX is a native executable required by the Qwen
FP32 audio tokenizer, so `uv` cannot install it as a Python dependency.

If Homebrew reports that SoX is installed but the command is not found, load
the Apple-Silicon Homebrew environment and retry:

```bash
eval "$(/opt/homebrew/bin/brew shellenv)"
sox --version
```

## 2. Download the original checkpoint

Skip this section if you only want to use the published INT8 checkpoint in
section 3. The original checkpoint is needed for original-precision inference
or local INT8/INT4 conversion.

The `chkpt/` directory in this repository contains metadata only. Download both
main safetensors shards and the complete audio tokenizer to `chkpt-full`:

```bash
uv run huggingface-cli download BreezeBlue/Breeze-TTS-2 \
  --revision c1c8ca18b70b30822735633991d9ebf4898e47d4 \
  --local-dir chkpt-full
```

`chkpt-full` is the `original` model. Inference loads its active weights
directly from the source shards and preserves every source dtype; it does not
create a floating-point conversion beside it.

## 3. Download INT8 weights or convert locally

For INT8 inference without downloading and converting the original checkpoint,
download the ready-to-run MLX artifact from
[`rishikksh20/Breeze-TTS-2-mlx`](https://huggingface.co/rishikksh20/Breeze-TTS-2-mlx):

```bash
uv run hf download rishikksh20/Breeze-TTS-2-mlx \
  --local-dir chkpt-mlx-int8
```

No conversion is required. The downloaded directory is self-contained and
includes the unchanged FP32 Qwen audio tokenizer. You can proceed directly to
[inference](#4-run-inference).

To build the artifact locally instead, first download the original checkpoint
as described above. Only INT8 and INT4 create converted artifacts. INT8 is the
default and does not require a precision flag:

```bash
uv run python -m breeze_tts_mlx.convert chkpt-full chkpt-mlx-int8 \
  --source-revision c1c8ca18b70b30822735633991d9ebf4898e47d4
```

Create an INT4 artifact separately:

```bash
uv run python -m breeze_tts_mlx.convert chkpt-full chkpt-mlx-int4 \
  --precision int4 \
  --source-revision c1c8ca18b70b30822735633991d9ebf4898e47d4
```

Each INT8/INT4 output directory is self-contained. The large eligible layers
are quantized; the roughly 13 MB projection and LM head remain FP16. The audio
tokenizer stays byte-for-byte identical to the original FP32 tokenizer.

Conversion is component-by-component to bound peak memory on a 16 GB M1 Pro.
It validates that every floating tensor in `audio_tokenizer/model.safetensors`
is FP32 before copying that directory unchanged. Existing non-empty output
directories are rejected unless `--overwrite` is explicitly supplied.

The output contains:

```text
chkpt-mlx-int8/
  mlx_config.json
  config.json
  tokenizer.json
  tokenizer_config.json
  generation_config.json
  text_encoder.safetensors
  text_encoder_proj.safetensors
  backbone.safetensors
  depth_decoder.safetensors
  audio_embedding.safetensors
  lm_head.safetensors
  audio_tokenizer/                 # original FP32 files
```

## 4. Run inference

Run the untouched original checkpoint:

```bash
uv run python infer.py chkpt-full \
  --text "Voice clone uses reference audio with its exact transcript to preserve timbre, rhythm, emotion, and style." \
  --instruction "Speak warmly and clearly." \
  --output output_original.wav
```

Or run the INT8 artifact:

```bash
uv run python infer.py chkpt-mlx-int8 \
  --text "Voice clone uses reference audio with its exact transcript to preserve timbre, rhythm, emotion, and style." \
  --instruction "Speak warmly and clearly." \
  --output output_mlx.wav
```

Voice design with single CFG:

```bash
uv run python infer.py chkpt-mlx-int8 \
  --text "Voice clone uses reference audio with its exact transcript to preserve timbre, rhythm, emotion, and style." \
  --instruction "Speak warmly and clearly." \
  --cfg-scale 4 \
  --output output_cfg_mlx.wav
```

Reference voice editing:

```bash
uv run python infer.py chkpt-mlx-int8 \
  --text "Voice clone uses reference audio with its exact transcript to preserve timbre, rhythm, emotion, and style." \
  --instruction "Keep the reference speaker and use a calm delivery." \
  --ref-audio reference.wav \
  --ref-text "The exact transcript of the reference recording." \
  --output output_reference_mlx.wav
```

`--audio-device auto` uses MPS when it is available and CPU otherwise. The FP32
audio tokenizer can run on either device for all three model choices.

During generation the CLI updates one compact line with generated codec frames,
audio duration, elapsed generation time, total loaded weight memory, and the
real-time generation speed. It is audio duration divided by generation time, so
`1.00x realtime` is real time, `0.53x realtime` is slower than real time, and
values above `1.00x` are faster than real time. No percentage or progress bar is
displayed.

## 5. Benchmark original vs INT8 vs INT4

The benchmark uses `chkpt-full` directly for `original`, converts missing
INT8/INT4 artifacts, then launches each model in a fresh process so Metal caches
and unified memory do not leak between variants:

```bash
uv run python benchmark.py \
  --source chkpt-full \
  --artifacts-dir mlx-benchmark-artifacts \
  --artifact int8=chkpt-mlx-int8 \
  --convert-missing \
  --max-new-tokens 64 \
  --warmup-runs 1 \
  --audio-output-dir mlx_benchmark_audio \
  --json-output mlx_benchmark.json
```

The terminal table and JSON report include main-weight disk size, complete
artifact size, load time, load RSS increase, MLX Metal memory, generation peak
RSS, codec frames/second, real-time speed, and the saved WAV path. The measured
run for each model is saved as `original.wav`, `int8.wav`, or `int4.wav` under
`--audio-output-dir`; warmup audio is discarded. These files use the same text,
instruction, sampling settings, and seed for subjective quality comparison.

To benchmark only quantized variants on a 16 GB machine:

```bash
uv run python benchmark.py \
  --precisions int8 int4 \
  --artifact int8=chkpt-mlx-int8 \
  --artifact int4=chkpt-mlx-int4 \
  --max-new-tokens 64
```

The original checkpoint can exceed comfortable unified-memory headroom on a
16 GB M1 Pro and may benchmark swap pressure rather than compute speed.

## 6. Developer checks

Lint and test tools are isolated in the optional `dev` group and are not
installed by the production `uv sync` command:

```bash
uv run --group dev ruff check infer.py benchmark.py breeze_tts_mlx tests
uv run --group dev python -m pytest
```

## 7. Validation before production use

This implementation needs a complete checkpoint for numerical parity and audio
quality testing. Before deployment, compare each precision's text embeddings,
backbone and depth logits, acoustic codes, speaker similarity, ASR WER/CER, and
streaming chunk boundaries with the existing PyTorch implementation. INT8 and
especially INT4 can alter sampling decisions even when tensor errors are small.
