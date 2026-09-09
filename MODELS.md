# The speech model

The model is not in this repository — it is several gigabytes, and it is not
ours. `download_model.py` fetches the right one for your machine:

```bash
python download_model.py          # ~3.5 GB on a Mac, ~5.9 GB on Windows
python download_model.py --check  # is it already here?
```

That is all most people need. The rest of this file is for when you want a
*different* model.

---

## Why the platforms download different files

| | macOS | Windows |
|---|---|---|
| Runtime | MLX | PyTorch + CUDA |
| Checkpoint | [`rishikksh20/Breeze-TTS-2-mlx`](https://huggingface.co/rishikksh20/Breeze-TTS-2-mlx) | [`smcleod/Breeze-TTS-2-int8`](https://huggingface.co/smcleod/Breeze-TTS-2-int8) |
| Precision | INT8, MLX affine quantization | INT8, torchao weight-only |
| Directory | `chkpt-mlx-int8/` | `chkpt-breeze-tts-2-int8/` |
| Download | ~3.5 GB | ~5.9 GB |
| Weights in memory | ~3 GB unified | ~5.2 GB VRAM |

Same model, same voices, same behaviour. What differs is the file format.

Each platform quantizes with its own toolchain, and neither can read the
other's. MLX packs scales and zero points per group of 64 in its own layout;
torchao stores per-row scales alongside int8 data in tensor subclasses that only
PyTorch reconstructs. So there are two INT8 artifacts of one model rather than
one shared file.

Both directories also contain the Qwen audio codec (`audio_tokenizer/`), which
is not quantized on either platform and is the same model in both.

### The BF16 alternative on Windows

```bash
python download_model.py --variant torch-bf16
```

[`BreezeBlue/Breeze-TTS-2`](https://huggingface.co/BreezeBlue/Breeze-TTS-2), the
original weights, ~7 GB down and ~7 GB of VRAM. Pinned to the revision the MLX
artifact was converted from, so it is the same weights the Mac runs.

Worth taking if you have the VRAM. The INT8 conversion's own measurements put
per-layer mean relative error at about 1%; that is small — changing the seed
moves the output far more — but it is not nothing, and BF16 needs no torchao and
no pickle shards. Both land in different directories, so having both is fine:
the backend prefers INT8 and falls back to whichever is present.

### What is actually quantized

Only the linear layers in the backbone and the depth decoder — the 1.8B
parameters that dominate the file. The text encoder, the embeddings, the LM
head, the codebook heads and every norm stay BF16. Our loader does not need to
know any of that: each tensor arrives already in whatever form it was saved, and
torchao intercepts the matmul for the ones that are quantized.

The INT8 checkpoint ships as pickle shards (`pytorch_model-*.bin`) rather than
safetensors, because transformers cannot currently round-trip a torchao INT8
checkpoint through safetensors. `torch.load` is called with `weights_only=True`,
which works because importing torchao registers its tensor classes as safe to
reconstruct. **torchao is required** for that checkpoint —
`requirements/windows.txt` installs it, pinned to 0.17.0 because 0.18+ needs
torch 2.11 and the CUDA wheels installed here are 2.9.

One consequence worth knowing: a pickle shard is materialized whole rather than
tensor by tensor, so loading briefly needs about 5 GB of host RAM. The BF16
safetensors path does not.

### Speed

Weight-only INT8 buys disk and memory, not necessarily speed. The conversion's
author measured it about 2.5× *slower* than BF16 on Apple's MPS, where torchao
has no fused int8 kernel and every linear dequantizes first. CUDA has that
kernel, so it should not behave the same way — but that is untested here. If
generation cannot keep up with playback, try `--variant torch-bf16`.

`test_torch_parity.py` checks that the two runtimes agree, by pushing identical
weights through both and comparing. It needs no GPU and no checkpoint.

## Using a different model

`SOURCES` at the top of `download_model.py` holds one entry per checkpoint:

```python
SOURCES = {
    "mlx":        {"repo": "rishikksh20/Breeze-TTS-2-mlx", ...},
    "torch-int8": {"repo": "smcleod/Breeze-TTS-2-int8",    ...},
    "torch-bf16": {"repo": "BreezeBlue/Breeze-TTS-2",      ...},
}
DEFAULT_VARIANT = {"mlx": "mlx", "torch": "torch-int8"}
```

Add an entry, or override without editing:

```bash
python download_model.py --repo OWNER/NAME --revision abc123 --dest ./my-model
```

`BREEZE_MODEL_REPO` and `BREEZE_MODEL_REVISION` in `.env` do the same thing.
If the directory is not the default one, point the server at it with
`BREEZE_MODEL=./my-model`.

### Finding a candidate on Hugging Face

Search for `Breeze-TTS-2` at <https://huggingface.co/models>. What you are
looking for is either a newer release of the same model, or someone's
re-quantization of it. Open the **Files** tab and check the contents against
the list below before downloading several gigabytes.

To get the direct URL for a single file — useful for checking a `config.json`
before committing to the whole thing — the pattern is:

```
https://huggingface.co/OWNER/NAME/resolve/REVISION/FILENAME
```

`REVISION` can be `main` or a commit hash. For example:

```bash
curl -L https://huggingface.co/BreezeBlue/Breeze-TTS-2/resolve/main/config.json
```

### What a replacement has to contain

**For the PyTorch backend** (Windows), a Hugging Face checkpoint in either
shard format:

```
config.json                     architectures: BreezeForConditionalGeneration
model.safetensors.index.json    ...and *.safetensors shards
   or
pytorch_model.bin.index.json    ...and pytorch_model-*.bin shards
tokenizer.json, tokenizer_config.json, special_tokens_map.json
generation_config.json
audio_tokenizer/                the FP32 Qwen codec, complete
```

The format is detected from which index file is present. A `quantization_config`
in `config.json` makes the loader require torchao and take the checkpoint's own
dtype for its unquantized weights, ignoring `BREEZE_TORCH_DTYPE` — mixing those
two would fail in the first matmul that touches both.

Tensor names have to match the original checkpoint's, because one mapping table
serves both backends. `test_cross_platform.py` builds a checkpoint in each
format and loads it back, so a change to that mapping fails a test rather than a
download.

**For the MLX backend** (macOS), a converted artifact:

```
mlx_config.json                 declares int8 or int4 and the audio dtype
config.json
text_encoder.safetensors, text_encoder_proj.safetensors,
backbone.safetensors, depth_decoder.safetensors,
audio_embedding.safetensors, lm_head.safetensors
tokenizer.json, tokenizer_config.json, generation_config.json
audio_tokenizer/
```

The config itself is validated on load — 16 codebooks, an audio vocabulary of
2051, a Qwen3 backbone, a bundled text-encoder config — so a checkpoint with a
different architecture is rejected at start-up with a message naming what is
wrong, rather than producing noise.

### Converting the original to MLX yourself

If only the original checkpoint exists for a release you want, `breeze-tts-mlx`
can quantize it:

```bash
python download_model.py --variant torch-bf16 --dest chkpt-full
cd breeze-tts-mlx
uv run python -m breeze_tts_mlx.convert ../chkpt-full ../chkpt-mlx-int8
```

`--precision int4` produces a smaller, lower-quality artifact.

---

## Disk and memory

| | Download | Loaded |
|---|---|---|
| macOS, MLX INT8 | 3.5 GB | ~3 GB unified memory |
| Windows, torchao INT8 | 5.9 GB | ~5.2 GB VRAM |
| Windows, BF16 | 7 GB | ~7 GB VRAM |

8 GB of VRAM is the practical minimum on Windows, and the INT8 checkpoint is
what makes that comfortable rather than marginal. `BREEZE_TORCH_DTYPE=float16`
applies to the BF16 checkpoint only — it is the same size, but faster on Turing
and Pascal cards, where BF16 is emulated. A quantized checkpoint ignores it.

There is no CPU mode. On the CPU this model generates several times slower than
the speech it produces, so the streaming player would underrun continuously; the
server refuses to start rather than appearing to work.
