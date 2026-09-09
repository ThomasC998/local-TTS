# The speech model

The model is not in this repository — it is several gigabytes, and it is not
ours. `download_model.py` fetches the right one for your machine:

```bash
python download_model.py          # ~3.5 GB on a Mac, ~7 GB on Windows
python download_model.py --check  # is it already here?
```

That is all most people need. The rest of this file is for when you want a
*different* model.

---

## Why the two platforms download different files

| | macOS | Windows |
|---|---|---|
| Runtime | MLX | PyTorch + CUDA |
| Checkpoint | [`rishikksh20/Breeze-TTS-2-mlx`](https://huggingface.co/rishikksh20/Breeze-TTS-2-mlx) | [`BreezeBlue/Breeze-TTS-2`](https://huggingface.co/BreezeBlue/Breeze-TTS-2) |
| Precision | INT8, quantized for Apple Silicon | BF16, the original weights |
| Directory | `chkpt-mlx-int8/` | `chkpt-breeze-tts-2/` |
| Size | ~3.5 GB | ~7 GB |

Same model, same voices, same behaviour. What differs is the file format.

MLX's INT8 artifact packs its weights in MLX's own affine quantization layout —
scales and zero points per group of 64 values, in MLX's own arrangement — and
nothing in PyTorch reads it. So Windows loads the untouched Hugging Face
checkpoint instead. The two are pinned to the same upstream revision, so both
machines are running the same weights rather than merely the same model name.

Both directories also contain the Qwen audio codec (`audio_tokenizer/`), which
is byte-identical between them and is the same FP32 model on both platforms.

`test_torch_parity.py` checks that the two runtimes actually agree, by pushing
identical weights through both and comparing. It needs no GPU and no checkpoint.

---

## Using a different model

Two identifiers control this, at the top of `download_model.py`:

```python
SOURCES = {
    "mlx":   {"repo": "rishikksh20/Breeze-TTS-2-mlx", "revision": None,   ...},
    "torch": {"repo": "BreezeBlue/Breeze-TTS-2",      "revision": "c1c8ca…", ...},
}
```

Edit them, or override without editing:

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

**For the PyTorch backend** (Windows), the original Hugging Face layout:

```
config.json                     architectures: BreezeForConditionalGeneration
model.safetensors.index.json    the shard map
model-0000?-of-0000?.safetensors
tokenizer.json, tokenizer_config.json, special_tokens_map.json
generation_config.json
audio_tokenizer/                the FP32 Qwen codec, complete
```

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
python download_model.py --backend torch --dest chkpt-full
cd breeze-tts-mlx
uv run python -m breeze_tts_mlx.convert ../chkpt-full ../chkpt-mlx-int8
```

`--precision int4` produces a smaller, lower-quality artifact.

---

## Disk and memory

| | Disk | Loaded |
|---|---|---|
| macOS (INT8) | 3.5 GB | ~3 GB of unified memory |
| Windows (BF16) | 7 GB | ~7 GB of VRAM |

8 GB of VRAM is the practical minimum on Windows, and it will be tight with a
browser open. `BREEZE_TORCH_DTYPE=float16` in `.env` is worth trying on an older
card, where BF16 is emulated rather than native — it is the same size, but
faster on Turing and Pascal.

There is no CPU mode. On the CPU this model generates several times slower than
the speech it produces, so the streaming player would underrun continuously; the
server refuses to start rather than appearing to work.
