#!/usr/bin/env bash
#
# Set up Breeze TTS on an Apple-Silicon Mac: Python packages, SoX, the model.
#
#     ./install.sh                 # everything
#     ./install.sh --skip-model    # packages only
#     ./install.sh --no-venv       # install into the current Python
#
# Nothing needs sudo, and nothing is installed outside this folder except SoX,
# which goes through Homebrew and can be declined.
#
# The Windows counterpart is install.ps1. The two do the same job with each
# system's own tools; they are not a shared script with branches, because
# almost every line would have been inside a branch.

set -euo pipefail
cd "$(dirname "$0")"
PROJECT="$PWD"

SKIP_MODEL=0
USE_VENV=1
for argument in "$@"; do
  case "$argument" in
    --skip-model) SKIP_MODEL=1 ;;
    --no-venv) USE_VENV=0 ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $argument" >&2; exit 1 ;;
  esac
done

step() { printf '\n== %s\n' "$1"; }
ok()   { printf '  \033[32m[ok]\033[0m   %s\n' "$1"; }
warn() { printf '  \033[33m[!]\033[0m    %s\n' "$1"; }
bad()  { printf '  \033[31m[x]\033[0m    %s\n' "$1"; }

echo
echo "Breeze TTS 2 - macOS setup"
echo "=========================="

# ---------------------------------------------------------------------------
# Hardware
#
# MLX is Apple Silicon only, and there is no fallback on this platform: an
# Intel Mac has no MLX and no CUDA, so there is nothing for the engine to run
# on. Better to say that here than to fail after a 3.5 GB download.
# ---------------------------------------------------------------------------
step "Hardware"
if [ "$(uname -s)" != "Darwin" ]; then
  bad "This is the macOS installer. On Windows use install.ps1."
  exit 1
fi
if [ "$(uname -m)" != "arm64" ]; then
  bad "Apple Silicon is required: MLX has no Intel build, and this Mac has no CUDA."
  exit 1
fi
ok "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"

# ---------------------------------------------------------------------------
# Python
#
# 3.13 is rejected rather than warned about: the audio codec has no wheel for
# it, and the failure is a compiler error deep in a pip log.
# ---------------------------------------------------------------------------
step "Python"
PYTHON="$(command -v python3 || true)"
if [ -z "$PYTHON" ]; then
  bad "python3 is not on PATH. Install 3.12: brew install python@3.12"
  exit 1
fi
VERSION="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$VERSION" in
  3.10|3.11|3.12) ok "Python $VERSION" ;;
  *) bad "Python $VERSION found, but this project needs 3.10-3.12."
     echo "         Try: brew install python@3.12"
     exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Virtual environment
# ---------------------------------------------------------------------------
if [ "$USE_VENV" = 1 ]; then
  step "Virtual environment"
  if [ ! -d .venv ]; then
    "$PYTHON" -m venv .venv
    ok "Created .venv"
  else
    ok ".venv already exists"
  fi
  PYTHON="$PROJECT/.venv/bin/python"
else
  warn "Installing into the current Python, as asked"
fi

"$PYTHON" -m pip install --quiet --upgrade pip setuptools wheel
ok "pip is up to date"

# ---------------------------------------------------------------------------
# Packages
# ---------------------------------------------------------------------------
step "Python packages"
# One requirements file for both platforms. The lines that differ carry an
# environment marker, so pip skips torchao here and MLX on a PC.
"$PYTHON" -m pip install -r requirements.txt
ok "Installed"

if "$PYTHON" -c 'import mlx.core' 2>/dev/null; then
  ok "MLX is working"
else
  bad "MLX did not import. Speech cannot run without it."
  exit 1
fi

# ---------------------------------------------------------------------------
# SoX
#
# A native executable the Qwen audio codec shells out to. pip cannot supply it,
# and without it the codec fails with an error naming a temporary file rather
# than the missing program.
# ---------------------------------------------------------------------------
step "SoX"
if command -v sox >/dev/null 2>&1; then
  ok "SoX is on PATH"
elif command -v brew >/dev/null 2>&1; then
  warn "SoX is not installed. The audio codec needs it."
  printf '  Install it now with Homebrew? [Y/n] '
  read -r answer </dev/tty || answer=n
  case "$answer" in
    ""|y|Y) brew install sox && ok "Installed SoX" ;;
    *) warn "Skipped. Install it later with: brew install sox" ;;
  esac
else
  bad "SoX is missing and Homebrew is not installed."
  echo "         Install Homebrew from brew.sh, then: brew install sox"
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
step "Configuration"
if [ ! -f .env ]; then
  cp .env.example .env
  ok "Created .env from the example"
  warn "Put an API key in it before the language-model pass will work:"
  echo "         GEMINI_API_KEY     from https://aistudio.google.com/apikey"
  echo "         OPENROUTER_API_KEY from https://openrouter.ai/keys"
  echo "         Or, if you already use gcloud: BREEZE_LLM_PROVIDER=vertex"
  echo "         Speech itself works without any of them."
else
  ok ".env already exists, leaving it alone"
fi

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
step "Speech model"
if [ "$SKIP_MODEL" = 1 ]; then
  warn "Skipped. Download it later with: python download_model.py"
elif ! "$PYTHON" download_model.py; then
  bad "The model download did not finish. Run it again -- it resumes:"
  echo "         python download_model.py"
fi

# ---------------------------------------------------------------------------
echo
printf '\033[32mDone\033[0m\n'
echo
echo "Start the server:"
[ "$USE_VENV" = 1 ] && echo "    source .venv/bin/activate"
echo "    python breeze_server.py"
echo
echo "Then open http://127.0.0.1:7860 and record or design a voice."
echo
echo "For the global hotkeys -- copy anything, press a key, hear it read:"
echo "    python install_hotkeys.py --startup"
echo
