<#
.SYNOPSIS
  Set up Breeze TTS on Windows 11: Python packages, SoX, and the speech model.

.DESCRIPTION
  Run this once from the project directory:

      powershell -ExecutionPolicy Bypass -File install.ps1

  It creates a virtual environment, installs PyTorch built for your GPU,
  installs everything else, checks for SoX, and downloads the model. Nothing
  needs administrator rights and nothing is installed outside this folder --
  except SoX, which is offered through winget and can be declined.

  This project needs an NVIDIA GPU. That is not a preference: on the CPU this
  model generates far slower than real time, so the audio it feeds to the
  speaker would underrun continuously. The script says so and stops rather than
  installing something that cannot work.

.PARAMETER Cuda
  Which CUDA wheel to install: cu124 (default), cu121, or cu128. Only change
  this if your driver is too old for the default -- `nvidia-smi` prints the
  highest CUDA version it supports in its top-right corner.

.PARAMETER SkipModel
  Install the packages but do not download the model (about 7 GB).

.PARAMETER NoVenv
  Install into the current Python instead of creating .venv. Use this inside an
  existing conda environment.
#>
[CmdletBinding()]
param(
    [ValidateSet('cu121', 'cu124', 'cu128')]
    [string]$Cuda = 'cu124',
    [switch]$SkipModel,
    [switch]$NoVenv
)

$ErrorActionPreference = 'Stop'
# PowerShell 7.4 turned "a native command wrote to stderr" into a terminating
# error by default. pip and winget both write ordinary progress there, so
# leaving that on would abort this script on a warning. Errors from those tools
# are checked explicitly below, via $LASTEXITCODE and by asking the result.
if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$Project = $PSScriptRoot
Set-Location $Project

function Say([string]$Message) { Write-Host $Message }
function Step([string]$Message) { Write-Host "`n== $Message" -ForegroundColor Cyan }
function Ok([string]$Message) { Write-Host "  [ok]   $Message" -ForegroundColor Green }
function Warn([string]$Message) { Write-Host "  [!]    $Message" -ForegroundColor Yellow }
function Fail([string]$Message) { Write-Host "  [x]    $Message" -ForegroundColor Red }

Say ""
Say "Breeze TTS 2 - Windows setup"
Say "==========================="

# ---------------------------------------------------------------------------
# Python
#
# 3.12 is what the project is developed against. 3.13 is rejected rather than
# warned about: qwen-tts has no wheel for it yet, and the failure that produces
# is a compiler error thirty lines into a pip log.
# ---------------------------------------------------------------------------
Step "Python"
$python = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $python) {
    Fail "Python is not on PATH. Install 3.12 from python.org or the Microsoft Store,"
    Say  "         ticking 'Add python.exe to PATH', then run this again."
    exit 1
}
$versionText = (& python -c "import sys; print('%d.%d' % sys.version_info[:2])")
$version = [version]$versionText
if ($version -lt [version]'3.10' -or $version -ge [version]'3.13') {
    Fail "Python $versionText found, but this project needs 3.10-3.12."
    Say  "         3.13 has no wheel for the audio codec yet."
    exit 1
}
Ok "Python $versionText"

# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
Step "Graphics card"
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
    Fail "nvidia-smi was not found, so there is no usable NVIDIA driver here."
    Say  ""
    Say  "  This project needs an NVIDIA GPU. AMD and Intel cards, and the CPU,"
    Say  "  are not supported: the model generates slower than it is spoken on"
    Say  "  them, so the audio would stutter continuously rather than merely"
    Say  "  being slow to start."
    Say  ""
    Say  "  If you do have an NVIDIA card, install its driver from nvidia.com"
    Say  "  and run this again."
    exit 1
}
$gpu = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) | Select-Object -First 1
Ok "$gpu"
$memoryMiB = [int](($gpu -split ',')[1] -replace '[^\d]', '')
if ($memoryMiB -lt 8000) {
    Warn "Under 8 GB of VRAM. The INT8 checkpoint installed below needs about"
    Say  "         5.2 GB of it, so this may still work with other GPU"
    Say  "         applications closed -- but it will be tight."
}

# ---------------------------------------------------------------------------
# Virtual environment
# ---------------------------------------------------------------------------
if (-not $NoVenv) {
    Step "Virtual environment"
    if (-not (Test-Path ".venv")) {
        & python -m venv .venv
        Ok "Created .venv"
    } else {
        Ok ".venv already exists"
    }
    $python = Join-Path $Project ".venv\Scripts\python.exe"
} else {
    $python = "python"
    Warn "Installing into the current Python, as asked"
}

& $python -m pip install --quiet --upgrade pip setuptools wheel
Ok "pip is up to date"

# ---------------------------------------------------------------------------
# PyTorch
#
# From NVIDIA's index, not PyPI. A plain `pip install torch` on Windows gets a
# CPU-only wheel, and the only symptom is the server refusing to start later
# with "no CUDA device is visible to PyTorch" -- on a machine that plainly has
# one. This is the single most common way this install goes wrong.
# ---------------------------------------------------------------------------
Step "PyTorch for $Cuda"
& $python -m pip install torch torchaudio --index-url "https://download.pytorch.org/whl/$Cuda"
$cudaOk = (& $python -c "import torch; print(torch.cuda.is_available())")
if ($cudaOk.Trim() -ne "True") {
    Fail "PyTorch installed, but it cannot see the GPU."
    Say  "         Try another CUDA build: install.ps1 -Cuda cu121  (or cu128)."
    Say  '         nvidia-smi prints the highest CUDA version your driver supports.'
    exit 1
}
$deviceName = (& $python -c "import torch; print(torch.cuda.get_device_name(0))")
Ok "PyTorch sees $($deviceName.Trim())"

# ---------------------------------------------------------------------------
# Everything else
# ---------------------------------------------------------------------------
Step "Python packages"
# One requirements file for both platforms. The lines that differ carry an
# environment marker, so pip skips MLX here and torchao on a Mac. torch is
# listed in it and is already satisfied by the CUDA wheel installed above.
& $python -m pip install -r (Join-Path $Project "requirements.txt")
Ok "Installed"

# Checked again, deliberately. Any package that declares torch as a dependency
# can pull the CPU wheel from PyPI over the CUDA one installed above, and the
# only symptom is the server refusing to start much later. Better to find out
# here, while it is obvious what happened.
$cudaStillOk = (& $python -c "import torch; print(torch.cuda.is_available())")
if ($cudaStillOk.Trim() -ne "True") {
    Fail "Installing the requirements replaced PyTorch with a CPU-only build."
    Say  "         Put the CUDA one back with:"
    Say  "         pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/$Cuda"
    exit 1
}
$torchao = (& $python -c "import torchao; print(torchao.__version__)" 2>&1)
if ($LASTEXITCODE -eq 0) {
    Ok "torchao $($torchao.Trim()) -- the INT8 checkpoint is readable"
} else {
    Warn "torchao did not import. The INT8 checkpoint needs it; the BF16 one does not."
    Say  "         python download_model.py --variant torch-bf16"
}

# ---------------------------------------------------------------------------
# SoX
#
# A native executable the Qwen audio codec shells out to. pip cannot supply it,
# and without it the codec fails with an error naming a temporary file rather
# than the missing program -- so it is worth installing here and saying so.
# ---------------------------------------------------------------------------
Step "SoX"
if (Get-Command sox -ErrorAction SilentlyContinue) {
    Ok "SoX is on PATH"
} else {
    Warn "SoX is not on PATH. The audio codec needs it."
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        $answer = Read-Host "  Install it now with winget? [Y/n]"
        if ($answer -eq '' -or $answer -match '^[Yy]') {
            & winget install --id ChrisBagwell.SoX --accept-source-agreements --accept-package-agreements
            Warn "Open a NEW terminal afterwards so PATH picks it up, then run: sox --version"
        }
    } else {
        Say "         Download it from https://sourceforge.net/projects/sox/ and add"
        Say "         its folder to PATH."
    }
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
Step "Configuration"
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Ok "Created .env from the example"
    Warn "Put an API key in it before the language-model pass will work:"
    Say  "         GEMINI_API_KEY     from https://aistudio.google.com/apikey"
    Say  "         OPENROUTER_API_KEY from https://openrouter.ai/keys"
    Say  "         Speech itself works without one."
} else {
    Ok ".env already exists, leaving it alone"
}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
if (-not $SkipModel) {
    Step "Speech model"
    & $python (Join-Path $Project "download_model.py")
    if ($LASTEXITCODE -ne 0) {
        Fail "The model download did not finish. Run it again -- it resumes:"
        Say  "         python download_model.py"
    }
} else {
    Step "Speech model"
    Warn "Skipped. Download it later with: python download_model.py"
}

# ---------------------------------------------------------------------------
Say ""
Write-Host "Done" -ForegroundColor Green
Say ""
Say "Start the server:"
if (-not $NoVenv) { Say "    .venv\Scripts\Activate.ps1" }
Say "    python breeze_server.py"
Say ""
Say "Then open http://127.0.0.1:7860. One voice ships with the repository,"
Say "so it will speak straight away; design or clone your own when you want to."
Say ""
Say "For the global hotkeys -- copy anything, press a key, hear it read:"
Say "    python install_hotkeys.py --startup"
Say ""
