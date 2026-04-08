#!/usr/bin/env bash
# Arca Setup Script
# Run this once to install all dependencies and download models.
# Usage: bash scripts/setup.sh

set -e

echo "=== Arca Setup ==="
echo ""

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required="3.11"
if [[ "$(printf '%s\n' "$required" "$python_version" | sort -V | head -n1)" != "$required" ]]; then
    echo "Error: Python 3.11+ required (found $python_version)"
    exit 1
fi
echo "✓ Python $python_version"

# Create / reuse virtualenv
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtualenv at .venv ..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtualenv created"
else
    echo "✓ Virtualenv exists at .venv"
fi

# Activate venv for this script
source "$VENV_DIR/bin/activate"

# Install core package in editable mode
echo ""
echo "Installing Arca core..."
pip install -e ".[tts]" --quiet
echo "✓ Core dependencies installed"

# Download Whisper model
echo ""
echo "Downloading Whisper base.en model (~145MB)..."
python -c "
from faster_whisper import WhisperModel
print('  Downloading...')
model = WhisperModel('base.en', device='auto', compute_type='auto')
print('  ✓ Model ready')
"

# Download Kokoro TTS model files to ~/.arca/kokoro/
echo ""
echo "Downloading Kokoro TTS model files..."
mkdir -p ~/.arca/kokoro

KOKORO_MODEL=~/.arca/kokoro/kokoro-v1.0.onnx
KOKORO_VOICES=~/.arca/kokoro/voices-v1.0.bin

if [ ! -f "$KOKORO_MODEL" ]; then
    echo "  Downloading kokoro-v1.0.onnx (~82MB)..."
    curl -L -o "$KOKORO_MODEL" \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx" \
        --progress-bar
    echo "  ✓ Model downloaded"
else
    echo "  ✓ kokoro-v1.0.onnx already present"
fi

if [ ! -f "$KOKORO_VOICES" ]; then
    echo "  Downloading voices-v1.0.bin (~11MB)..."
    curl -L -o "$KOKORO_VOICES" \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" \
        --progress-bar
    echo "  ✓ Voices downloaded"
else
    echo "  ✓ voices-v1.0.bin already present"
fi

# Create user config dir
echo ""
mkdir -p ~/.arca
if [ ! -f ~/.arca/config.toml ]; then
    cp arca.toml ~/.arca/config.toml
    echo "✓ Created ~/.arca/config.toml"
else
    echo "✓ Config exists at ~/.arca/config.toml"
fi

# Check for ANTHROPIC_API_KEY
echo ""
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  ANTHROPIC_API_KEY not set."
    echo "   Add to your ~/.zshrc or ~/.bashrc:"
    echo "   export ANTHROPIC_API_KEY=sk-ant-..."
else
    echo "✓ ANTHROPIC_API_KEY detected"
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Activate the virtualenv first, then run Arca:"
echo "  source .venv/bin/activate"
echo "  arca                  # Full TUI with voice"
echo "  arca --no-voice       # Text-only mode"
echo "  arca --text 'hello'   # One-shot query"
echo ""
