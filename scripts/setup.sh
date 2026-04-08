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

# Install core package in editable mode
echo ""
echo "Installing Arca core..."
pip install -e ".[tts]" --quiet
echo "✓ Core dependencies installed"

# Download Whisper model
echo ""
echo "Downloading Whisper base.en model (~145MB)..."
python3 -c "
from faster_whisper import WhisperModel
print('  Downloading...')
model = WhisperModel('base.en', device='auto', compute_type='auto')
print('  ✓ Model ready')
"

# Download Kokoro TTS model
echo ""
echo "Downloading Kokoro TTS model (~82MB)..."
python3 -c "
try:
    import kokoro_onnx
    print('  Running: python -m kokoro_onnx.download')
    import subprocess
    subprocess.run(['python3', '-m', 'kokoro_onnx.download'], check=True)
    print('  ✓ Kokoro TTS ready')
except ImportError:
    print('  kokoro-onnx not installed, skipping TTS setup')
except Exception as e:
    print(f'  Warning: {e}')
    print('  Run manually: python3 -m kokoro_onnx.download')
"

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
echo "Run Arca:"
echo "  arca                  # Full TUI with voice"
echo "  arca --no-voice       # Text-only mode"
echo "  arca --text 'hello'   # One-shot query"
echo ""
