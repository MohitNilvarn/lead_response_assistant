#!/usr/bin/env bash
# Render build script
# Installs CPU-only PyTorch first (≈200 MB vs ≈2 GB with CUDA),
# then installs the remaining dependencies.

set -o errexit  # exit on error

echo "==> Installing CPU-only PyTorch..."
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing remaining dependencies..."
pip install --no-cache-dir -r requirements.txt
