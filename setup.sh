#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

echo "[1/6] Creating virtual environment..."
python3 -m venv venv

echo "[2/6] Activating virtual environment..."
source venv/bin/activate

echo "[3/6] Installing Python packages..."
pip install --upgrade pip
pip install torch transformers tqdm numpy nltk textstat

echo "[4/6] Running NLTK data check and download..."
python3 -c "
import nltk
required_packages = ['punkt', 'averaged_perceptron_tagger']
for package in required_packages:
    try:
        nltk.data.find(f'tokenizers/{package}')
        print(f\"[NLTK] '{package}' already installed.\")
    except LookupError:
        print(f\"[NLTK] Downloading '{package}'...\")
        nltk.download(package, quiet=True)
"

echo "[5/6] Creating model directories..."
mkdir -p models/distilgpt2
mkdir -p models/desklib-detector

echo "[6/6] Downloading model files..."

# Files for distilgpt2
DISTILGPT2_URL_BASE="https://huggingface.co/distilbert/distilgpt2/resolve/main"
DISTILGPT2_FILES=(
  "config.json"
  "merges.txt"
  "model.safetensors"
  "tokenizer.json"
  "vocab.json"
)

for file in "${DISTILGPT2_FILES[@]}"; do
  echo "Downloading $file for distilgpt2..."
  curl -L -o "models/distilgpt2/$file" "$DISTILGPT2_URL_BASE/$file?download=true"
done

# Files for desklib-detector
DESKLIB_URL_BASE="https://huggingface.co/desklib/ai-text-detector-v1.01/resolve/main"
DESKLIB_FILES=(
  "added_tokens.json"
  "config.json"
  "model.safetensors"
  "special_tokens_map.json"
  "spm.model"
  "tokenizer_config.json"
  "tokenizer.json"
)

for file in "${DESKLIB_FILES[@]}"; do
  echo "Downloading $file for desklib-detector..."
  curl -L -o "models/desklib-detector/$file" "$DESKLIB_URL_BASE/$file?download=true"
done

echo "✅ Setup complete!"
