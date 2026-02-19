#!/bin/bash
# Run this in your Terminal to verify the HF model tagger works.
# Usage: ./test_tagger.sh   or   bash test_tagger.sh

set -e
cd "$(dirname "$0")"

if [ -x "/opt/anaconda3/bin/python" ]; then
  PY="/opt/anaconda3/bin/python"
else
  PY="python3"
fi

echo "Using: $PY"
echo ""

echo "=== 1. Offline test (no network) ==="
$PY hf_model_tagger.py --test
echo ""

echo "=== 2. List all tags ==="
$PY hf_model_tagger.py --list-tags
echo ""

echo "=== 3. Live test — rules only (--no-ai) ==="
$PY hf_model_tagger.py --no-ai "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2"
echo ""

if [ -n "$OPENAI_API_KEY" ] || [ -n "$ANTHROPIC_API_KEY" ]; then
  echo "=== 4. Live test — with AI supplementation ==="
  $PY hf_model_tagger.py "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2"
  echo ""
else
  echo "=== 4. Skipping AI test (no OPENAI_API_KEY or ANTHROPIC_API_KEY) ==="
  echo "   To test AI: export OPENAI_API_KEY=sk-... then run ./test_tagger.sh again"
  echo ""
fi

echo "Done."
