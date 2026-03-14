#!/usr/bin/env bash
# Train and evaluate a routing checkpoint for a subset of models.
#
# Usage:
#   ./scripts/train_subset.sh "gpt-5-4-high,claude-opus-4-6-high"
#   ./scripts/train_subset.sh "nemotron-3-nano-reasoning,gpt-oss-20b-high,nemotron-3-super"
#
# Input files (must exist):
#   data/v1-9models-pool/train.pt    Aligned prefill features
#   data/v1-9models-pool/test.pt     Aligned prefill features
#   data/train_v1.csv                Training labels
#   data/test_v1.csv                 Test labels
#   configs/v1-9models.yaml          Pool config (full 9-model pool)
#
# Output files (in checkpoints/subset-<N>models/):
#   prefill_router.pt                Trained subset checkpoint
#   serve.yaml                       Serve config

set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
    echo "Usage: $0 <comma-separated-model-names>" >&2
    echo "" >&2
    echo "Available models:" >&2
    echo "  nemotron-3-nano-reasoning, gpt-oss-20b-high, nemotron-3-super," >&2
    echo "  gpt-oss-120b-high, qwen-3-5-35b, qwen-3-5-122b," >&2
    echo "  gpt-5-2-high, gpt-5-4-high, claude-opus-4-6-high" >&2
    exit 1
fi

MODELS="$1"
N_MODELS=$(echo "$MODELS" | tr ',' '\n' | wc -l | tr -d ' ')

CONFIG=configs/v1-9models.yaml
TRAIN_CSV=data/train_v1.csv
TEST_CSV=data/test_v1.csv
TRAIN_CACHE=data/v1-9models-pool/train.pt
TEST_CACHE=data/v1-9models-pool/test.pt
OUTPUT_DIR="checkpoints/subset-${N_MODELS}models"

echo "=== Subset Training ==="
echo "  Models ($N_MODELS): $MODELS"
echo "  Output: $OUTPUT_DIR/"
echo ""

echo "=== Checking inputs ==="
for f in "$CONFIG" "$TRAIN_CSV" "$TEST_CSV" "$TRAIN_CACHE" "$TEST_CACHE"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f" >&2
        exit 1
    fi
done
echo "  All inputs OK"

echo ""
echo "=== Training ==="
python -m model_router_toolkit train \
    --config "$CONFIG" \
    --data "$TRAIN_CSV" \
    --prefill-cache "$TRAIN_CACHE" \
    --output-dir "$OUTPUT_DIR" \
    --models "$MODELS" \
    --device cpu

echo ""
echo "=== Evaluating ==="
python -m model_router_toolkit evaluate \
    --config "$CONFIG" \
    --checkpoint "$OUTPUT_DIR/prefill_router.pt" \
    --data "$TEST_CSV" \
    --prefill-cache "$TEST_CACHE" \
    --models "$MODELS" \
    --device cpu

echo ""
echo "=== Done ==="
echo "  Checkpoint: $OUTPUT_DIR/prefill_router.pt"
