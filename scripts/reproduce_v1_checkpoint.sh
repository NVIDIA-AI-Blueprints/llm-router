#!/usr/bin/env bash
# Reproduce checkpoints/prefill_router.pt from prepared data.
#
# Prerequisites:
#   pip install -e '.[prefill]'
#
# Input files (must exist):
#   data/v1-9models-pool/train.pt    Aligned prefill features (12,299 questions)
#   data/v1-9models-pool/test.pt     Aligned prefill features (2,170 questions)
#   data/train_v1.csv                Training labels (9 models)
#   data/test_v1.csv                 Test labels (9 models)
#   configs/v1-9models.yaml          Pool config
#
# Output files:
#   checkpoints/prefill_router.pt    Trained routing checkpoint
#   checkpoints/serve.yaml           Serve config (informational)

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=configs/v1-9models.yaml
TRAIN_CSV=data/train_v1.csv
TEST_CSV=data/test_v1.csv
TRAIN_CACHE=data/v1-9models-pool/train.pt
TEST_CACHE=data/v1-9models-pool/test.pt
OUTPUT_DIR=checkpoints

echo "=== Checking inputs ==="
for f in "$CONFIG" "$TRAIN_CSV" "$TEST_CSV" "$TRAIN_CACHE" "$TEST_CACHE"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f" >&2
        exit 1
    fi
    echo "  OK: $f"
done

echo ""
echo "=== Training ==="
python -m model_router_toolkit train \
    --config "$CONFIG" \
    --data "$TRAIN_CSV" \
    --prefill-cache "$TRAIN_CACHE" \
    --output-dir "$OUTPUT_DIR" \
    --device cpu

echo ""
echo "=== Evaluating ==="
python -m model_router_toolkit evaluate \
    --config "$CONFIG" \
    --checkpoint "$OUTPUT_DIR/prefill_router.pt" \
    --data "$TEST_CSV" \
    --prefill-cache "$TEST_CACHE" \
    --device cpu

echo ""
echo "=== Done ==="
echo "  Checkpoint: $OUTPUT_DIR/prefill_router.pt"
echo "  Serve config: $OUTPUT_DIR/serve.yaml"
