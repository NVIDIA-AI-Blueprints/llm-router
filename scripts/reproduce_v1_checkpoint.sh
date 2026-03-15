#!/usr/bin/env bash
# Reproduce checkpoints/prefill_router.pt from prepared data.
#
# Usage:
#   ./scripts/reproduce_v1_checkpoint.sh          # full pipeline (sweep + train)
#   ./scripts/reproduce_v1_checkpoint.sh --lean    # fast mode (pre-transformed, ~30s)
#
# Prerequisites:
#   pip install -e '.[prefill]'
#
# Full mode inputs:
#   data/v1-9models-pool/train.pt    Aligned prefill features (12,299 questions, 2 GB)
#   data/v1-9models-pool/test.pt     Aligned prefill features (2,170 questions, 357 MB)
#
# Lean mode inputs:
#   data/v1-9models-lean/train_features.pt   Pre-transformed features (~69 MB)
#   data/v1-9models-lean/test_features.pt    Pre-transformed features (~20 MB)
#
# Common inputs:
#   data/train_v1.csv                Training labels (9 models)
#   data/test_v1.csv                 Test labels (9 models)
#   configs/v1-9models-qwen35b.yaml   Pool config
#
# Output files:
#   checkpoints/prefill_router_qwen35b.pt   Trained routing checkpoint

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=configs/v1-9models-qwen35b.yaml
TRAIN_CSV=data/train_v1.csv
TEST_CSV=data/test_v1.csv
OUTPUT_DIR=checkpoints

LEAN=false
if [ "${1:-}" = "--lean" ]; then
    LEAN=true
fi

if [ "$LEAN" = true ]; then
    TRAIN_FEAT=data/v1-9models-lean/train_features.pt
    TEST_FEAT=data/v1-9models-lean/test_features.pt

    echo "=== Lean mode (pre-transformed features) ==="
    for f in "$CONFIG" "$TRAIN_CSV" "$TEST_CSV" "$TRAIN_FEAT" "$TEST_FEAT"; do
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
        --features-from "$TRAIN_FEAT" \
        --output-dir "$OUTPUT_DIR" \
        --device cpu
    mv "$OUTPUT_DIR/prefill_router.pt" "$OUTPUT_DIR/prefill_router_qwen35b.pt"

    echo ""
    echo "=== Evaluating ==="
    python -m model_router_toolkit evaluate \
        --config "$CONFIG" \
        --checkpoint "$OUTPUT_DIR/prefill_router_qwen35b.pt" \
        --data "$TEST_CSV" \
        --prefill-cache "$TEST_FEAT" \
        --device cpu
else
    TRAIN_CACHE=data/v1-9models-pool/train.pt
    TEST_CACHE=data/v1-9models-pool/test.pt

    echo "=== Full mode (sweep + train) ==="
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
        --device cpu
    mv "$OUTPUT_DIR/prefill_router.pt" "$OUTPUT_DIR/prefill_router_qwen35b.pt"

    echo ""
    echo "=== Evaluating ==="
    python -m model_router_toolkit evaluate \
        --config "$CONFIG" \
        --checkpoint "$OUTPUT_DIR/prefill_router_qwen35b.pt" \
        --data "$TEST_CSV" \
        --prefill-cache "$TEST_CACHE" \
        --device cpu
fi

echo ""
echo "=== Done ==="
echo "  Checkpoint: $OUTPUT_DIR/prefill_router_qwen35b.pt"
