#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SKILL_SRC="$PROJECT_DIR/skills/model-router-toolkit"
SKILL_NAME="model-router-toolkit"

usage() {
    echo "Usage: $0 [--extras EXTRAS] [--skip-skill] [--cursor]"
    echo ""
    echo "Install the Model Router Toolkit and its AI assistant skill."
    echo ""
    echo "Options:"
    echo "  --extras EXTRAS  pip extras to install (default: prefill)"
    echo "                   Examples: prefill, 'dev,prefill', 'dev,prefill,proxy'"
    echo "  --skip-skill     Skip copying the skill to ~/.claude/skills/"
    echo "  --cursor         Copy skill to ~/.cursor/skills/ instead of ~/.claude/skills/"
    echo "  -h, --help       Show this help message"
}

EXTRAS="prefill"
SKIP_SKILL=false
SKILL_TARGET="$HOME/.claude/skills"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --extras) EXTRAS="$2"; shift 2 ;;
        --skip-skill) SKIP_SKILL=true; shift ;;
        --cursor) SKILL_TARGET="$HOME/.cursor/skills"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

echo "==> Installing model-router-toolkit with extras: [$EXTRAS]"
pip install -e "$PROJECT_DIR/.[$EXTRAS]"

if [ "$SKIP_SKILL" = false ]; then
    if [ ! -d "$SKILL_SRC" ]; then
        echo "Warning: Skill directory not found at $SKILL_SRC — skipping skill install."
    else
        echo "==> Installing AI skill to $SKILL_TARGET/$SKILL_NAME/"
        mkdir -p "$SKILL_TARGET/$SKILL_NAME"
        cp -r "$SKILL_SRC/"* "$SKILL_TARGET/$SKILL_NAME/"
        echo "    Skill installed. Restart your AI assistant to pick it up."
    fi
fi

echo ""
echo "==> Done. Run 'model-router --help' to get started."
