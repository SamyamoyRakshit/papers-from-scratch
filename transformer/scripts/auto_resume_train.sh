#!/usr/bin/env bash
# Auto-resume training across MPS OOM crashes.
# Each restart frees macOS/MPS cache pressure that accumulates over a process's
# lifetime. Loops: train → crash → grab latest run's last.pt → resume → repeat.
#
# Usage:
#   bash transformer/scripts/auto_resume_train.sh <config> [target_epoch] [initial_resume]
#
# Example:
#   bash transformer/scripts/auto_resume_train.sh transformer/configs/base.yaml 10
#   bash transformer/scripts/auto_resume_train.sh transformer/configs/base.yaml 10 \
#       transformer/checkpoints/base/run_2026-05-25_13-17-17/last.pt

set -u  # error on unset vars (no -e: we want to handle training failures)

CONFIG="${1:?Usage: $0 <config> [target_epoch] [initial_resume]}"
TARGET_EPOCH="${2:-10}"
INITIAL_RESUME="${3:-}"

# Pull checkpoint_dir from the config — same parse strategy train.py uses.
CKPT_DIR=$(grep "checkpoint_dir:" "$CONFIG" | sed -E 's/.*checkpoint_dir:[[:space:]]*"?([^"]+)"?.*/\1/' | tr -d '[:space:]')
if [[ -z "$CKPT_DIR" ]]; then
    echo "Could not extract checkpoint_dir from $CONFIG" >&2
    exit 1
fi
echo "[auto-resume] checkpoint_dir = $CKPT_DIR, target_epoch = $TARGET_EPOCH"

RESUME_ARG=""
if [[ -n "$INITIAL_RESUME" ]]; then
    RESUME_ARG="--resume $INITIAL_RESUME"
fi

# Marker file — only consider runs created AFTER the script started.
# Without this, ls -t finds last.pt from OLDER unrelated training runs
# (e.g. previous config with different max_rows/num_epochs) and the
# epoch check exits prematurely. Using a touched file is more reliable
# than $SECONDS / date math across cross-platform `find -newer`.
SESSION_MARKER=$(mktemp -t auto_resume_session)
trap "rm -f $SESSION_MARKER" EXIT

MAX_ATTEMPTS=20
for attempt in $(seq 1 $MAX_ATTEMPTS); do
    echo ""
    echo "============================================================"
    echo "[auto-resume] Attempt $attempt/$MAX_ATTEMPTS"
    echo "[auto-resume] Command: caffeinate -s uv run python -m transformer.scripts.train --config $CONFIG $RESUME_ARG"
    echo "============================================================"

    caffeinate -s uv run python -m transformer.scripts.train --config "$CONFIG" $RESUME_ARG
    EXIT_CODE=$?

    # Find the most-recent last.pt across runs created in THIS session only.
    # -newer "$SESSION_MARKER" filters out unrelated previous runs whose last.pt
    # would otherwise be picked up by a blind `ls -t`.
    # LATEST_LAST=$(ls -t "$CKPT_DIR"/run_*/last.pt 2>/dev/null | head -1)
    LATEST_LAST=$(find "$CKPT_DIR" -name last.pt -newer "$SESSION_MARKER" 2>/dev/null \
        | xargs -I {} stat -f "%m %N" {} 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-)
    if [[ -z "$LATEST_LAST" ]]; then
        echo "[auto-resume] No last.pt found — nothing to resume from. Exiting."
        exit 1
    fi

    # Extract epoch from the checkpoint via a tiny python one-liner.
    CURRENT_EPOCH=$(uv run python -c "
import torch
ckpt = torch.load('$LATEST_LAST', map_location='cpu', weights_only=False)
print(ckpt.get('epoch', 0))
" 2>/dev/null)

    echo "[auto-resume] Latest last.pt: $LATEST_LAST (epoch $CURRENT_EPOCH)"

    if [[ "$CURRENT_EPOCH" -ge "$TARGET_EPOCH" ]]; then
        echo "[auto-resume] Reached target epoch $TARGET_EPOCH. Done."
        exit 0
    fi

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "[auto-resume] Training exited cleanly but epoch $CURRENT_EPOCH < target $TARGET_EPOCH. Continuing."
    else
        echo "[auto-resume] Training crashed (exit $EXIT_CODE). Resuming from epoch $CURRENT_EPOCH."
    fi

    RESUME_ARG="--resume $LATEST_LAST"
done

echo "[auto-resume] Hit MAX_ATTEMPTS=$MAX_ATTEMPTS without finishing. Last epoch reached: $CURRENT_EPOCH"
exit 1
