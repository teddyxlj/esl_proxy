#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CASE_ROOT="$SCRIPT_DIR"

case "${1:-}" in
    build)
        bash "$CASE_ROOT/ccec/build.sh"
        ;;
    run)
        DEVICE="${2:-0}"
        BATCHES="${SIMT_CASE_BATCHES:-4}"
        BUILDERS="${SIMT_CASE_BUILDERS:-1}"
        RUNS="${SIMT_CASE_RUNS:-1}"
        "$CASE_ROOT/build/ccec/simt_case_host" \
            --kernel "$CASE_ROOT/build/ccec/simt_case_kernel.o" \
            --device "$DEVICE" --batches "$BATCHES" --builders "$BUILDERS" --runs "$RUNS"
        ;;
    run-task)
        BATCHES="${SIMT_CASE_BATCHES:-4}"
        BUILDERS="${SIMT_CASE_BUILDERS:-1}"
        RUNS="${SIMT_CASE_RUNS:-1}"
        task-submit --device 0 --timeout 600 --run \
            "cd $CASE_ROOT && SIMT_CASE_BATCHES=$BATCHES SIMT_CASE_BUILDERS=$BUILDERS SIMT_CASE_RUNS=$RUNS ./run.sh run \$TASK_DEVICE"
        ;;
    *)
        echo "Usage: $0 {build|run [device]|run-task}"
        echo "  Env: SIMT_CASE_WARP_COUNT=4|8  (default 8)"
        echo "       SIMT_CASE_TASKS=64         (default 64)"
        echo "       SIMT_CASE_RUNS=1           (default 1)"
        exit 1
        ;;
esac
