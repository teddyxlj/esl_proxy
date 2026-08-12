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
        BATCHES="${SC_BATCHES:-256}"
        BUILDERS="${SC_BUILDERS:-1}"
        RUNS="${SC_RUNS:-1}"
        "$CASE_ROOT/build/ccec/simt_case_host" \
            --kernel "$CASE_ROOT/build/ccec/simt_case_kernel.o" \
            --device "$DEVICE" --batches "$BATCHES" --builders "$BUILDERS" --runs "$RUNS"
        ;;
    run-task)
        BATCHES="${SC_BATCHES:-256}"
        BUILDERS="${SC_BUILDERS:-1}"
        RUNS="${SC_RUNS:-1}"
        task-submit --device 0 --timeout 600 --run \
            "cd $CASE_ROOT && SC_BATCHES=$BATCHES SC_BUILDERS=$BUILDERS SC_RUNS=$RUNS ./run.sh run \$TASK_DEVICE"
        ;;
    *)
        echo "Usage: $0 {build|run [device]|run-task}"
        echo "  Env: SC_WARP_COUNT=16  SC_BATCHES=256  SC_BUILDERS=1  SC_RUNS=1"
        exit 1
        ;;
esac
