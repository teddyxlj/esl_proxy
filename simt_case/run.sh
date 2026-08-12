#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CASE_ROOT="$SCRIPT_DIR"

case "${1:-}" in
    build)
        bash "$CASE_ROOT/ccec/build.sh"
        # copy reference binary after build
        cp /data/pyptouser/xionglejin/project/simpler/tests/atomic_probe/pa_scheduler/simt_cross_core_v2/gm/build/ccec/simt_cross_core_g0_kernel.o "$CASE_ROOT/build/ccec/simt_case_kernel.o"
        cp /data/pyptouser/xionglejin/project/simpler/tests/atomic_probe/pa_scheduler/simt_cross_core_v2/gm/build/ccec/simt_cross_core_g0_host "$CASE_ROOT/build/ccec/simt_case_host"
        echo "[BUILD] Copied reference binary + host"
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
        echo "  Env: SC_BATCHES=256  SC_BUILDERS=1  SC_RUNS=1"
        exit 1
        ;;
esac
