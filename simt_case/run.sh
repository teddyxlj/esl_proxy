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
        RUNS="${SC_RUNS:-1}"
        WARP="${SC_WARP_COUNT:-8}"
        DISP="${SIMPLER_DISPATCHER_SO:-}"
        if [ -z "$DISP" ]; then
            # try to find dispatcher SO
            for p in \
                /data/pyptouser/xionglejin/project/simpler/build/lib/aarch64/dispatcher/libsimpler_aicpu_dispatcher.so \
                /data/pyptouser/xionglejin/project/simpler/build/lib/x86_64/dispatcher/libsimpler_aicpu_dispatcher.so; do
                if [ -f "$p" ]; then DISP="$p"; break; fi
            done
        fi
        if [ -z "$DISP" ]; then
            echo "ERROR: set SIMPLER_DISPATCHER_SO to dispatcher .so path"
            exit 1
        fi
        SC_WARP_COUNT=$WARP "$CASE_ROOT/build/simt_case_host" \
            --device "$DEVICE" \
            --aicpu "$CASE_ROOT/build/libsimt_case_aicpu.so" \
            --aicore "$CASE_ROOT/build/simt_case_aicore.o" \
            --dispatcher "$DISP" \
            --runs "$RUNS"
        ;;
    run-task)
        RUNS="${SC_RUNS:-1}"
        WARP="${SC_WARP_COUNT:-8}"
        DISP="${SIMPLER_DISPATCHER_SO:-}"
        if [ -z "$DISP" ]; then
            for p in \
                /data/pyptouser/xionglejin/project/simpler/build/lib/aarch64/dispatcher/libsimpler_aicpu_dispatcher.so \
                /data/pyptouser/xionglejin/project/simpler/build/lib/x86_64/dispatcher/libsimpler_aicpu_dispatcher.so; do
                if [ -f "$p" ]; then DISP="$p"; break; fi
            done
        fi
        if [ -z "$DISP" ]; then
            echo "ERROR: set SIMPLER_DISPATCHER_SO"
            exit 1
        fi
        task-submit --device 0 --timeout 600 --run \
            "cd $CASE_ROOT && SC_RUNS=$RUNS SC_WARP_COUNT=$WARP SIMPLER_DISPATCHER_SO=$DISP ./run.sh run \$TASK_DEVICE"
        ;;
    *)
        echo "Usage: $0 {build|run [device]|run-task}"
        echo "  Env: SC_WARP_COUNT=8  SC_RUNS=1  SIMPLER_DISPATCHER_SO=<path>"
        exit 1
        ;;
esac
