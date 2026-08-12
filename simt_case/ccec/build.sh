#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ASCEND_HOME_PATH="${ASCEND_HOME_PATH:?ASCEND_HOME_PATH not set}"
CCEC="$ASCEND_HOME_PATH/bin/ccec"
LD_LLD="$ASCEND_HOME_PATH/bin/ld.lld"
AARCH64_GXX="$ASCEND_HOME_PATH/tools/hcc/bin/aarch64-target-linux-gnu-g++"
GXX15="${GXX15:-g++-15}"
WARP_COUNT="${SC_WARP_COUNT:-8}"

BUILD_DIR="$CASE_ROOT/build"
mkdir -p "$BUILD_DIR"

echo "=== Build AICPU .so (aarch64 cross-compile) ==="
"$AARCH64_GXX" -O2 -std=c++17 -fPIC -shared -Wall -Wextra \
    -D__gm__= -D__aicore__= -D__ubuf__= -D__cce__= \
    -I"$CASE_ROOT/common" \
    -I"$CASE_ROOT/kernel" \
    -I"$ASCEND_HOME_PATH/include" \
    -o "$BUILD_DIR/libsimt_case_aicpu.so" \
    "$CASE_ROOT/aicpu/simt_case_aicpu.cpp" \
    -L"$ASCEND_HOME_PATH/x86_64-linux/devlib/linux/aarch64" \
    -L"$ASCEND_HOME_PATH/x86_64-linux/devlib/device" \
    -lascend_hal \
    -Wl,--build-id
echo "[BUILD] AICPU .so: $BUILD_DIR/libsimt_case_aicpu.so"

# Copy reference AICore binary (generates .rodata + TLV 0x10/0x11)
REF_BIN="/data/pyptouser/xionglejin/project/simpler/tests/atomic_probe/pa_scheduler/simt_cross_core_v2/gm/build/ccec/simt_cross_core_g0_kernel.o"
if [ -f "$REF_BIN" ]; then
    cp "$REF_BIN" "$BUILD_DIR/simt_case_aicore.o"
    echo "[BUILD] Copied reference AICore binary"
else
    echo "[ERROR] Reference binary not found: $REF_BIN"
    exit 1
fi

echo "=== Build host (x86) ==="
"$GXX15" -O2 -std=c++17 -Wall -Wextra \
    -D__gm__= -D__aicore__= \
    -I"$CASE_ROOT/common" \
    -I"$ASCEND_HOME_PATH/include" \
    -I"$ASCEND_HOME_PATH/x86_64-linux/pkg_inc" \
    -I"$ASCEND_HOME_PATH/x86_64-linux/pkg_inc/runtime" \
    -I"$ASCEND_HOME_PATH/x86_64-linux/pkg_inc/runtime/runtime" \
    -I"$ASCEND_HOME_PATH/x86_64-linux/pkg_inc/profiling" \
    "$CASE_ROOT/host/simt_case_host.cpp" \
    -L"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
    -L"$ASCEND_HOME_PATH/runtime/lib64" \
    -Wl,-rpath,"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
    -lascendcl -lruntime -ldl \
    -o "$BUILD_DIR/simt_case_host"
echo "[BUILD] host: $BUILD_DIR/simt_case_host"

echo "=== Build complete ==="
echo "  AICPU .so:   $BUILD_DIR/libsimt_case_aicpu.so"
echo "  AICore ELF:  $BUILD_DIR/simt_case_aicore.o"
echo "  Host:        $BUILD_DIR/simt_case_host"
