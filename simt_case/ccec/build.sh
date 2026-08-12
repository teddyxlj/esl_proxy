#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:?ASCEND_HOME_PATH not set}"
CCEC="$ASCEND_HOME_PATH/bin/ccec"
LD_LLD="$ASCEND_HOME_PATH/bin/ld.lld"
WARP_COUNT="${SC_WARP_COUNT:-16}"
BUILD_DIR="$CASE_ROOT/build/ccec"
KERNEL_SOURCE="$CASE_ROOT/kernel/g0_full_pa_kernel.cpp"
AIC_OBJECT="$BUILD_DIR/simt_case_aic.o"
AIV_OBJECT="$BUILD_DIR/simt_case_aiv.o"
KERNEL_ELF="$BUILD_DIR/simt_case_kernel.o"
mkdir -p "$BUILD_DIR"

COMMON_FLAGS=(
    -c -O3 -g -x cce -Wall -std=c++17
    --cce-aicore-only
    -DSIMT_CROSS_CORE_G0_BUILDER_WARP_COUNT=$WARP_COUNT
    -Wno-logical-op-parentheses -Wno-unused-but-set-variable -Wno-bitwise-op-parentheses
    -Wno-unused-local-typedef -Wno-missing-braces -Wno-unused-variable -Wno-unused-function
    -Wno-unneeded-internal-declaration
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=false
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -mllvm -cce-aicore-dcci-before-kernel-end=false
    -I"$ASCEND_HOME_PATH/x86_64-linux/include"
    -I"$ASCEND_HOME_PATH/x86_64-linux/asc"
    -I"$ASCEND_HOME_PATH/x86_64-linux/asc/include"
    -I"$CASE_ROOT/common"
    -I"$CASE_ROOT/kernel"
)

echo "[BUILD] CCEC AIC (dav-c310-cube)"
"$CCEC" "${COMMON_FLAGS[@]}" --cce-aicore-arch=dav-c310-cube -o "$AIC_OBJECT" "$KERNEL_SOURCE" 2>&1 | grep "error:" | head -5 || true

echo "[BUILD] CCEC AIV (dav-c310-vec, SIMT)"
"$CCEC" "${COMMON_FLAGS[@]}" --cce-aicore-arch=dav-c310-vec -o "$AIV_OBJECT" "$KERNEL_SOURCE" 2>&1 | grep "error:" | head -5 || true

echo "[BUILD] Link mixed ELF"
"$LD_LLD" -m aicorelinux -Ttext=0 -static -o "$KERNEL_ELF" "$AIC_OBJECT" "$AIV_OBJECT"

echo "[BUILD] complete (warp=$WARP_COUNT)"
echo "[BUILD] kernel: $KERNEL_ELF"

# Verify .rodata and TLV 0x10
echo "---sections---"
readelf --sections --wide "$KERNEL_ELF" 2>/dev/null | grep -E "rodata|ascend"
echo "---globals---"
readelf --symbols --wide "$KERNEL_ELF" 2>/dev/null | awk '$5 == "GLOBAL" && $4 == "FUNC"'

# Build host (use reference host source)
GXX15="${GXX15:-g++-15}"
HOST_BINARY="$BUILD_DIR/simt_case_host"
HOST_SOURCE="$CASE_ROOT/host/simt_case_host.cpp"

echo "[BUILD] GCC-15 ACL host"
"$GXX15" -O2 -std=c++17 -Wall -Wextra -Wno-deprecated-declarations \
    -DSIMT_CROSS_CORE_G0_BUILDER_WARP_COUNT=$WARP_COUNT \
    -I"$CASE_ROOT/common" \
    -I"$ASCEND_HOME_PATH/include" \
    -I"$ASCEND_HOME_PATH/pkg_inc" \
    -I"$ASCEND_HOME_PATH/pkg_inc/runtime" \
    -I"$ASCEND_HOME_PATH/pkg_inc/runtime/runtime" \
    "$HOST_SOURCE" \
    -L"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
    -Wl,-rpath,"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
    -lascendcl -lruntime -ldl \
    -o "$HOST_BINARY"

echo "[BUILD] host: $HOST_BINARY"
