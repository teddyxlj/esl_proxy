// Based on cann-examples/aicpu-kernel-launch/host/launch_hello.cpp
// Modified for: AICPU alloc (FullPaState) + AICore executor (reference binary)

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <runtime/rt.h>
#include <runtime/runtime/rts/rts_kernel.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include "../common/g0_full_pa.h"

using namespace pa_scheduler::simt_cross_core::g0;

/* ── ELF build-id fingerprint ── */
namespace fp {
bool ElfBuildId(const char *data, size_t len, uint64_t *out) {
    if (len < 64 || std::memcmp(data, "\x7fELF", 4) != 0 || data[4] != 2) return false;
    uint64_t e_shoff; uint16_t e_shentsize, e_shnum, e_shstrndx;
    std::memcpy(&e_shoff, data+40, 8);
    std::memcpy(&e_shentsize, data+58, 2);
    std::memcpy(&e_shnum, data+60, 2);
    std::memcpy(&e_shstrndx, data+62, 2);
    if (e_shentsize != 64 || e_shoff > len) return false;
    if ((uint64_t)e_shentsize * e_shnum > len - e_shoff || e_shstrndx >= e_shnum) return false;
    const char *strtab = nullptr;
    { const char *sh = data + e_shoff + (uint64_t)e_shentsize * e_shstrndx;
      uint64_t off, sz; std::memcpy(&off, sh+24, 8); std::memcpy(&sz, sh+32, 8);
      if (off > len || sz > len - off) return false; strtab = data + off; }
    for (uint16_t i = 0; i < e_shnum; ++i) {
        const char *sh = data + e_shoff + (uint64_t)e_shentsize * i;
        uint32_t sh_name, sh_type; uint64_t sh_off, sh_size;
        std::memcpy(&sh_name, sh+0, 4); std::memcpy(&sh_type, sh+4, 4);
        std::memcpy(&sh_off, sh+24, 8); std::memcpy(&sh_size, sh+32, 8);
        if (sh_type != 7) continue;
        if (sh_name >= 0x10000) continue; /* bounds */
        if (std::strcmp(strtab+sh_name, ".note.gnu.build-id") != 0) continue;
        if (sh_size < 16 || sh_off > len) return false;
        const char *p = data + sh_off;
        uint32_t namesz, descsz, type;
        std::memcpy(&namesz, p, 4); std::memcpy(&descsz, p+4, 4); std::memcpy(&type, p+8, 4);
        if (type != 3 || descsz < 8) return false;
        size_t na = (namesz + 3u) & ~3u;
        std::memcpy(out, p + 12 + na, 8);
        return true;
    }
    return false;
}
uint64_t Fnv1a(const char *data, size_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; ++i) { h ^= (unsigned char)data[i]; h *= 0x100000001b3ULL; }
    return h;
}
uint64_t Compute(const char *data, size_t len) {
    uint64_t v; if (ElfBuildId(data, len, &v)) return v; return Fnv1a(data, len);
}
} /* namespace fp */

/* ── Helpers ── */
#define ACL_CHECK(call, msg) do { aclError _rc = (call); if (_rc != ACL_SUCCESS) { \
    std::fprintf(stderr, "[ACL ERROR] %s: %d\n", msg, (int)_rc); return 1; } } while(0)
#define RT_CHECK(call, msg) do { rtError_t _rc = (call); if (_rc != RT_ERROR_NONE) { \
    std::fprintf(stderr, "[RT ERROR] %s: %d\n", msg, (int)_rc); return 1; } } while(0)

bool ReadFile(const std::string &path, std::vector<char> *out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "open %s failed\n", path.c_str()); return false; }
    f.seekg(0, std::ios::end); std::streamoff sz = f.tellg();
    if (sz < 0) return false;
    out->resize(static_cast<size_t>(sz));
    f.seekg(0); f.read(out->data(), static_cast<std::streamsize>(out->size()));
    return f.gcount() == static_cast<std::streamsize>(out->size());
}

struct DevBuf {
    void *ptr{nullptr};
    aclError Alloc(size_t n) { aclError rc = aclrtMalloc(&ptr, n, ACL_MEM_MALLOC_HUGE_FIRST); if (rc != ACL_SUCCESS) ptr = nullptr; return rc; }
    ~DevBuf() { if (ptr) aclrtFree(ptr); }
};

void WriteU64(char *buf, size_t off, uint64_t v) { std::memcpy(buf + off, &v, sizeof(v)); }

std::string MakeJsonDescriptor(uint64_t fp, const std::string &so_basename) {
    char init_op[128], alloc_op[128];
    std::snprintf(init_op, sizeof(init_op), "simpler_aicpu_init_%016lx", fp);
    std::snprintf(alloc_op, sizeof(alloc_op), "simpler_aicpu_run_%016lx", fp);
    auto entry = [&](const char *op, const char *fn) -> std::string {
        std::string s = "  \""; s += op; s += "\": {\n    \"opInfo\": {\n";
        s += "      \"functionName\": \""; s += fn; s += "\",\n";
        s += "      \"kernelSo\": \""; s += so_basename; s += "\",\n";
        s += "      \"opKernelLib\": \"AICPUKernel\",\n";
        s += "      \"computeCost\": \"100\",\n";
        s += "      \"engine\": \"DNN_VM_AICPU\",\n";
        s += "      \"flagAsync\": \"False\",\n";
        s += "      \"flagPartial\": \"False\",\n";
        s += "      \"userDefined\": \"False\"\n    }\n  }";
        return s;
    };
    return "{\n" + entry(init_op, "simpler_aicpu_init") + ",\n" + entry(alloc_op, "simpler_aicpu_run") + "\n}\n";
}

/* ── Main ── */
int main(int argc, char **argv) {
    const char *dispatcher_env = std::getenv("SIMPLER_DISPATCHER_SO");
    if (!dispatcher_env) { std::fprintf(stderr, "SIMPLER_DISPATCHER_SO must be set\n"); return 1; }
    std::string inner_path = "build/libsimt_case_aicpu.so";
    std::string aicore_path = "build/simt_case_aicore.o";
    int device_id = 0;
    uint32_t runs = 1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--device" && i+1 < argc) device_id = std::atoi(argv[++i]);
        else if (a == "--aicpu" && i+1 < argc) inner_path = argv[++i];
        else if (a == "--aicore" && i+1 < argc) aicore_path = argv[++i];
        else if (a == "--runs" && i+1 < argc) runs = (uint32_t)std::atoi(argv[++i]);
    }

    /* read files */
    std::vector<char> dispatcher_bytes, inner_bytes, aicore_bytes;
    if (!ReadFile(dispatcher_env, &dispatcher_bytes)) return 1;
    if (!ReadFile(inner_path, &inner_bytes)) return 1;
    if (!ReadFile(aicore_path, &aicore_bytes)) return 1;
    std::printf("[INIT] disp=%zuB aicpu=%zuB aicore=%zuB\n",
        dispatcher_bytes.size(), inner_bytes.size(), aicore_bytes.size());

    /* ACL init */
    ACL_CHECK(aclInit(nullptr), "aclInit");
    ACL_CHECK(aclrtSetDevice(device_id), "aclrtSetDevice");
    std::printf("[DEVICE] soc=%s\n", aclrtGetSocName() ?: "?");
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream), "createStream");

    /* alloc GM */
    DevBuf dev_dispatcher, dev_inner, dev_args, dev_state, dev_workspace;
    ACL_CHECK(dev_dispatcher.Alloc(dispatcher_bytes.size()), "malloc disp");
    ACL_CHECK(dev_inner.Alloc(inner_bytes.size()), "malloc inner");
    ACL_CHECK(dev_args.Alloc(160), "malloc args");
    ACL_CHECK(dev_state.Alloc(sizeof(FullPaState)), "malloc state");
    ACL_CHECK(dev_workspace.Alloc(kWorkloadBytes), "malloc ws");
    std::printf("[ALLOC] state=%p ws=%p\n", dev_state.ptr, dev_workspace.ptr);

    /* H2D dispatcher + inner */
    ACL_CHECK(aclrtMemcpy(dev_dispatcher.ptr, dispatcher_bytes.size(), dispatcher_bytes.data(), dispatcher_bytes.size(), ACL_MEMCPY_HOST_TO_DEVICE), "H2D disp");
    ACL_CHECK(aclrtMemcpy(dev_inner.ptr, inner_bytes.size(), inner_bytes.data(), inner_bytes.size(), ACL_MEMCPY_HOST_TO_DEVICE), "H2D inner");

    /* bootstrap: upload AICPU .so */
    {
        char hostargs[160] = {};
        WriteU64(hostargs, 96, reinterpret_cast<uint64_t>(dev_dispatcher.ptr));
        WriteU64(hostargs, 104, dispatcher_bytes.size());
        WriteU64(hostargs, 112, (uint64_t)device_id);
        WriteU64(hostargs, 120, reinterpret_cast<uint64_t>(dev_inner.ptr));
        WriteU64(hostargs, 128, inner_bytes.size());
        ACL_CHECK(aclrtMemcpy(dev_args.ptr, 160, hostargs, 160, ACL_MEMCPY_HOST_TO_DEVICE), "H2D args bootstrap");

        struct Args {
            struct { uint64_t unused[5] = {0}; uint64_t device_args_ptr = 0; uint64_t pad[20] = {0}; } k_args;
            char kernel_name[32]; char so_name[32]; char op_name[32];
        } args = {};
        args.k_args.device_args_ptr = reinterpret_cast<uint64_t>(dev_args.ptr);
        std::strncpy(args.kernel_name, "DynTileFwkKernelServerInit", sizeof(args.kernel_name) - 1);
        std::strncpy(args.so_name, "libaicpu_extend_kernels.so", sizeof(args.so_name) - 1);
        rtAicpuArgsEx_t rt_args = {};
        rt_args.args = &args; rt_args.argsSize = sizeof(args);
        rt_args.kernelNameAddrOffset = offsetof(Args, kernel_name);
        rt_args.soNameAddrOffset = offsetof(Args, so_name);
        RT_CHECK(rtAicpuKernelLaunchExWithArgs(rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", 1, &rt_args, nullptr, stream, 0), "bootstrap");
        ACL_CHECK(aclrtSynchronizeStream(stream), "sync bootstrap");
    }

    /* fingerprint + register */
    uint64_t fp = fp::Compute(inner_bytes.data(), inner_bytes.size());
    char base[64];
    std::snprintf(base, sizeof(base), "simt_case_inner_%016lx_%d.so", fp, device_id);
    std::printf("[bootstrap] %s fp=%016lx\n", base, fp);

    char json_tmpl[] = "/tmp/simt_case_XXXXXX.json";
    int fd = mkstemps(json_tmpl, 5);
    if (fd < 0) { std::fprintf(stderr, "mkstemps: %s\n", std::strerror(errno)); return 1; }
    std::string json = MakeJsonDescriptor(fp, base);
    ssize_t written = write(fd, json.data(), json.size());
    close(fd);
    if (written != (ssize_t)json.size()) { std::fprintf(stderr, "write json failed\n"); return 1; }

    rtLoadBinaryOption_t option = {};
    option.optionId = RT_LOAD_BINARY_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 0;
    rtLoadBinaryConfig_t load_config = {};
    load_config.options = &option; load_config.numOpt = 1;
    void *binary_handle = nullptr;
    RT_CHECK(rtsBinaryLoadFromFile(json_tmpl, &load_config, &binary_handle), "rtsBinaryLoadFromFile");
    std::remove(json_tmpl);

    rtFuncHandle init_handle = nullptr, alloc_handle = nullptr;
    {
        char init_op[128], alloc_op[128];
        std::snprintf(init_op, sizeof(init_op), "simpler_aicpu_init_%016lx", fp);
        std::snprintf(alloc_op, sizeof(alloc_op), "simpler_aicpu_run_%016lx", fp);
        RT_CHECK(rtsFuncGetByName(binary_handle, init_op, &init_handle), "get init"); std::printf("[DEBUG] init_handle=%p\n", init_handle);
        RT_CHECK(rtsFuncGetByName(binary_handle, alloc_op, &alloc_handle), "get alloc"); std::printf("[DEBUG] alloc_handle=%p\n", alloc_handle);
    }
    (void)init_handle;

    /* load AICore binary */
    aclrtBinHandle aicore_bin = nullptr; aclrtFuncHandle aicore_func = nullptr;
    {
        aclrtBinaryLoadOption lo{}; lo.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
        lo.value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
        aclrtBinaryLoadOptions los{&lo, 1U};
        ACL_CHECK(aclrtBinaryLoadFromData(aicore_bytes.data(), aicore_bytes.size(), &los, &aicore_bin), "load aicore");
        ACL_CHECK(aclrtBinaryGetFunctionByEntry(aicore_bin, 0, &aicore_func), "get aicore func");
    }

    /* init workspace */
    {
        std::vector<float> ws(static_cast<size_t>(kWorkloadTiles) * kWorkloadTileElements, kWorkloadOutputSentinel);
        std::fill_n(ws.data(), kWorkloadTileElements, kWorkloadInputA);
        std::fill_n(ws.data() + kWorkloadTileElements, kWorkloadTileElements, kWorkloadInputB);
        ACL_CHECK(aclrtMemcpy(dev_workspace.ptr, kWorkloadBytes, ws.data(), kWorkloadBytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D ws");
    }

    uint32_t passes = 0;
    for (uint32_t run = 0; run < runs; ++run) {
        /* clear state */
        ACL_CHECK(aclrtMemset(dev_state.ptr, sizeof(FullPaState), 0, sizeof(FullPaState)), "memset state");

        /* rewrite DeviceArgs for alloc: state_addr + workspace_addr */
        {
            char hostargs[160] = {};
            WriteU64(hostargs, 96, reinterpret_cast<uint64_t>(dev_state.ptr));
            WriteU64(hostargs, 104, reinterpret_cast<uint64_t>(dev_workspace.ptr));
            ACL_CHECK(aclrtMemcpy(dev_args.ptr, 160, hostargs, 160, ACL_MEMCPY_HOST_TO_DEVICE), "H2D args alloc");
        }

        /* launch AICPU init first */
        {
            struct LaunchArgs {
                uint64_t _pad[5] = {0};
                uint64_t device_args_ptr = 0;
                uint64_t reserved[20] = {0};
            } la = {};
            la.device_args_ptr = reinterpret_cast<uint64_t>(dev_args.ptr);
            rtCpuKernelArgs_t cpu_args = {};
            cpu_args.baseArgs.args = &la;
            cpu_args.baseArgs.argsSize = sizeof(la);
            rtLaunchKernelAttr_t attr = {};
            rtKernelLaunchCfg_t cfg = {&attr, 0};
            std::printf("[DEBUG] launching init_handle=%p\n", init_handle);
            RT_CHECK(rtsLaunchCpuKernel(init_handle, 1u, stream, &cfg, &cpu_args), "init launch");
            ACL_CHECK(aclrtSynchronizeStream(stream), "sync init");
        }

        /* launch AICPU alloc */
        {
            struct LaunchArgs {
                uint64_t _pad[5] = {0};
                uint64_t device_args_ptr = 0;
                uint64_t reserved[20] = {0};
            } la = {};
            la.device_args_ptr = reinterpret_cast<uint64_t>(dev_args.ptr);
            rtCpuKernelArgs_t cpu_args = {};
            cpu_args.baseArgs.args = &la;
            cpu_args.baseArgs.argsSize = sizeof(la);
            rtLaunchKernelAttr_t attr = {};
            rtKernelLaunchCfg_t cfg = {&attr, 0};
            std::printf("[DEBUG] launching alloc_handle=%p\n", alloc_handle); RT_CHECK(rtsLaunchCpuKernel(alloc_handle, 1u, stream, &cfg, &cpu_args), "alloc launch");
            ACL_CHECK(aclrtSynchronizeStream(stream), "sync alloc");
        }
        std::printf("[ALLOC] AICPU alloc done\n");

        /* launch AICore (executor + drain) */
        {
            struct KArgs { uint64_t ptr; } args{reinterpret_cast<uint64_t>(dev_state.ptr)};
            ACL_CHECK(aclrtLaunchKernelWithHostArgs(aicore_func, 32U, stream, nullptr, &args, sizeof(args), nullptr, 0), "launch aicore");
            ACL_CHECK(aclrtSynchronizeStream(stream), "sync aicore");
        }

        /* D2H verify */
        auto hs = std::make_unique<FullPaState>();
        ACL_CHECK(aclrtMemcpy(hs.get(), sizeof(FullPaState), dev_state.ptr, sizeof(FullPaState), ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
        bool ok = hs->drain.root_finished.value == 1U;
        uint32_t done = (uint32_t)hs->drain.done_count.value;
        uint32_t expected = KernelTaskCount(256);
        std::printf("[RESULT] run=%u root=%lld done=%u/%u fatal=0x%llx\n",
            run+1, (long long)hs->drain.root_finished.value, done, expected,
            (unsigned long long)hs->fatal.state);
        if (ok && done == expected) { ++passes; std::printf("[PASS]\n"); }
        else { std::printf("[FAIL]\n"); }
    }

    std::printf("[PERF] passes=%u/%u\n", passes, runs);
    aclrtBinaryUnLoad(aicore_bin);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return passes == runs ? 0 : 1;
}
