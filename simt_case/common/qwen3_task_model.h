#ifndef SIMT_CASE_QWEN3_TASK_MODEL_H
#define SIMT_CASE_QWEN3_TASK_MODEL_H

#include <stddef.h>
#include <stdint.h>

namespace simt_case_qwen3 {

constexpr uint32_t kUserBatch = 90;
constexpr uint32_t kBatchPadded = 96;
constexpr uint32_t kNumTiles = kBatchPadded / 16;
constexpr uint32_t kNumRows = kUserBatch;

constexpr uint32_t kQProjChunks = 20;
constexpr uint32_t kKVProjChunks = 8;
constexpr uint32_t kAttnChunks = 4;
constexpr uint32_t kOutProjChunks = 40;
constexpr uint32_t kMlpChunks = 34;
constexpr uint32_t kDownChunks = 40;

enum class TaskKind : uint32_t {
    RMSNorm = 0,
    QProj = 1,
    KProj = 2,
    VProj = 3,
    QKNorm = 4,
    ROPE = 5,
    QKMatmul = 6,
    Softmax = 7,
    SVMatmul = 8,
    OnlineSoftmax = 9,
    OutProj = 10,
    PostRMSNorm = 11,
    GateProj = 12,
    UpProj = 13,
    SILU = 14,
    DownProj = 15,
    DownProjRes = 16,
    Count = 17,
};

enum class EngineClass : uint8_t {
    Cube = 0,
    Vector = 1,
};

struct TaskInfo {
    TaskKind kind;
    EngineClass engine;
    uint32_t duration;
    uint32_t chunk_base;
    uint32_t chunk_count;
    uint32_t tile_idx;
    uint32_t row_idx;
    uint32_t pred_count;
    uint32_t pred_offset;
};

struct Qwen3TaskGraph {
    static constexpr uint32_t kMaxTasks = 4096;
    static constexpr uint32_t kMaxPredEdges = 65536;

    TaskInfo tasks[kMaxTasks];
    uint32_t pred_edges[kMaxPredEdges];
    uint32_t task_count;
    uint32_t pred_edge_count;

    void build_tier0();
};

inline uint32_t blocks_per_task(uint32_t total_chunks, uint32_t tier) {
    static const uint32_t targets[5] = {1, 2, 4, 8, 0x7FFFFFFF};
    uint32_t t = targets[tier < 5 ? tier : 0];
    return total_chunks < t ? total_chunks : t;
}

inline uint32_t n_tasks(uint32_t total_chunks, uint32_t bpt) {
    uint32_t n = 0;
    for (uint32_t base = 0; base < total_chunks; base += bpt)
        n++;
    return n;
}

inline void Qwen3TaskGraph::build_tier0() {
    task_count = 0;
    pred_edge_count = 0;

    uint32_t rmsnorm_ids[6];
    uint32_t qk_norm_ids[6];
    uint32_t v_ids[6][8];
    uint32_t v_cnt[6];
    uint32_t os_ids[90][4];
    uint32_t os_cnt[90];
    uint32_t op_ids[6][40];
    uint32_t op_cnt[6];
    uint32_t gate_ids[6][34];
    uint32_t up_ids[6][34];
    uint32_t silu_ids[6][34];
    uint32_t mlp_cnt[6];
    uint32_t down_ids[6][40];
    uint32_t down_cnt[6];

    for (uint32_t t = 0; t < 6; t++) {
        v_cnt[t] = 0;
        op_cnt[t] = 0;
        mlp_cnt[t] = 0;
        down_cnt[t] = 0;
    }
    for (uint32_t r = 0; r < 90; r++) os_cnt[r] = 0;

    auto add_task = [&](TaskKind kind, EngineClass eng, uint32_t dur,
                        uint32_t chunk_base, uint32_t chunk_count,
                        uint32_t tile, uint32_t row) -> uint32_t {
        uint32_t id = task_count++;
        tasks[id].kind = kind;
        tasks[id].engine = eng;
        tasks[id].duration = dur;
        tasks[id].chunk_base = chunk_base;
        tasks[id].chunk_count = chunk_count;
        tasks[id].tile_idx = tile;
        tasks[id].row_idx = row;
        tasks[id].pred_count = 0;
        tasks[id].pred_offset = pred_edge_count;
        return id;
    };

    auto add_pred = [&](uint32_t task_id, uint32_t pred_id) {
        if (tasks[task_id].pred_count == 0)
            tasks[task_id].pred_offset = pred_edge_count;
        pred_edges[pred_edge_count++] = pred_id;
        tasks[task_id].pred_count++;
    };

    for (uint32_t b0 = 0; b0 < kBatchPadded; b0 += 16) {
        uint32_t t = b0 / 16;

        rmsnorm_ids[t] = add_task(TaskKind::RMSNorm, EngineClass::Vector, 23950, 0, 1, t, 0);

        for (uint32_t base = 0; base < kQProjChunks; base++) {
            uint32_t id = add_task(TaskKind::QProj, EngineClass::Cube, 26060, base, 1, t, 0);
            add_pred(id, rmsnorm_ids[t]);
        }
        for (uint32_t base = 0; base < kKVProjChunks; base++) {
            uint32_t kid = add_task(TaskKind::KProj, EngineClass::Cube, 18170, base, 1, t, 0);
            add_pred(kid, rmsnorm_ids[t]);
            uint32_t vid = add_task(TaskKind::VProj, EngineClass::Cube, 17890, base, 1, t, 0);
            add_pred(vid, rmsnorm_ids[t]);
            v_ids[t][v_cnt[t]++] = vid;
        }
        qk_norm_ids[t] = add_task(TaskKind::QKNorm, EngineClass::Vector, 13190, 0, 1, t, 0);
        for (uint32_t i = 0; i < kQProjChunks; i++)
            add_pred(qk_norm_ids[t], rmsnorm_ids[t] + 1 + i);
        for (uint32_t i = 0; i < kKVProjChunks; i++)
            add_pred(qk_norm_ids[t], rmsnorm_ids[t] + 1 + kQProjChunks + i * 2);
    }

    for (uint32_t b = 0; b < kNumRows; b++) {
        uint32_t t = b / 16;
        uint32_t rope_id = add_task(TaskKind::ROPE, EngineClass::Vector, 9480, 0, 1, t, b);
        add_pred(rope_id, qk_norm_ids[t]);
        for (uint32_t i = 0; i < v_cnt[t]; i++)
            add_pred(rope_id, v_ids[t][i]);

        for (uint32_t base = 0; base < kAttnChunks; base++) {
            uint32_t qk_id = add_task(TaskKind::QKMatmul, EngineClass::Cube, 29350, base, 1, t, b);
            add_pred(qk_id, rope_id);
            uint32_t sm_id = add_task(TaskKind::Softmax, EngineClass::Vector, 19400, base, 1, t, b);
            add_pred(sm_id, qk_id);
            uint32_t sv_id = add_task(TaskKind::SVMatmul, EngineClass::Cube, 31650, base, 1, t, b);
            add_pred(sv_id, rope_id);
            add_pred(sv_id, sm_id);
            uint32_t os_id = add_task(TaskKind::OnlineSoftmax, EngineClass::Vector, 20820, base, 1, t, b);
            add_pred(os_id, sv_id);
            add_pred(os_id, sm_id);
            os_ids[b][os_cnt[b]++] = os_id;
        }
    }

    for (uint32_t b0 = 0; b0 < kBatchPadded; b0 += 16) {
        uint32_t t = b0 / 16;
        uint32_t cur_valid = (kUserBatch - b0 > 16) ? 16 : (kUserBatch - b0);

        for (uint32_t base = 0; base < kOutProjChunks; base++) {
            uint32_t id = add_task(TaskKind::OutProj, EngineClass::Cube, 40750, base, 1, t, 0);
            op_ids[t][op_cnt[t]++] = id;
            for (uint32_t row = 0; row < cur_valid; row++) {
                uint32_t bb = b0 + row;
                for (uint32_t i = 0; i < os_cnt[bb]; i++)
                    add_pred(id, os_ids[bb][i]);
            }
        }
        uint32_t post_id = add_task(TaskKind::PostRMSNorm, EngineClass::Vector, 24390, 0, 1, t, 0);
        for (uint32_t i = 0; i < op_cnt[t]; i++)
            add_pred(post_id, op_ids[t][i]);

        for (uint32_t base = 0; base < kMlpChunks; base++) {
            uint32_t gid = add_task(TaskKind::GateProj, EngineClass::Cube, 95700, base, 1, t, 0);
            add_pred(gid, post_id);
            gate_ids[t][mlp_cnt[t]] = gid;
            uint32_t uid = add_task(TaskKind::UpProj, EngineClass::Cube, 97140, base, 1, t, 0);
            add_pred(uid, post_id);
            up_ids[t][mlp_cnt[t]] = uid;
            uint32_t sid = add_task(TaskKind::SILU, EngineClass::Vector, 2820, base, 1, t, 0);
            add_pred(sid, gid);
            add_pred(sid, uid);
            silu_ids[t][mlp_cnt[t]] = sid;
            mlp_cnt[t]++;
        }
        for (uint32_t base = 0; base < kDownChunks; base++) {
            uint32_t did = add_task(TaskKind::DownProj, EngineClass::Cube, 72220, base, 1, t, 0);
            for (uint32_t i = 0; i < mlp_cnt[t]; i++)
                add_pred(did, silu_ids[t][i]);
            down_ids[t][down_cnt[t]++] = did;
            uint32_t drid = add_task(TaskKind::DownProjRes, EngineClass::Vector, 2590, base, 1, t, 0);
            add_pred(drid, did);
            add_pred(drid, op_ids[t][base < op_cnt[t] ? base : op_cnt[t] - 1]);
        }
    }
}

}

#endif
