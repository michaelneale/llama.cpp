#pragma once

// CommitLLM challenge-response verification for llama-server.
//
// Implements a node validation protocol for mesh-llm:
//   1. Client enables capture for a single-token inference (the "challenge")
//   2. Server captures per-matmul INT32 accumulators during inference
//   3. Client fetches the captured data and verifies with Freivalds check
//   4. Capture turns off — zero overhead for all subsequent inference
//
// Data flow per matmul (Q8_0 × Q8_0 dot product):
//   - z[row] = Σ_b sumi[row, b]  (one INT32 per output row, summed across blocks)
//   - Stored per-matmul, indexed by call order within the forward pass
//
// Memory: for a 7B model single token decode, ~224 matmuls × 4096 rows × 4 bytes = ~3.5MB
//
// Thread safety: the hook fires from ggml worker threads. The buffer is
// guarded by a mutex. Each challenge produces one snapshot.

#include "ggml-capture.h"
#include "server-http.h"

#include <mutex>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <string>
#include <atomic>

// Per-matmul captured data: the INT32 accumulator vector z (one per output row).
// z[row] = Σ_blocks sumi[row, block] — the integer dot product before scale application.
struct commitllm_matmul_capture {
    std::vector<int32_t> z;      // INT32 accumulators, one per output row
    int                  n_rows; // number of output rows (= z.size())
    int                  n_blocks; // blocks per dot product (input_dim / 32)
};

// Challenge capture state. Accumulates matmul data during one forward pass.
struct commitllm_challenge_state {
    std::mutex mtx;

    // Map from matmul_id to accumulated z vector.
    // Each dot product appends z_row to the matmul identified by matmul_id.
    struct matmul_acc {
        std::vector<int32_t> z;
        int n_blocks = 0;
    };
    std::unordered_map<uint64_t, matmul_acc> acc_map;

    // Stats
    uint64_t total_dot_products = 0;

    // Hook callback — called from ggml worker threads for each dot product.
    // Sums per-block sumi into a single INT32 accumulator per row.
    static void hook(const struct ggml_capture_q8_data * data, void * user_data) {
        auto * state = static_cast<commitllm_challenge_state *>(user_data);

        // Sum per-block sumi to get z_row = Σ_b sumi[b]
        int32_t z_row = 0;
        for (int b = 0; b < data->n_blocks; b++) {
            z_row += data->sumi[b];
        }

        std::lock_guard<std::mutex> lock(state->mtx);
        state->total_dot_products++;

        auto & acc = state->acc_map[data->matmul_id];
        acc.z.push_back(z_row);
        acc.n_blocks = data->n_blocks;
    }

    // Flush accumulated data into sorted matmul captures.
    std::vector<commitllm_matmul_capture> flush() {
        std::lock_guard<std::mutex> lock(mtx);

        // Sort by matmul_id to get deterministic order.
        std::vector<uint64_t> ids;
        ids.reserve(acc_map.size());
        for (auto & kv : acc_map) {
            ids.push_back(kv.first);
        }
        std::sort(ids.begin(), ids.end());

        std::vector<commitllm_matmul_capture> out;
        out.reserve(ids.size());
        for (uint64_t id : ids) {
            auto & acc = acc_map[id];
            commitllm_matmul_capture cap;
            cap.z = std::move(acc.z);
            cap.n_rows = (int)cap.z.size();
            cap.n_blocks = acc.n_blocks;
            out.push_back(std::move(cap));
        }
        acc_map.clear();
        total_dot_products = 0;
        return out;
    }

    void install() {
        ggml_capture_set_hook(commitllm_challenge_state::hook, this);
    }
};

// Register CommitLLM challenge endpoints on the HTTP server.
//
// POST /commitllm/challenge/begin — enable capture for the next inference
// POST /commitllm/challenge/end   — disable capture, return captured matmul data
// GET  /commitllm/status          — check capture state
//
inline void commitllm_register_routes(server_http_context & ctx_http, commitllm_challenge_state & state) {

    ctx_http.post("/commitllm/challenge/begin", [&state](const server_http_req &) -> server_http_res_ptr {
        // Reset state and enable capture
        {
            std::lock_guard<std::mutex> lock(state.mtx);
            state.acc_map.clear();
            state.total_dot_products = 0;
        }
        ggml_capture_enable(true);
        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->data = "{\"status\":\"capturing\"}";
        return res;
    });

    ctx_http.post("/commitllm/challenge/end", [&state](const server_http_req &) -> server_http_res_ptr {
        ggml_capture_enable(false);

        auto matmuls = state.flush();

        // Build JSON response with per-matmul z vectors.
        // Binary format (msgpack) would be better for large models,
        // but JSON is fine for the handshake validation use case.
        std::string json = "{\"n_matmuls\":" + std::to_string(matmuls.size()) + ",\"matmuls\":[";
        for (size_t i = 0; i < matmuls.size(); i++) {
            if (i > 0) json += ",";
            const auto & m = matmuls[i];
            json += "{\"n_rows\":" + std::to_string(m.n_rows)
                  + ",\"n_blocks\":" + std::to_string(m.n_blocks)
                  + ",\"z\":[";
            for (size_t r = 0; r < m.z.size(); r++) {
                if (r > 0) json += ",";
                json += std::to_string(m.z[r]);
            }
            json += "]}";
        }
        json += "]}";

        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->data = std::move(json);
        return res;
    });

    ctx_http.get("/commitllm/status", [&state](const server_http_req &) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        std::lock_guard<std::mutex> lock(state.mtx);
        std::string json = "{\"enabled\":" + std::string(ggml_capture_is_enabled() ? "true" : "false")
            + ",\"matmuls_captured\":" + std::to_string(state.acc_map.size())
            + ",\"total_dot_products\":" + std::to_string(state.total_dot_products) + "}";
        res->status = 200;
        res->data = std::move(json);
        return res;
    });
}
