#pragma once

// CommitLLM capture interface for ggml.
//
// Provides a hook into Q8_0 dot products to extract per-block INT32
// accumulators (sumi) needed for Freivalds verification.
//
// Usage:
//   1. Set the capture callback with ggml_capture_set_hook()
//   2. Enable capture with ggml_capture_enable(true) before a graph compute
//   3. The callback fires for every Q8_0 dot product with per-block sumi
//   4. Disable capture when done
//
// Thread safety: the callback must be thread-safe. It will be called from
// multiple threads during parallel mul_mat computation. The enable flag
// is atomic.

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Per-block capture data passed to the hook.
//
// For a single dot product (one output row), this contains the per-block
// INT32 accumulators and FP32 block scales. The dot product computes:
//
//   output = Σ_b (d_w[b] * d_x[b] * sumi[b])
//
// where sumi[b] = Σ_{j=0}^{31} w_q8[b*32+j] * x_q8[b*32+j]
//
struct ggml_capture_q8_data {
    int      n_blocks;       // number of Q8_0 blocks (= n_elements / 32)
    int32_t *sumi;           // per-block INT32 accumulators [n_blocks]
    float   *d_w;            // per-block weight scales (FP16→FP32) [n_blocks]
    float   *d_x;            // per-block input scales (FP16→FP32) [n_blocks]
    float    result;         // final dot product result (sum of d_w*d_x*sumi)
};

// Callback type. Called once per dot product (once per output row per matmul).
//
// user_data: opaque pointer set via ggml_capture_set_hook()
// data:      per-block capture data (valid only during the callback)
//
typedef void (*ggml_capture_hook_t)(const struct ggml_capture_q8_data * data, void * user_data);

// Set the capture hook. Pass NULL to clear.
void ggml_capture_set_hook(ggml_capture_hook_t hook, void * user_data);

// Enable or disable capture globally. When disabled, the dot product
// runs without any overhead. Uses atomic operations.
void ggml_capture_enable(bool enabled);

// Check if capture is currently enabled.
bool ggml_capture_is_enabled(void);

// Internal: check if capture is active and return hook/userdata.
// Used by instrumented dot products. Returns true if a hook should fire.
bool ggml_capture_active(ggml_capture_hook_t * out_hook, void ** out_user_data);

#ifdef __cplusplus
}
#endif
