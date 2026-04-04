// CommitLLM capture implementation.
//
// Global state for the Q8_0 dot product capture hook.
// The hook and userdata are set once at startup (not thread-safe to change).
// The enable flag is atomic and can be toggled between graph computes.

#include "ggml-capture.h"
#include <stdatomic.h>
#include <stddef.h>

static ggml_capture_hook_t g_capture_hook      = NULL;
static void *              g_capture_user_data  = NULL;
static atomic_bool         g_capture_enabled    = false;

void ggml_capture_set_hook(ggml_capture_hook_t hook, void * user_data) {
    g_capture_hook     = hook;
    g_capture_user_data = user_data;
}

void ggml_capture_enable(bool enabled) {
    atomic_store(&g_capture_enabled, enabled);
}

bool ggml_capture_is_enabled(void) {
    return atomic_load(&g_capture_enabled);
}

// Internal: called by the instrumented dot product.
// Returns the hook and userdata if capture is active, NULL otherwise.
// This is the fast path — just an atomic load.
bool ggml_capture_active(ggml_capture_hook_t * out_hook, void ** out_user_data) {
    if (!atomic_load_explicit(&g_capture_enabled, memory_order_relaxed)) {
        return false;
    }
    ggml_capture_hook_t h = g_capture_hook;
    if (!h) {
        return false;
    }
    *out_hook      = h;
    *out_user_data = g_capture_user_data;
    return true;
}
