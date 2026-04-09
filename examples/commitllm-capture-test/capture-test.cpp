// CommitLLM capture test.
//
// Loads a GGUF model, runs a short prompt with capture enabled,
// and prints per-block sumi statistics to verify the capture hook works.
//
// Usage:
//   ./commitllm-capture-test -m model.gguf -p "Hello world" -ngl 0
//
// The -ngl 0 forces CPU backend (capture only works on CPU today).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"
#include "ggml-capture.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

// Capture statistics — accumulated across all dot products.
struct CaptureStats {
    std::mutex mtx;
    std::atomic<uint64_t> n_dot_products{0};
    std::atomic<uint64_t> n_blocks_total{0};

    // Per-block sumi histogram: count of sumi values by magnitude bucket
    // bucket 0: |sumi| < 100
    // bucket 1: |sumi| < 1000
    // bucket 2: |sumi| < 10000
    // bucket 3: |sumi| >= 10000
    std::atomic<uint64_t> sumi_buckets[4]{};

    // Track min/max sumi seen
    std::atomic<int32_t> sumi_min{INT32_MAX};
    std::atomic<int32_t> sumi_max{INT32_MIN};
};

static CaptureStats g_stats;

static void capture_hook(const struct ggml_capture_q8_data * data, void * /*user_data*/) {
    g_stats.n_dot_products.fetch_add(1, std::memory_order_relaxed);
    g_stats.n_blocks_total.fetch_add(data->n_blocks, std::memory_order_relaxed);

    for (int b = 0; b < data->n_blocks; b++) {
        int32_t s = data->sumi[b];
        int32_t abs_s = s < 0 ? -s : s;

        if (abs_s < 100)       g_stats.sumi_buckets[0].fetch_add(1, std::memory_order_relaxed);
        else if (abs_s < 1000) g_stats.sumi_buckets[1].fetch_add(1, std::memory_order_relaxed);
        else if (abs_s < 10000) g_stats.sumi_buckets[2].fetch_add(1, std::memory_order_relaxed);
        else                    g_stats.sumi_buckets[3].fetch_add(1, std::memory_order_relaxed);

        // Atomic min/max
        int32_t cur_min = g_stats.sumi_min.load(std::memory_order_relaxed);
        while (s < cur_min && !g_stats.sumi_min.compare_exchange_weak(cur_min, s, std::memory_order_relaxed));

        int32_t cur_max = g_stats.sumi_max.load(std::memory_order_relaxed);
        while (s > cur_max && !g_stats.sumi_max.compare_exchange_weak(cur_max, s, std::memory_order_relaxed));
    }
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

    if (tokens.empty()) {
        LOG_ERR("%s: no input tokens\n", __func__);
        return false;
    }

    LOG_INF("%s: running %zu tokens with capture enabled\n", __func__, tokens.size());

    // Enable capture before decode
    ggml_capture_enable(true);

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s: failed to eval\n", __func__);
        return false;
    }

    // Disable capture
    ggml_capture_enable(false);

    return true;
}

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    // Install capture hook
    ggml_capture_set_hook(capture_hook, nullptr);

    // Force CPU — capture only works on CPU backend
    params.n_gpu_layers = 0;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (!model || !ctx) {
        LOG_ERR("Failed to init model/context\n");
        return 1;
    }

    LOG_INF("\n=== CommitLLM Capture Test ===\n\n");

    bool ok = run(ctx, params);

    // Print results
    uint64_t n_dots   = g_stats.n_dot_products.load();
    uint64_t n_blocks = g_stats.n_blocks_total.load();
    int32_t  s_min    = g_stats.sumi_min.load();
    int32_t  s_max    = g_stats.sumi_max.load();

    LOG_INF("\n=== Capture Results ===\n");
    LOG_INF("  dot products captured: %llu\n", (unsigned long long)n_dots);
    LOG_INF("  total blocks:          %llu\n", (unsigned long long)n_blocks);
    LOG_INF("  sumi range:            [%d, %d]\n", s_min, s_max);
    LOG_INF("  sumi histogram:\n");
    LOG_INF("    |sumi| < 100:        %llu\n", (unsigned long long)g_stats.sumi_buckets[0].load());
    LOG_INF("    |sumi| < 1000:       %llu\n", (unsigned long long)g_stats.sumi_buckets[1].load());
    LOG_INF("    |sumi| < 10000:      %llu\n", (unsigned long long)g_stats.sumi_buckets[2].load());
    LOG_INF("    |sumi| >= 10000:     %llu\n", (unsigned long long)g_stats.sumi_buckets[3].load());

    if (n_dots > 0) {
        LOG_INF("\n  ✅ Capture hook is working — %llu dot products intercepted\n", (unsigned long long)n_dots);
        LOG_INF("  Average blocks per dot: %.1f\n", (double)n_blocks / n_dots);
    } else {
        LOG_INF("\n  ❌ No dot products captured. Is the model Q8_0? Is -ngl 0?\n");
    }

    llama_perf_context_print(ctx);
    llama_backend_free();

    return ok ? 0 : 1;
}
