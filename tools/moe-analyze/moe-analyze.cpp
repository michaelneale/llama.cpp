// MoE Expert Routing Analyzer
//
// Loads an MoE model, runs prompts, and collects expert routing statistics
// using the eval callback to observe ffn_moe_probs tensors.
//
// This answers: "If I restrict routing to a subset of experts, how much
// router probability mass do I lose?" — the key question for distributed
// MoE with masked expert groups.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <set>
#include <fstream>

// ---- data structures ----

struct expert_stats {
    // per-expert counters across all layers and tokens
    std::vector<double> total_mass;     // [n_expert] sum of gate probs
    std::vector<int64_t> selection_count; // [n_expert] times in top-k
    int64_t total_tokens = 0;
    int64_t total_moe_layers_seen = 0;
};

struct layer_snapshot {
    int layer_id;
    int n_expert;
    int n_tokens;
    std::vector<float> probs; // [n_expert * n_tokens], row-major
};

struct callback_data {
    std::vector<uint8_t> buf;           // scratch for GPU->CPU copy
    std::vector<layer_snapshot> snapshots;
    int n_expert = 0;
    int n_expert_used = 0;
    int max_layers_to_log = 6;         // only log first N MoE layers (low bias)
    int layers_seen_this_eval = 0;
    bool collect = true;
};

static std::string export_ranking_path;  // if non-empty, export per-expert mass to this file

// ---- eval callback: intercept ffn_moe_probs ----

static bool moe_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb = (callback_data *) user_data;

    if (!cb->collect) return false;

    // we only care about ffn_moe_probs (the router output before masking/top-k)
    if (ask) {
        if (strstr(t->name, "ffn_moe_probs") != nullptr
            && strstr(t->name, "masked") == nullptr
            && strstr(t->name, "biased") == nullptr) {
            return true;
        }
        return false;
    }

    // data retrieval phase
    if (strstr(t->name, "ffn_moe_probs") == nullptr) return true;
    if (strstr(t->name, "masked") != nullptr || strstr(t->name, "biased") != nullptr) return true;

    // parse layer id from name: "ffn_moe_probs-L" where L is the layer
    int layer_id = -1;
    const char * dash = strrchr(t->name, '-');
    if (dash) layer_id = atoi(dash + 1);

    if (layer_id >= cb->max_layers_to_log) return true;

    cb->layers_seen_this_eval++;

    // tensor shape: [n_expert, n_tokens]
    int n_expert = (int)t->ne[0];
    int n_tokens = (int)t->ne[1];
    cb->n_expert = n_expert;

    // copy data from GPU if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    float * data_ptr;
    if (!is_host) {
        size_t n_bytes = ggml_nbytes(t);
        cb->buf.resize(n_bytes);
        ggml_backend_tensor_get(t, cb->buf.data(), 0, n_bytes);
        data_ptr = (float *)cb->buf.data();
    } else {
        data_ptr = (float *)t->data;
    }

    // store snapshot
    layer_snapshot snap;
    snap.layer_id = layer_id;
    snap.n_expert = n_expert;
    snap.n_tokens = n_tokens;
    snap.probs.assign(data_ptr, data_ptr + n_expert * n_tokens);
    cb->snapshots.push_back(std::move(snap));

    return true;
}

// ---- analysis functions ----

struct group_score {
    int group_id;
    double mass_captured;   // sum of top-k probs within group (incl replicated)
    double mass_total;      // total prob mass in group
};

// Given a set of replicated expert IDs, compute group capture ratio
// where each group gets its own experts PLUS the replicated set
static double compute_group_capture_with_replicas(
    const float * probs, int n_expert, int top_k,
    int group_id, int experts_per_group,
    const std::set<int> & replicated
) {
    // collect probs for experts in this group + replicated
    std::vector<float> eligible_probs;
    for (int e = group_id * experts_per_group;
         e < (group_id + 1) * experts_per_group && e < n_expert; e++) {
        eligible_probs.push_back(probs[e]);
    }
    for (int r : replicated) {
        // don't double-count if already in group
        if (r >= group_id * experts_per_group && r < (group_id + 1) * experts_per_group) continue;
        if (r >= 0 && r < n_expert) {
            eligible_probs.push_back(probs[r]);
        }
    }
    std::sort(eligible_probs.begin(), eligible_probs.end(), std::greater<float>());
    double captured = 0;
    for (int i = 0; i < top_k && i < (int)eligible_probs.size(); i++) {
        captured += eligible_probs[i];
    }
    return captured;
}

// For a given snapshot (one layer, one or more tokens), compute:
// - per-expert mass
// - group scores for different group assignments
static void analyze_snapshot(
    const layer_snapshot & snap,
    int n_groups,
    int top_k,
    expert_stats & stats,
    std::vector<std::vector<group_score>> & group_results  // [token][group]
) {
    int n_exp = snap.n_expert;

    for (int tok = 0; tok < snap.n_tokens; tok++) {
        const float * p = snap.probs.data() + tok * n_exp;

        // accumulate per-expert stats
        for (int e = 0; e < n_exp; e++) {
            stats.total_mass[e] += p[e];
        }

        // find global top-k
        std::vector<int> sorted_idx(n_exp);
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::sort(sorted_idx.begin(), sorted_idx.end(),
            [&](int a, int b) { return p[a] > p[b]; });

        for (int i = 0; i < top_k && i < n_exp; i++) {
            stats.selection_count[sorted_idx[i]]++;
        }
        stats.total_tokens++;

        // compute group scores
        int experts_per_group = n_exp / n_groups;
        std::vector<group_score> scores(n_groups);

        for (int g = 0; g < n_groups; g++) {
            scores[g].group_id = g;
            scores[g].mass_total = 0;
            scores[g].mass_captured = 0;

            // collect probs for experts in this group
            std::vector<float> group_probs;
            for (int e = g * experts_per_group; e < (g + 1) * experts_per_group && e < n_exp; e++) {
                group_probs.push_back(p[e]);
                scores[g].mass_total += p[e];
            }

            // sort descending and take top-k
            std::sort(group_probs.begin(), group_probs.end(), std::greater<float>());
            for (int i = 0; i < top_k && i < (int)group_probs.size(); i++) {
                scores[g].mass_captured += group_probs[i];
            }
        }

        group_results.push_back(std::move(scores));
    }

    stats.total_moe_layers_seen++;
}

// ---- prompts ----

static std::vector<std::string> get_test_prompts() {
    return {
        // code
        "<|im_start|>user\nWrite a Python function to find the nth Fibonacci number using dynamic programming.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite a Rust function that reads a CSV file and returns a Vec of structs.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain how a B-tree index works in a database.<|im_end|>\n<|im_start|>assistant\n",
        // reasoning
        "<|im_start|>user\nIf all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nA train travels 120km in 2 hours. It then speeds up and covers the next 180km in 2 hours. What is the average speed for the whole journey?<|im_end|>\n<|im_start|>assistant\n",
        // chat
        "<|im_start|>user\nHello! What's the best way to learn a new language?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nTell me a joke about programmers.<|im_end|>\n<|im_start|>assistant\n",
        // instruction
        "<|im_start|>user\nSummarize the key differences between TCP and UDP in 3 bullet points.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nTranslate 'The weather is beautiful today' to French, Spanish, and German.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nList 5 healthy breakfast options with brief descriptions.<|im_end|>\n<|im_start|>assistant\n",
    };
}

// ---- main ----

int main(int argc, char ** argv) {
    common_params params;

    params.prompt = ""; // will be set per-prompt
    params.n_predict = 32;

    // Pre-scan argv for our custom args and strip them before common_params_parse
    bool all_layers = false;
    std::vector<const char *> filtered_argv;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--export-ranking") == 0 && i + 1 < argc) {
            export_ranking_path = argv[++i];
        } else if (strcmp(argv[i], "--all-layers") == 0) {
            all_layers = true;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }

    int filtered_argc = (int)filtered_argv.size();
    if (!common_params_parse(filtered_argc, const_cast<char **>(filtered_argv.data()), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    // setup callback
    callback_data cb_data;
    params.cb_eval = moe_callback;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    // load model
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (!model || !ctx) {
        LOG_ERR("Failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    // get model info from metadata
    int n_expert = 0;
    int n_expert_used = 0;
    {
        int n_meta = llama_model_meta_count(model);
        for (int i = 0; i < n_meta; i++) {
            char key[256], val[256];
            llama_model_meta_key_by_index(model, i, key, sizeof(key));
            llama_model_meta_val_str_by_index(model, i, val, sizeof(val));
            if (strstr(key, "expert_count") && !strstr(key, "used") && !strstr(key, "shared")) {
                n_expert = atoi(val);
            }
            if (strstr(key, "expert_used_count")) {
                n_expert_used = atoi(val);
            }
        }
    }

    LOG_INF("\n=== MoE Expert Routing Analysis ===\n");
    LOG_INF("Model experts: %d, used per token: %d\n", n_expert, n_expert_used);

    if (n_expert == 0) {
        LOG_ERR("Model is not MoE (n_expert=0). Nothing to analyze.\n");
        return 1;
    }

    cb_data.n_expert = n_expert;
    cb_data.n_expert_used = n_expert_used;

    if (all_layers) {
        cb_data.max_layers_to_log = 9999; // log all MoE layers
        LOG_INF("Logging ALL MoE layers (--all-layers)\n");
    }
    if (!export_ranking_path.empty()) {
        LOG_INF("Will export expert ranking to: %s\n", export_ranking_path.c_str());
        if (!all_layers) {
            LOG_INF("  NOTE: consider --all-layers for better ranking accuracy\n");
        }
    }

    // run prompts and collect routing data
    auto prompts = get_test_prompts();
    expert_stats stats;
    stats.total_mass.resize(n_expert, 0.0);
    stats.selection_count.resize(n_expert, 0);

    // group analysis: try 2, 3, 4, 8 groups
    std::vector<int> group_configs = {2, 3, 4, 8};

    // keep all snapshots for final analysis
    std::vector<layer_snapshot> all_snapshots;

    int n_gen = params.n_predict;
    LOG_INF("Running %zu prompts, generating %d tokens each\n", prompts.size(), n_gen);
    LOG_INF("Logging first %d MoE layers per eval\n\n", cb_data.max_layers_to_log);

    for (size_t pi = 0; pi < prompts.size(); pi++) {
        LOG_INF("Prompt %zu/%zu: %.60s...\n", pi+1, prompts.size(), prompts[pi].c_str());

        // tokenize
        std::vector<llama_token> tokens = common_tokenize(ctx, prompts[pi], add_bos);

        // clear KV cache
        llama_memory_clear(llama_get_memory(ctx), true);

        // reset callback state
        cb_data.snapshots.clear();
        cb_data.layers_seen_this_eval = 0;
        cb_data.collect = true;

        // prefill
        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
            LOG_ERR("  prefill failed\n");
            continue;
        }

        // generate tokens
        for (int i = 0; i < n_gen; i++) {
            // sample next token (greedy)
            const float * logits = llama_get_logits_ith(ctx, -1);
            llama_token next = 0;
            float max_logit = -INFINITY;
            int n_vocab = llama_vocab_n_tokens(vocab);
            for (int v = 0; v < n_vocab; v++) {
                if (logits[v] > max_logit) {
                    max_logit = logits[v];
                    next = v;
                }
            }

            if (llama_vocab_is_eog(vocab, next)) break;

            cb_data.layers_seen_this_eval = 0;
            if (llama_decode(ctx, llama_batch_get_one(&next, 1))) {
                LOG_ERR("  decode failed at token %d\n", i);
                break;
            }
        }

        // accumulate stats from collected snapshots
        for (auto & snap : cb_data.snapshots) {
            std::vector<std::vector<group_score>> dummy_results;
            analyze_snapshot(snap, 1, n_expert_used, stats, dummy_results);
        }

        // keep snapshots for group analysis
        all_snapshots.insert(all_snapshots.end(),
            std::make_move_iterator(cb_data.snapshots.begin()),
            std::make_move_iterator(cb_data.snapshots.end()));

        LOG_INF("  collected %zu layer snapshots (total: %zu)\n",
            cb_data.snapshots.size() - 0, all_snapshots.size());
    }

    // ---- print results ----

    LOG_INF("\n=== Expert Popularity (gate mass, summed across all tokens & logged layers) ===\n");
    {
        // sort experts by total mass
        std::vector<int> sorted_exp(n_expert);
        std::iota(sorted_exp.begin(), sorted_exp.end(), 0);
        std::sort(sorted_exp.begin(), sorted_exp.end(),
            [&](int a, int b) { return stats.total_mass[a] > stats.total_mass[b]; });

        double total = 0;
        for (int e = 0; e < n_expert; e++) total += stats.total_mass[e];

        LOG_INF("Total tokens × layers: %lld\n", (long long)stats.total_tokens);
        LOG_INF("\nTop 20 experts by gate mass:\n");
        LOG_INF("  %-8s  %-12s  %-8s  %-10s\n", "Expert", "Mass", "Mass%", "Selected");
        for (int i = 0; i < std::min(20, n_expert); i++) {
            int e = sorted_exp[i];
            LOG_INF("  %-8d  %-12.4f  %-8.2f  %-10lld\n",
                e, stats.total_mass[e],
                100.0 * stats.total_mass[e] / total,
                (long long)stats.selection_count[e]);
        }

        // concentration: top-N experts capture X% of mass
        double cumulative = 0;
        LOG_INF("\nConcentration:\n");
        for (int n : {4, 8, 16, 32, 64}) {
            if (n > n_expert) break;
            cumulative = 0;
            for (int i = 0; i < n; i++) cumulative += stats.total_mass[sorted_exp[i]];
            LOG_INF("  Top %3d experts: %.1f%% of total gate mass\n", n, 100.0 * cumulative / total);
        }
    }

    // identify hot experts for replication analysis
    std::vector<int> hot_experts(n_expert);
    std::iota(hot_experts.begin(), hot_experts.end(), 0);
    std::sort(hot_experts.begin(), hot_experts.end(),
        [&](int a, int b) { return stats.total_mass[a] > stats.total_mass[b]; });

    // Export ranking if requested
    if (!export_ranking_path.empty()) {
        double total = 0;
        for (int e = 0; e < n_expert; e++) total += stats.total_mass[e];

        std::ofstream fout(export_ranking_path);
        if (!fout.is_open()) {
            LOG_ERR("Failed to open export file: %s\n", export_ranking_path.c_str());
        } else {
            fout << "# MoE expert ranking by gate mass\n";
            fout << "# Model: " << params.model.path << "\n";
            fout << "# Experts: " << n_expert << " (top-" << n_expert_used << ")\n";
            fout << "# Prompts: " << prompts.size() << " x " << n_gen << " tokens\n";
            fout << "# Layers logged: " << (all_layers ? "all" : std::to_string(cb_data.max_layers_to_log)) << "\n";
            fout << "# Total token-layer observations: " << stats.total_tokens << "\n";
            fout << "#\n";
            fout << "# Format: expert_id,gate_mass,mass_pct,selection_count\n";
            fout << "# Sorted by gate_mass descending (hottest first)\n";
            for (int i = 0; i < n_expert; i++) {
                int e = hot_experts[i];
                fout << e << "," << stats.total_mass[e] << ","
                     << (100.0 * stats.total_mass[e] / total) << ","
                     << stats.selection_count[e] << "\n";
            }
            fout.close();
            LOG_INF("\nExported expert ranking to: %s\n", export_ranking_path.c_str());
            LOG_INF("Use with moe-split: --group-map <file generated from this ranking>\n\n");
        }
    }

    LOG_INF("\n=== Group Masking Analysis (best-group capture ratio) ===\n");
    LOG_INF("For each group count, what fraction of the unrestricted top-%d mass\n", n_expert_used);
    LOG_INF("is captured by the best single group?\n\n");

    // compute with replicated experts too
    std::vector<int> replica_counts = {0, 1, 2, 4, 8};

    LOG_INF("  %-8s  %-8s  %-10s  %-10s  %-10s  %-10s  %-10s\n",
        "Groups", "Replicas", "Exp/Grp", "Mean", "P25", "P50", "P5");

    for (int ng : group_configs) {
        if (n_expert % ng != 0) continue;
        int epg = n_expert / ng;

        for (int nr : replica_counts) {
            if (nr > n_expert / 2) continue;  // don't replicate more than half

            // build replicated set from top hot experts
            std::set<int> replicated;
            for (int i = 0; i < nr; i++) {
                replicated.insert(hot_experts[i]);
            }

            // re-analyze all snapshots with this config
            std::vector<double> ratios;

            for (auto & snap : all_snapshots) {
                for (int tok = 0; tok < snap.n_tokens; tok++) {
                    const float * p = snap.probs.data() + tok * snap.n_expert;

                    // baseline: unrestricted top-k mass
                    std::vector<float> all_probs(p, p + snap.n_expert);
                    std::sort(all_probs.begin(), all_probs.end(), std::greater<float>());
                    double baseline = 0;
                    for (int i = 0; i < n_expert_used && i < (int)all_probs.size(); i++) {
                        baseline += all_probs[i];
                    }

                    // find best group
                    double best_group = 0;
                    for (int g = 0; g < ng; g++) {
                        double cap = compute_group_capture_with_replicas(
                            p, snap.n_expert, n_expert_used, g, epg, replicated);
                        best_group = std::max(best_group, cap);
                    }

                    if (baseline > 1e-9) {
                        ratios.push_back(best_group / baseline);
                    }
                }
            }

            if (ratios.empty()) continue;
            std::sort(ratios.begin(), ratios.end());
            double mean = std::accumulate(ratios.begin(), ratios.end(), 0.0) / ratios.size();
            double p5  = ratios[std::max(0, (int)(0.05 * ratios.size()))];
            double p25 = ratios[std::max(0, (int)(0.25 * ratios.size()))];
            double p50 = ratios[std::max(0, (int)(0.50 * ratios.size()))];

            LOG_INF("  %-8d  %-8d  %-10d  %-10.3f  %-10.3f  %-10.3f  %-10.3f\n",
                ng, nr, epg, mean, p25, p50, p5);
        }
        LOG_INF("\n");
    }

    LOG_INF("=== Interpretation ===\n");
    LOG_INF("Mean close to 1.0 = masking barely hurts (best group captures most of top-k mass)\n");
    LOG_INF("Mean < 0.7 = significant quality risk from group restriction\n");
    LOG_INF("P5 close to 1.0 = even worst-case tokens are OK\n");
    LOG_INF("P5 < 0.5 = some tokens will be badly served by any single group\n\n");

    // ---- Phase 1b: actual masked generation quality comparison ----
    LOG_INF("=== Phase 1b: Masked Generation Quality (logprob comparison) ===\n");

    // test with 4 groups (32 experts each for 128-expert models)
    int test_groups = 4;
    if (n_expert % test_groups != 0) test_groups = 2;
    int test_epg = n_expert / test_groups;

    LOG_INF("Testing %d groups (%d experts/group) vs baseline (all %d experts)\n", test_groups, test_epg, n_expert);
    LOG_INF("Using first 5 prompts, generating %d tokens each\n\n", n_gen);

    cb_data.collect = false; // stop collecting routing data

    auto run_generation = [&](const std::string & /*label*/, int prompt_count) -> std::vector<double> {
        std::vector<double> avg_logprobs;
        for (int pi = 0; pi < prompt_count && pi < (int)prompts.size(); pi++) {
            std::vector<llama_token> tokens = common_tokenize(ctx, prompts[pi], add_bos);
            llama_memory_clear(llama_get_memory(ctx), true);

            if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) continue;

            double sum_logprob = 0;
            int gen_count = 0;
            for (int i = 0; i < n_gen; i++) {
                const float * logits = llama_get_logits_ith(ctx, -1);
                int n_vocab_val = llama_vocab_n_tokens(vocab);

                // find argmax and compute logprob
                llama_token next = 0;
                float max_logit = -INFINITY;
                for (int v = 0; v < n_vocab_val; v++) {
                    if (logits[v] > max_logit) { max_logit = logits[v]; next = v; }
                }
                if (llama_vocab_is_eog(vocab, next)) break;

                // compute log softmax for the chosen token
                float log_sum_exp = 0;
                float max_l = max_logit;
                for (int v = 0; v < n_vocab_val; v++) {
                    log_sum_exp += expf(logits[v] - max_l);
                }
                float logprob = max_logit - max_l - logf(log_sum_exp);
                sum_logprob += logprob;
                gen_count++;

                if (llama_decode(ctx, llama_batch_get_one(&next, 1))) break;
            }
            if (gen_count > 0) {
                avg_logprobs.push_back(sum_logprob / gen_count);
            }
        }
        return avg_logprobs;
    };

    // baseline: no mask
    llama_model_set_expert_mask(const_cast<llama_model *>(llama_get_model(ctx)), nullptr, 0);
    auto baseline_lp = run_generation("baseline", 5);

    // masked: test each group
    for (int g = 0; g < test_groups; g++) {
        bool mask[512] = {};
        for (int e = g * test_epg; e < (g + 1) * test_epg && e < n_expert; e++) {
            mask[e] = true;
        }
        // also include top-2 hot experts as replicas
        mask[hot_experts[0]] = true;
        mask[hot_experts[1]] = true;

        llama_model_set_expert_mask(
            const_cast<llama_model *>(llama_get_model(ctx)),
            mask, n_expert);

        auto masked_lp = run_generation("group_" + std::to_string(g), 5);

        // compute delta
        double delta_sum = 0;
        int delta_count = 0;
        for (int i = 0; i < (int)std::min(baseline_lp.size(), masked_lp.size()); i++) {
            delta_sum += (masked_lp[i] - baseline_lp[i]);
            delta_count++;
        }

        if (delta_count > 0) {
            LOG_INF("  Group %d (experts %d-%d + 2 hot replicas): avg logprob delta = %+.4f\n",
                g, g * test_epg, (g+1) * test_epg - 1, delta_sum / delta_count);
        }
    }

    // clear mask
    llama_model_set_expert_mask(const_cast<llama_model *>(llama_get_model(ctx)), nullptr, 0);

    LOG_INF("\n=== Interpretation (logprob delta) ===\n");
    LOG_INF("Delta near 0.0 = masking barely affects generation quality\n");
    LOG_INF("Delta < -0.1 = noticeable quality loss\n");
    LOG_INF("Delta < -0.5 = significant degradation\n\n");

    llama_backend_free();
    return 0;
}
