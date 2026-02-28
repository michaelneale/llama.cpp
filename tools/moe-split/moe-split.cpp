// moe-split: split a MoE GGUF into per-node GGUFs with trunk + expert group
//
// Each output GGUF contains:
//   - Full trunk (embeddings, attention, norms, router gate)
//   - Only the expert tensors for the assigned group
//   - Updated metadata (n_expert = group_size)
//
// Expert selection modes:
//   1. Contiguous: --groups N splits experts into N equal contiguous groups
//   2. Custom:     --expert-list 4,72,80,... selects specific experts by ID
//
// Router gate rows are gathered to match the selected experts, so
// expert i in the output GGUF has the correct routing weights.
//
// Usage:
//   llama-moe-split --model input.gguf --groups 4 --output-prefix node
//   → produces node-0.gguf, node-1.gguf, node-2.gguf, node-3.gguf
//
//   llama-moe-split --model input.gguf --groups 4 --group-id 2 --output node2.gguf
//   → produces only group 2
//
//   llama-moe-split --model input.gguf --expert-list 64,65,72,80 -o custom.gguf
//   → produces a GGUF with only those 4 experts (renumbered 0-3)

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

struct moe_split_params {
    std::string input;
    std::string output;           // single output (with --group-id or --expert-list)
    std::string output_prefix;    // prefix for multi-output (with --groups)
    int n_groups   = 0;
    int group_id   = -1;          // -1 = produce all groups
    bool dry_run   = false;
    std::vector<int> expert_list; // custom expert selection (empty = use contiguous groups)
};

static void print_usage(const char * prog) {
    printf("\n");
    printf("usage: %s [options]\n", prog);
    printf("\n");
    printf("Split a MoE GGUF into per-node GGUFs (trunk + expert group).\n");
    printf("\n");
    printf("options:\n");
    printf("  -m, --model FILE          input GGUF file\n");
    printf("  -g, --groups N            number of expert groups to split into\n");
    printf("  --group-id N              produce only this group (0-indexed)\n");
    printf("  --expert-list IDs         comma-separated expert IDs (e.g. 64,65,72,80)\n");
    printf("  -o, --output FILE         output file (requires --group-id or --expert-list)\n");
    printf("  --output-prefix PREFIX    output prefix (produces PREFIX-N.gguf)\n");
    printf("  --dry-run                 show plan without writing\n");
    printf("  -h, --help                show this help\n");
    printf("\n");
}

static std::vector<int> parse_int_list(const std::string & s) {
    std::vector<int> result;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        result.push_back(std::atoi(token.c_str()));
    }
    return result;
}

static void parse_args(int argc, const char ** argv, moe_split_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "-m" || arg == "--model") {
            params.input = argv[++i];
        } else if (arg == "-g" || arg == "--groups") {
            params.n_groups = std::atoi(argv[++i]);
        } else if (arg == "--group-id") {
            params.group_id = std::atoi(argv[++i]);
        } else if (arg == "--expert-list") {
            params.expert_list = parse_int_list(argv[++i]);
        } else if (arg == "-o" || arg == "--output") {
            params.output = argv[++i];
        } else if (arg == "--output-prefix") {
            params.output_prefix = argv[++i];
        } else if (arg == "--dry-run") {
            params.dry_run = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            exit(1);
        }
    }

    if (params.input.empty()) {
        fprintf(stderr, "error: --model is required\n");
        exit(1);
    }

    // Validate argument combinations
    if (!params.expert_list.empty()) {
        // Custom expert list mode
        if (params.output.empty()) {
            fprintf(stderr, "error: --output is required with --expert-list\n");
            exit(1);
        }
        if (params.n_groups > 0) {
            fprintf(stderr, "error: --groups and --expert-list are mutually exclusive\n");
            exit(1);
        }
    } else {
        // Contiguous group mode
        if (params.n_groups <= 0) {
            fprintf(stderr, "error: --groups or --expert-list is required\n");
            exit(1);
        }
        if (params.group_id >= 0 && params.output.empty()) {
            fprintf(stderr, "error: --output is required when --group-id is set\n");
            exit(1);
        }
        if (params.group_id < 0 && params.output_prefix.empty()) {
            fprintf(stderr, "error: --output-prefix is required when --group-id is not set\n");
            exit(1);
        }
    }
}

// Check if tensor name matches an expert-packed tensor (ffn_gate_exps, ffn_up_exps, ffn_down_exps)
static bool is_expert_tensor(const char * name) {
    return strstr(name, "ffn_gate_exps") != nullptr
        || strstr(name, "ffn_up_exps")   != nullptr
        || strstr(name, "ffn_down_exps") != nullptr;
}

// Check if tensor is the MoE router gate (ffn_gate_inp)
static bool is_router_gate(const char * name) {
    return strstr(name, "ffn_gate_inp") != nullptr;
}

static void zeros(std::ofstream & f, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; i++) {
        f.write(&zero, 1);
    }
}

// Write a single group/expert-list GGUF
// expert_ids: which original expert indices to include (renumbered 0..N-1 in output)
static void write_group(
        const moe_split_params & params,
        struct gguf_context * ctx_in,
        struct ggml_context * ctx_meta,
        std::ifstream & f_in,
        const std::vector<int> & expert_ids,
        const std::string & output_path,
        int n_expert,
        int n_expert_tensors,
        const char * label) {
    int experts_per_group = (int)expert_ids.size();

    printf("\n--- %s: %d experts → %s ---\n", label, experts_per_group, output_path.c_str());
    printf("  Expert IDs:");
    for (int i = 0; i < (int)expert_ids.size() && i < 16; i++) {
        printf(" %d", expert_ids[i]);
    }
    if ((int)expert_ids.size() > 16) printf(" ... (%d more)", (int)expert_ids.size() - 16);
    printf("\n");

    // Check if expert_ids are contiguous (optimization: can use single read)
    bool contiguous = true;
    for (int i = 1; i < (int)expert_ids.size(); i++) {
        if (expert_ids[i] != expert_ids[i-1] + 1) {
            contiguous = false;
            break;
        }
    }
    if (contiguous) {
        printf("  (contiguous range [%d, %d) — using fast path)\n",
            expert_ids[0], expert_ids[0] + experts_per_group);
    }

    // Create output GGUF context
    auto * ctx_out = gguf_init_empty();

    // Copy all KV metadata
    gguf_set_kv(ctx_out, ctx_in);

    int n_tensors = gguf_get_n_tensors(ctx_in);

    // Find the architecture
    int arch_key = gguf_find_key(ctx_in, "general.architecture");
    std::string arch_name;
    if (arch_key >= 0) {
        arch_name = gguf_get_val_str(ctx_in, arch_key);
    }

    // Update expert count metadata
    std::string expert_count_key = arch_name + ".expert_count";
    gguf_set_val_u32(ctx_out, expert_count_key.c_str(), experts_per_group);

    // Clamp expert_used_count if needed
    std::string eu_key_str = arch_name + ".expert_used_count";
    int eu_idx_local = gguf_find_key(ctx_out, eu_key_str.c_str());
    if (eu_idx_local >= 0) {
        int eu_val = gguf_get_val_u32(ctx_out, eu_idx_local);
        if (eu_val > experts_per_group) {
            printf("  Clamping expert_used_count: %d → %d\n", eu_val, experts_per_group);
            gguf_set_val_u32(ctx_out, eu_key_str.c_str(), experts_per_group);
        }
    }

    // Create a separate ggml context for sliced tensors
    size_t ctx_size = (size_t)(n_expert_tensors + 48 + 16) * ggml_tensor_overhead() + 4096;
    struct ggml_init_params ctx_params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx_new = ggml_init(ctx_params);

    size_t total_expert_bytes_saved = 0;

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx_in, i);
        struct ggml_tensor * t = ggml_get_tensor(ctx_meta, name);

        if (is_expert_tensor(name)) {
            int n_dims = ggml_n_dims(t);
            int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
            for (int d = 0; d < n_dims; d++) {
                ne[d] = t->ne[d];
            }

            int expert_dim = n_dims - 1;
            if (ne[expert_dim] != n_expert) {
                fprintf(stderr, "warning: tensor %s has ne[%d]=%lld, expected %d experts. Copying as-is.\n",
                    name, expert_dim, (long long)ne[expert_dim], n_expert);
                gguf_add_tensor(ctx_out, t);
                continue;
            }

            int64_t ne_new[GGML_MAX_DIMS] = {1, 1, 1, 1};
            for (int d = 0; d < n_dims; d++) {
                ne_new[d] = ne[d];
            }
            ne_new[expert_dim] = experts_per_group;

            struct ggml_tensor * t_new = ggml_new_tensor(ctx_new, t->type, n_dims, ne_new);
            ggml_set_name(t_new, name);
            gguf_add_tensor(ctx_out, t_new);

            size_t bytes_per_expert = ggml_nbytes(t) / n_expert;
            total_expert_bytes_saved += bytes_per_expert * (n_expert - experts_per_group);

            printf("  %-40s [%lld x %lld x %lld] → [%lld x %lld x %lld]  (%.1f MB → %.1f MB)\n",
                name,
                (long long)ne[0], (long long)ne[1], (long long)ne[2],
                (long long)ne_new[0], (long long)ne_new[1], (long long)ne_new[2],
                ggml_nbytes(t) / 1e6,
                ggml_nbytes(t_new) / 1e6);
        } else if (is_router_gate(name)) {
            int n_dims = ggml_n_dims(t);
            int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
            for (int d = 0; d < n_dims; d++) {
                ne[d] = t->ne[d];
            }

            int expert_dim = n_dims - 1;
            if (ne[expert_dim] != n_expert) {
                fprintf(stderr, "warning: router gate %s has ne[%d]=%lld, expected %d. Copying as-is.\n",
                    name, expert_dim, (long long)ne[expert_dim], n_expert);
                gguf_add_tensor(ctx_out, t);
                continue;
            }

            int64_t ne_new[GGML_MAX_DIMS] = {1, 1, 1, 1};
            for (int d = 0; d < n_dims; d++) {
                ne_new[d] = ne[d];
            }
            ne_new[expert_dim] = experts_per_group;

            struct ggml_tensor * t_new = ggml_new_tensor(ctx_new, t->type, n_dims, ne_new);
            ggml_set_name(t_new, name);
            gguf_add_tensor(ctx_out, t_new);

            size_t bytes_per_expert = ggml_nbytes(t) / n_expert;
            total_expert_bytes_saved += bytes_per_expert * (n_expert - experts_per_group);

            printf("  %-40s [%lld x %lld] → [%lld x %lld]  (router gate gathered)\n",
                name,
                (long long)ne[0], (long long)ne[1],
                (long long)ne_new[0], (long long)ne_new[1]);
        } else {
            gguf_add_tensor(ctx_out, t);
        }
    }

    printf("  Total expert bytes saved: %.1f MB\n", total_expert_bytes_saved / 1e6);

    if (params.dry_run) {
        printf("  (dry run, not writing)\n");
        gguf_free(ctx_out);
        ggml_free(ctx_new);
        return;
    }

    // Write the output file
    printf("  Writing %s ...\n", output_path.c_str());

    std::ofstream f_out(output_path, std::ios::binary);
    if (!f_out.is_open()) {
        fprintf(stderr, "error: cannot open output file %s\n", output_path.c_str());
        gguf_free(ctx_out);
        ggml_free(ctx_new);
        return;
    }
    f_out.exceptions(std::ofstream::failbit);

    // Write metadata header
    size_t meta_size = gguf_get_meta_size(ctx_out);
    std::vector<uint8_t> meta_data(meta_size);
    gguf_get_meta_data(ctx_out, meta_data.data());
    f_out.write((const char *)meta_data.data(), meta_size);

    // Write tensor data — gather expert rows by expert_ids
    std::vector<uint8_t> read_buf;
    std::vector<uint8_t> gather_buf;

    for (int i = 0; i < gguf_get_n_tensors(ctx_out); i++) {
        const char * name = gguf_get_tensor_name(ctx_out, i);

        int i_in = gguf_find_tensor(ctx_in, name);
        if (i_in < 0) {
            fprintf(stderr, "error: tensor %s not found in input\n", name);
            exit(1);
        }

        struct ggml_tensor * t_in = ggml_get_tensor(ctx_meta, name);
        size_t offset_in = gguf_get_data_offset(ctx_in) + gguf_get_tensor_offset(ctx_in, i_in);

        if (is_expert_tensor(name) || is_router_gate(name)) {
            size_t bytes_per_expert = ggml_nbytes(t_in) / n_expert;
            size_t total_out_bytes  = (size_t)experts_per_group * bytes_per_expert;

            if (contiguous) {
                // Fast path: single contiguous read
                size_t slice_offset = offset_in + (size_t)expert_ids[0] * bytes_per_expert;
                read_buf.resize(total_out_bytes);
                f_in.seekg(slice_offset);
                f_in.read((char *)read_buf.data(), total_out_bytes);
                f_out.write((const char *)read_buf.data(), total_out_bytes);
            } else {
                // Gather path: read each expert's slice individually
                gather_buf.resize(total_out_bytes);
                read_buf.resize(bytes_per_expert);
                for (int e = 0; e < experts_per_group; e++) {
                    size_t src_offset = offset_in + (size_t)expert_ids[e] * bytes_per_expert;
                    f_in.seekg(src_offset);
                    f_in.read((char *)read_buf.data(), bytes_per_expert);
                    memcpy(gather_buf.data() + (size_t)e * bytes_per_expert,
                           read_buf.data(), bytes_per_expert);
                }
                f_out.write((const char *)gather_buf.data(), total_out_bytes);
            }
            zeros(f_out, GGML_PAD(total_out_bytes, GGUF_DEFAULT_ALIGNMENT) - total_out_bytes);
        } else {
            // Copy full tensor
            size_t n_bytes = ggml_nbytes(t_in);
            read_buf.resize(n_bytes);
            f_in.seekg(offset_in);
            f_in.read((char *)read_buf.data(), n_bytes);
            f_out.write((const char *)read_buf.data(), n_bytes);
            zeros(f_out, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);
        }
    }

    f_out.close();
    printf("  Done. Output size: %.1f MB\n",
        (double)std::ifstream(output_path, std::ios::ate | std::ios::binary).tellg() / 1e6);

    gguf_free(ctx_out);
    ggml_free(ctx_new);
}

int main(int argc, const char ** argv) {
    moe_split_params params;
    parse_args(argc, argv, params);

    // Load input GGUF metadata (no tensor data)
    struct ggml_context * ctx_meta = nullptr;
    struct gguf_init_params init_params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_meta,
    };

    auto * ctx_in = gguf_init_from_file(params.input.c_str(), init_params);
    if (!ctx_in) {
        fprintf(stderr, "error: failed to load %s\n", params.input.c_str());
        return 1;
    }

    std::ifstream f_in(params.input, std::ios::binary);
    if (!f_in.is_open()) {
        fprintf(stderr, "error: cannot open %s\n", params.input.c_str());
        return 1;
    }

    // Read model metadata
    int n_tensors = gguf_get_n_tensors(ctx_in);
    int arch_key = gguf_find_key(ctx_in, "general.architecture");
    std::string arch_name;
    if (arch_key >= 0) {
        arch_name = gguf_get_val_str(ctx_in, arch_key);
    }

    // Find expert count
    std::string expert_key = arch_name + ".expert_count";
    int ec_idx = gguf_find_key(ctx_in, expert_key.c_str());
    if (ec_idx < 0) {
        fprintf(stderr, "error: no expert count found (key: %s). Is this a MoE model?\n", expert_key.c_str());
        return 1;
    }
    int n_expert = gguf_get_val_u32(ctx_in, ec_idx);

    // Find expert_used count
    std::string eu_key = arch_name + ".expert_used_count";
    int eu_idx = gguf_find_key(ctx_in, eu_key.c_str());
    int n_expert_used = eu_idx >= 0 ? gguf_get_val_u32(ctx_in, eu_idx) : 0;

    printf("Model: %s\n", params.input.c_str());
    printf("Architecture: %s\n", arch_name.c_str());
    printf("Experts: %d (top-%d)\n", n_expert, n_expert_used);
    printf("Tensors: %d\n", n_tensors);

    // Count expert vs trunk tensors
    int n_expert_tensors = 0;
    size_t expert_total_bytes = 0;
    size_t trunk_total_bytes = 0;
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx_in, i);
        struct ggml_tensor * t = ggml_get_tensor(ctx_meta, name);
        if (is_expert_tensor(name)) {
            n_expert_tensors++;
            expert_total_bytes += ggml_nbytes(t);
        } else {
            trunk_total_bytes += ggml_nbytes(t);
        }
    }

    if (!params.expert_list.empty()) {
        // ======= Custom expert list mode =======
        int experts_per_group = (int)params.expert_list.size();

        printf("Mode: custom expert list\n");
        printf("Selected experts: %d\n", experts_per_group);

        // Validate expert IDs
        for (int eid : params.expert_list) {
            if (eid < 0 || eid >= n_expert) {
                fprintf(stderr, "error: expert ID %d out of range [0, %d)\n", eid, n_expert);
                return 1;
            }
        }

        if (experts_per_group < n_expert_used) {
            fprintf(stderr, "warning: selected %d experts < expert_used (%d). "
                "Routing will be forced to reuse experts within group.\n",
                experts_per_group, n_expert_used);
        }

        printf("\nTrunk tensors: %d (%.1f MB)\n", n_tensors - n_expert_tensors, trunk_total_bytes / 1e6);
        printf("Expert tensors: %d (%.1f MB total)\n", n_expert_tensors, expert_total_bytes / 1e6);
        printf("Expected output size: %.1f MB\n",
            (trunk_total_bytes + expert_total_bytes * experts_per_group / n_expert) / 1e6);

        write_group(params, ctx_in, ctx_meta, f_in, params.expert_list,
                    params.output, n_expert, n_expert_tensors, "Custom");
    } else {
        // ======= Contiguous group mode =======
        printf("Mode: contiguous groups\n");
        printf("Groups: %d\n", params.n_groups);

        if (n_expert % params.n_groups != 0) {
            fprintf(stderr, "error: n_expert (%d) must be divisible by n_groups (%d)\n",
                n_expert, params.n_groups);
            return 1;
        }

        int experts_per_group = n_expert / params.n_groups;
        printf("Experts per group: %d\n", experts_per_group);

        if (experts_per_group < n_expert_used) {
            fprintf(stderr, "warning: experts_per_group (%d) < expert_used (%d). "
                "Routing will be forced to reuse experts within group.\n",
                experts_per_group, n_expert_used);
        }

        printf("\nTrunk tensors: %d (%.1f MB)\n", n_tensors - n_expert_tensors, trunk_total_bytes / 1e6);
        printf("Expert tensors: %d (%.1f MB total, %.1f MB per group)\n",
            n_expert_tensors, expert_total_bytes / 1e6,
            expert_total_bytes / (double)params.n_groups / 1e6);
        printf("Expected output size per node: %.1f MB\n",
            (trunk_total_bytes + expert_total_bytes / params.n_groups) / 1e6);

        if (params.group_id >= 0) {
            // Single group
            if (params.group_id >= params.n_groups) {
                fprintf(stderr, "error: group_id %d >= n_groups %d\n", params.group_id, params.n_groups);
                return 1;
            }
            int start = params.group_id * experts_per_group;
            std::vector<int> ids(experts_per_group);
            for (int i = 0; i < experts_per_group; i++) ids[i] = start + i;

            char label[64];
            snprintf(label, sizeof(label), "Group %d [%d, %d)", params.group_id, start, start + experts_per_group);
            write_group(params, ctx_in, ctx_meta, f_in, ids,
                        params.output, n_expert, n_expert_tensors, label);
        } else {
            // All groups
            for (int g = 0; g < params.n_groups; g++) {
                int start = g * experts_per_group;
                std::vector<int> ids(experts_per_group);
                for (int i = 0; i < experts_per_group; i++) ids[i] = start + i;

                char path[1024], label[64];
                snprintf(path, sizeof(path), "%s-%d.gguf", params.output_prefix.c_str(), g);
                snprintf(label, sizeof(label), "Group %d [%d, %d)", g, start, start + experts_per_group);
                write_group(params, ctx_in, ctx_meta, f_in, ids,
                            path, n_expert, n_expert_tensors, label);
            }
        }
    }

    f_in.close();
    gguf_free(ctx_in);
    ggml_free(ctx_meta);

    printf("\nDone.\n");
    return 0;
}
