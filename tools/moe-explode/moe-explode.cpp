// moe-explode: split a MoE GGUF into trunk + per-expert files, or reassemble them
//
// Explode mode (default):
//   Reads a full MoE GGUF and produces:
//     trunk.gguf          — all non-expert tensors (attention, embeddings, norms, router gates, shared expert)
//     expert-NNN.gguf     — one file per expert (expert tensors only, single expert slice)
//
//   The trunk.gguf has expert_count set to 0 and expert tensors removed.
//   Each expert-NNN.gguf has expert_count=1 and contains only that expert's tensor slices.
//
// Assemble mode (--assemble):
//   Reads trunk.gguf + a list of expert files and produces a single shard GGUF.
//   The output is loadable by llama-server with the expert mask set.
//
// Usage:
//   llama-moe-explode -m model.gguf -o output-dir/          # explode
//   llama-moe-explode --assemble -o shard.gguf --trunk output-dir/trunk.gguf \
//     --experts output-dir/expert-000.gguf,output-dir/expert-003.gguf,...

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

struct explode_params {
    std::string input;            // full MoE GGUF (explode mode)
    std::string output;           // output directory (explode) or output file (assemble)
    std::string trunk_file;       // trunk.gguf path (assemble mode)
    std::vector<std::string> expert_files; // expert file paths (assemble mode)
    bool assemble  = false;
    bool dry_run   = false;
};

static void print_usage(const char * prog) {
    printf("\nusage: %s [options]\n\n", prog);
    printf("Explode a MoE GGUF into trunk + per-expert files, or reassemble them.\n\n");
    printf("Explode mode (default):\n");
    printf("  %s -m model.gguf -o output-dir/\n\n", prog);
    printf("Assemble mode:\n");
    printf("  %s --assemble -o shard.gguf --trunk dir/trunk.gguf --experts dir/expert-000.gguf,dir/expert-005.gguf,...\n\n", prog);
    printf("Options:\n");
    printf("  -m, --model FILE       input MoE GGUF (explode mode)\n");
    printf("  -o, --output PATH      output directory (explode) or output file (assemble)\n");
    printf("  --assemble             assemble mode: combine trunk + experts into shard\n");
    printf("  --trunk FILE           trunk.gguf path (assemble mode)\n");
    printf("  --experts FILE,FILE,.. comma-separated expert file paths (assemble mode)\n");
    printf("  --dry-run              print plan without writing\n");
    printf("  -h, --help             show this help\n");
}

static void parse_args(int argc, const char ** argv, explode_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { print_usage(argv[0]); exit(0); }
        else if (arg == "-m" || arg == "--model")    { params.input = argv[++i]; }
        else if (arg == "-o" || arg == "--output")   { params.output = argv[++i]; }
        else if (arg == "--assemble")                { params.assemble = true; }
        else if (arg == "--trunk")                   { params.trunk_file = argv[++i]; }
        else if (arg == "--dry-run")                 { params.dry_run = true; }
        else if (arg == "--experts") {
            std::string list = argv[++i];
            std::stringstream ss(list);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) params.expert_files.push_back(item);
            }
        }
        else { fprintf(stderr, "error: unknown argument: %s\n", arg.c_str()); exit(1); }
    }
}

static bool is_expert_tensor(const char * name) {
    return strstr(name, "ffn_gate_exps") != nullptr
        || strstr(name, "ffn_up_exps")   != nullptr
        || strstr(name, "ffn_down_exps") != nullptr;
}

static bool is_router_gate(const char * name) {
    return strstr(name, "ffn_gate_inp") != nullptr;
}

static void zeros(std::ofstream & f, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; i++) f.write(&zero, 1);
}

static std::string find_arch(struct gguf_context * ctx) {
    int k = gguf_find_key(ctx, "general.architecture");
    if (k >= 0) return gguf_get_val_str(ctx, k);
    return "";
}

// ── Explode: full GGUF → trunk.gguf + expert-NNN.gguf ──

static void do_explode(const explode_params & params) {
    if (params.input.empty() || params.output.empty()) {
        fprintf(stderr, "error: need -m and -o for explode mode\n");
        exit(1);
    }

    struct gguf_init_params gguf_params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    struct ggml_context * ctx_meta = nullptr;
    gguf_params.ctx = &ctx_meta;
    struct gguf_context * ctx_in = gguf_init_from_file(params.input.c_str(), gguf_params);
    if (!ctx_in) { fprintf(stderr, "error: failed to open %s\n", params.input.c_str()); exit(1); }

    std::ifstream f_in(params.input, std::ios::binary);
    if (!f_in.is_open()) { fprintf(stderr, "error: cannot read %s\n", params.input.c_str()); exit(1); }

    int n_tensors = gguf_get_n_tensors(ctx_in);
    std::string arch = find_arch(ctx_in);

    // Find expert count
    std::string ec_key = arch + ".expert_count";
    int ec_idx = gguf_find_key(ctx_in, ec_key.c_str());
    if (ec_idx < 0) { fprintf(stderr, "error: not a MoE model (no %s)\n", ec_key.c_str()); exit(1); }
    int n_expert = gguf_get_val_u32(ctx_in, ec_idx);
    printf("Model: %s, %d experts, %d tensors\n", params.input.c_str(), n_expert, n_tensors);

    // Count expert tensors
    int n_expert_tensors = 0;
    int n_trunk_tensors = 0;
    for (int i = 0; i < n_tensors; i++) {
        if (is_expert_tensor(gguf_get_tensor_name(ctx_in, i))) n_expert_tensors++;
        else n_trunk_tensors++;
    }
    printf("Trunk tensors: %d, Expert tensors: %d (%d per expert)\n",
        n_trunk_tensors, n_expert_tensors, n_expert_tensors > 0 ? n_expert_tensors : 0);

    fs::path out_dir(params.output);
    if (!params.dry_run) fs::create_directories(out_dir);

    // ── Write trunk.gguf ──
    {
        printf("\n--- trunk.gguf ---\n");
        auto * ctx_out = gguf_init_empty();
        gguf_set_kv(ctx_out, ctx_in);
        // Set expert_count to 0 in trunk
        gguf_set_val_u32(ctx_out, ec_key.c_str(), 0);

        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(ctx_in, i);
            if (is_expert_tensor(name)) continue; // skip expert tensors
            struct ggml_tensor * t = ggml_get_tensor(ctx_meta, name);
            gguf_add_tensor(ctx_out, t);
        }

        fs::path trunk_path = out_dir / "trunk.gguf";
        printf("  %d tensors\n", n_trunk_tensors);

        if (!params.dry_run) {
            std::ofstream f_out(trunk_path.string(), std::ios::binary);
            size_t meta_size = gguf_get_meta_size(ctx_out);
            std::vector<uint8_t> meta_data(meta_size);
            gguf_get_meta_data(ctx_out, meta_data.data());
            f_out.write((const char *)meta_data.data(), meta_size);

            std::vector<uint8_t> buf;
            for (int i = 0; i < gguf_get_n_tensors(ctx_out); i++) {
                const char * name = gguf_get_tensor_name(ctx_out, i);
                int i_in = gguf_find_tensor(ctx_in, name);
                struct ggml_tensor * t_in = ggml_get_tensor(ctx_meta, name);
                size_t offset_in = gguf_get_data_offset(ctx_in) + gguf_get_tensor_offset(ctx_in, i_in);
                size_t n_bytes = ggml_nbytes(t_in);
                buf.resize(n_bytes);
                f_in.seekg(offset_in);
                f_in.read((char *)buf.data(), n_bytes);
                f_out.write((const char *)buf.data(), n_bytes);
                zeros(f_out, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);
            }
            f_out.close();
            auto sz = fs::file_size(trunk_path);
            printf("  Written: %s (%.1f GB)\n", trunk_path.string().c_str(), sz / 1e9);
        }
        gguf_free(ctx_out);
    }

    // ── Write expert-NNN.gguf for each expert ──
    for (int eid = 0; eid < n_expert; eid++) {
        char fname[64];
        snprintf(fname, sizeof(fname), "expert-%03d.gguf", eid);
        fs::path expert_path = out_dir / fname;

        if (eid % 32 == 0 || eid == n_expert - 1) {
            printf("\n--- expert-%03d ... ---\n", eid);
        }

        // Build a GGUF with just this expert's slices (expert_count=1)
        auto * ctx_out = gguf_init_empty();

        // Minimal metadata: architecture, expert_count=1
        gguf_set_val_str(ctx_out, "general.architecture", arch.c_str());
        gguf_set_val_u32(ctx_out, ec_key.c_str(), 1);
        // Store original expert ID so assemble knows where it came from
        gguf_set_val_u32(ctx_out, "moe_explode.expert_id", eid);
        gguf_set_val_u32(ctx_out, "moe_explode.original_expert_count", n_expert);

        // Create ggml context for sliced expert tensors
        size_t ctx_size = (size_t)(n_expert_tensors + 32) * ggml_tensor_overhead() + 4096;
        struct ggml_init_params gp = { ctx_size, nullptr, true };
        struct ggml_context * ctx_new = ggml_init(gp);

        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(ctx_in, i);
            if (!is_expert_tensor(name)) continue;

            struct ggml_tensor * t = ggml_get_tensor(ctx_meta, name);
            int n_dims = ggml_n_dims(t);
            int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
            for (int d = 0; d < n_dims; d++) ne[d] = t->ne[d];

            // Expert dim is outermost — set to 1
            ne[n_dims - 1] = 1;
            struct ggml_tensor * t_new = ggml_new_tensor(ctx_new, t->type, n_dims, ne);
            ggml_set_name(t_new, name);
            gguf_add_tensor(ctx_out, t_new);
        }

        if (!params.dry_run) {
            std::ofstream f_out(expert_path.string(), std::ios::binary);
            size_t meta_size = gguf_get_meta_size(ctx_out);
            std::vector<uint8_t> meta_data(meta_size);
            gguf_get_meta_data(ctx_out, meta_data.data());
            f_out.write((const char *)meta_data.data(), meta_size);

            std::vector<uint8_t> buf;
            for (int i = 0; i < gguf_get_n_tensors(ctx_out); i++) {
                const char * name = gguf_get_tensor_name(ctx_out, i);
                int i_in = gguf_find_tensor(ctx_in, name);
                struct ggml_tensor * t_in = ggml_get_tensor(ctx_meta, name);
                size_t offset_in = gguf_get_data_offset(ctx_in) + gguf_get_tensor_offset(ctx_in, i_in);
                size_t bytes_per_expert = ggml_nbytes(t_in) / n_expert;

                buf.resize(bytes_per_expert);
                f_in.seekg(offset_in + (size_t)eid * bytes_per_expert);
                f_in.read((char *)buf.data(), bytes_per_expert);
                f_out.write((const char *)buf.data(), bytes_per_expert);
                zeros(f_out, GGML_PAD(bytes_per_expert, GGUF_DEFAULT_ALIGNMENT) - bytes_per_expert);
            }
            f_out.close();
        }

        ggml_free(ctx_new);
        gguf_free(ctx_out);
    }

    if (!params.dry_run) {
        size_t trunk_sz = fs::file_size(out_dir / "trunk.gguf");
        size_t expert_sz = 0;
        for (int eid = 0; eid < n_expert; eid++) {
            char fname[64]; snprintf(fname, sizeof(fname), "expert-%03d.gguf", eid);
            expert_sz += fs::file_size(out_dir / fname);
        }
        printf("\nDone. trunk: %.1f GB, experts: %d × %.0f MB = %.1f GB, total: %.1f GB\n",
            trunk_sz / 1e9, n_expert, (expert_sz / (double)n_expert) / 1e6,
            expert_sz / 1e9, (trunk_sz + expert_sz) / 1e9);
    }

    gguf_free(ctx_in);
    ggml_free(ctx_meta);
}

// ── Assemble: trunk.gguf + expert-NNN.gguf files → single shard GGUF ──

static void do_assemble(const explode_params & params) {
    if (params.trunk_file.empty() || params.expert_files.empty() || params.output.empty()) {
        fprintf(stderr, "error: need --trunk, --experts, and -o for assemble mode\n");
        exit(1);
    }

    int n_experts_out = (int)params.expert_files.size();
    printf("Assembling shard: trunk + %d experts → %s\n", n_experts_out, params.output.c_str());

    // Open trunk
    struct gguf_init_params gguf_params = { true, nullptr };
    struct ggml_context * trunk_meta = nullptr;
    gguf_params.ctx = &trunk_meta;
    struct gguf_context * trunk_ctx = gguf_init_from_file(params.trunk_file.c_str(), gguf_params);
    if (!trunk_ctx) { fprintf(stderr, "error: cannot open trunk %s\n", params.trunk_file.c_str()); exit(1); }
    std::ifstream f_trunk(params.trunk_file, std::ios::binary);

    // Open first expert to get tensor shapes and original expert count
    struct ggml_context * exp0_meta = nullptr;
    struct gguf_init_params ep = { true, nullptr };
    ep.ctx = &exp0_meta;
    struct gguf_context * exp0_ctx = gguf_init_from_file(params.expert_files[0].c_str(), ep);
    if (!exp0_ctx) { fprintf(stderr, "error: cannot open expert %s\n", params.expert_files[0].c_str()); exit(1); }

    int orig_n_expert_key = gguf_find_key(exp0_ctx, "moe_explode.original_expert_count");
    int orig_n_expert = orig_n_expert_key >= 0 ? (int)gguf_get_val_u32(exp0_ctx, orig_n_expert_key) : 0;
    printf("  Original expert count: %d\n", orig_n_expert);

    // Collect expert IDs
    std::vector<int> expert_ids;
    for (const auto & ef : params.expert_files) {
        struct ggml_context * em = nullptr;
        struct gguf_init_params p = { true, nullptr };
        p.ctx = &em;
        auto * ec = gguf_init_from_file(ef.c_str(), p);
        int id_key = gguf_find_key(ec, "moe_explode.expert_id");
        int eid = id_key >= 0 ? gguf_get_val_u32(ec, id_key) : -1;
        expert_ids.push_back(eid);
        printf("  %s → expert %d\n", ef.c_str(), eid);
        gguf_free(ec);
        ggml_free(em);
    }

    // Build output: start from trunk metadata
    auto * ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, trunk_ctx);

    std::string arch = find_arch(trunk_ctx);
    std::string ec_key = arch + ".expert_count";
    gguf_set_val_u32(ctx_out, ec_key.c_str(), n_experts_out);

    // Clamp expert_used_count
    std::string eu_key = arch + ".expert_used_count";
    int eu_idx = gguf_find_key(ctx_out, eu_key.c_str());
    if (eu_idx >= 0) {
        int eu_val = gguf_get_val_u32(ctx_out, eu_idx);
        if (eu_val > n_experts_out) {
            printf("  Clamping expert_used_count: %d → %d\n", eu_val, n_experts_out);
            gguf_set_val_u32(ctx_out, eu_key.c_str(), n_experts_out);
        }
    }

    int n_trunk_tensors = gguf_get_n_tensors(trunk_ctx);
    int n_exp_tensors = gguf_get_n_tensors(exp0_ctx);

    // Context for assembled expert tensors + resized router gates
    // Count router gates in trunk
    int n_trunk_router_gates = 0;
    for (int i = 0; i < n_trunk_tensors; i++) {
        if (is_router_gate(gguf_get_tensor_name(trunk_ctx, i))) n_trunk_router_gates++;
    }
    size_t ctx_size = (size_t)(n_exp_tensors + n_trunk_router_gates + 32) * ggml_tensor_overhead() + 4096;
    struct ggml_init_params gp = { ctx_size, nullptr, true };
    struct ggml_context * ctx_new = ggml_init(gp);

    // Add trunk tensors — router gates get resized to n_experts_out
    int n_router_gates = 0;
    for (int i = 0; i < n_trunk_tensors; i++) {
        const char * name = gguf_get_tensor_name(trunk_ctx, i);
        struct ggml_tensor * t = ggml_get_tensor(trunk_meta, name);
        if (is_router_gate(name)) {
            // Router gate: [n_embd, n_expert_original] → [n_embd, n_experts_out]
            int n_dims = ggml_n_dims(t);
            int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
            for (int d = 0; d < n_dims; d++) ne[d] = t->ne[d];
            ne[n_dims - 1] = n_experts_out;
            struct ggml_tensor * t_new = ggml_new_tensor(ctx_new, t->type, n_dims, ne);
            ggml_set_name(t_new, name);
            gguf_add_tensor(ctx_out, t_new);
            n_router_gates++;
        } else {
            gguf_add_tensor(ctx_out, t);
        }
    }

    // Add expert tensors: each expert file has single-expert slices where the
    // expert dimension collapsed (e.g. [2048, 768, 1] stored as 2D [2048, 768]).
    // We need to add a dimension back for the expert count.
    for (int i = 0; i < n_exp_tensors; i++) {
        const char * name = gguf_get_tensor_name(exp0_ctx, i);
        struct ggml_tensor * t = ggml_get_tensor(exp0_meta, name);
        int n_dims = ggml_n_dims(t);
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        for (int d = 0; d < n_dims; d++) ne[d] = t->ne[d];
        // Add expert dimension: original tensor had n_dims+1 with experts as outermost
        int out_dims = n_dims + 1;
        ne[n_dims] = n_experts_out;
        struct ggml_tensor * t_new = ggml_new_tensor(ctx_new, t->type, out_dims, ne);
        ggml_set_name(t_new, name);
        gguf_add_tensor(ctx_out, t_new);
    }

    if (params.dry_run) {
        printf("  (dry run, not writing)\n");
        gguf_free(ctx_out); ggml_free(ctx_new);
        gguf_free(trunk_ctx); ggml_free(trunk_meta);
        gguf_free(exp0_ctx); ggml_free(exp0_meta);
        return;
    }

    printf("  Writing %s ...\n", params.output.c_str());
    std::ofstream f_out(params.output, std::ios::binary);

    // Write metadata
    size_t meta_size = gguf_get_meta_size(ctx_out);
    std::vector<uint8_t> meta_data(meta_size);
    gguf_get_meta_data(ctx_out, meta_data.data());
    f_out.write((const char *)meta_data.data(), meta_size);

    // Write trunk tensor data — router gates gathered for selected experts
    std::vector<uint8_t> buf;
    std::vector<uint8_t> gather_buf;
    for (int i = 0; i < n_trunk_tensors; i++) {
        const char * name = gguf_get_tensor_name(trunk_ctx, i);
        int i_in = gguf_find_tensor(trunk_ctx, name);
        struct ggml_tensor * t = ggml_get_tensor(trunk_meta, name);
        size_t offset = gguf_get_data_offset(trunk_ctx) + gguf_get_tensor_offset(trunk_ctx, i_in);

        if (is_router_gate(name)) {
            // Gather selected expert rows from the full router gate
            int n_dims = ggml_n_dims(t);
            int64_t n_orig = t->ne[n_dims - 1]; // original expert count
            size_t bytes_per_expert = ggml_nbytes(t) / n_orig;
            size_t total_out = (size_t)n_experts_out * bytes_per_expert;

            buf.resize(ggml_nbytes(t));
            f_trunk.seekg(offset);
            f_trunk.read((char *)buf.data(), ggml_nbytes(t));

            gather_buf.resize(total_out);
            for (int e = 0; e < n_experts_out; e++) {
                int eid = expert_ids[e];
                if (eid >= 0 && eid < (int)n_orig) {
                    memcpy(gather_buf.data() + (size_t)e * bytes_per_expert,
                           buf.data() + (size_t)eid * bytes_per_expert,
                           bytes_per_expert);
                }
            }
            f_out.write((const char *)gather_buf.data(), total_out);
            zeros(f_out, GGML_PAD(total_out, GGUF_DEFAULT_ALIGNMENT) - total_out);
        } else {
            size_t n_bytes = ggml_nbytes(t);
            buf.resize(n_bytes);
            f_trunk.seekg(offset);
            f_trunk.read((char *)buf.data(), n_bytes);
            f_out.write((const char *)buf.data(), n_bytes);
            zeros(f_out, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);
        }
    }

    // Write expert tensor data: interleave slices from each expert file
    // For each expert tensor, read the single-expert slice from each file in order
    for (int ti = 0; ti < n_exp_tensors; ti++) {
        const char * name = gguf_get_tensor_name(exp0_ctx, ti);
        struct ggml_tensor * t0 = ggml_get_tensor(exp0_meta, name);
        size_t bytes_per_expert = ggml_nbytes(t0); // single expert slice

        buf.resize(bytes_per_expert);
        size_t total_written = 0;

        for (int e = 0; e < n_experts_out; e++) {
            // Open this expert's file, find tensor, read slice
            struct ggml_context * em = nullptr;
            struct gguf_init_params p = { true, nullptr };
            p.ctx = &em;
            auto * ec = gguf_init_from_file(params.expert_files[e].c_str(), p);
            int idx = gguf_find_tensor(ec, name);
            if (idx < 0) {
                fprintf(stderr, "error: tensor %s not found in %s\n", name, params.expert_files[e].c_str());
                exit(1);
            }
            size_t offset = gguf_get_data_offset(ec) + gguf_get_tensor_offset(ec, idx);
            std::ifstream f_exp(params.expert_files[e], std::ios::binary);
            f_exp.seekg(offset);
            f_exp.read((char *)buf.data(), bytes_per_expert);
            f_out.write((const char *)buf.data(), bytes_per_expert);
            total_written += bytes_per_expert;

            gguf_free(ec);
            ggml_free(em);
        }
        zeros(f_out, GGML_PAD(total_written, GGUF_DEFAULT_ALIGNMENT) - total_written);
    }

    f_out.close();
    auto out_sz = fs::file_size(params.output);
    printf("  Done. Output: %s (%.1f GB)\n", params.output.c_str(), out_sz / 1e9);

    gguf_free(ctx_out); ggml_free(ctx_new);
    gguf_free(trunk_ctx); ggml_free(trunk_meta);
    gguf_free(exp0_ctx); ggml_free(exp0_meta);
}

int main(int argc, const char ** argv) {
    explode_params params;
    parse_args(argc, argv, params);

    if (params.assemble) {
        do_assemble(params);
    } else {
        do_explode(params);
    }

    return 0;
}
