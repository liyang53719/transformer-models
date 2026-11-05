#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct dump_context {
    fs::path directory;
    std::string stage;
    std::ofstream index;
};

static bool selected(const std::string & name) {
    const auto split = name.rfind('-');
    if (split != std::string::npos) {
        const std::string suffix = name.substr(split + 1);
        if (suffix.empty() || !std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) { return std::isdigit(c); })) {
            return false;
        }
    }
    const std::string base = split == std::string::npos ? name : name.substr(0, split);
    const std::vector<std::string> names = {
        "attn_norm", "attn_residual", "attn_post_norm", "ffn_out", "l_out",
        "q_conv_predelta", "k_conv_predelta", "v_conv_predelta", "gate",
        "beta_sigmoid", "final_output", "linear_attn_out", "new_state",
        "Qcur", "Kcur", "Vcur", "attn_pregate", "gate_sigmoid", "attn_gated",
        "attn_output", "result_norm", "conv_state_last"
    };
    return std::find(names.begin(), names.end(), base) != names.end();
}

static void write_floats(const fs::path & file, const float * data, size_t count) {
    std::ofstream output(file, std::ios::binary);
    if (!output.write(reinterpret_cast<const char *>(data), count * sizeof(float))) {
        throw std::runtime_error("Cannot write " + file.string());
    }
}

static bool dump_tensor(ggml_tensor * tensor, bool ask, void * opaque) {
    if (!selected(tensor->name) || tensor->type != GGML_TYPE_F32) {
        return false;
    }
    if (ask) {
        return true;
    }
    auto & dump = *static_cast<dump_context *>(opaque);
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    std::vector<float> packed(ggml_nelements(tensor));
    size_t offset = 0;
    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                    const size_t source = i0*tensor->nb[0] + i1*tensor->nb[1] +
                                          i2*tensor->nb[2] + i3*tensor->nb[3];
                    std::memcpy(&packed[offset++], bytes.data() + source, sizeof(float));
                }
            }
        }
    }
    const fs::path relative = fs::path(dump.stage) / (std::string(tensor->name) + ".bin");
    write_floats(dump.directory / relative, packed.data(), packed.size());
    dump.index << dump.stage << '\t' << tensor->name;
    for (auto dimension : tensor->ne) {
        dump.index << '\t' << dimension;
    }
    dump.index << '\n';
    return true;
}

static int32_t argmax(const float * logits, int32_t vocabulary) {
    if (std::any_of(logits, logits + vocabulary, [](float x) { return !std::isfinite(x); })) {
        throw std::runtime_error("Nonfinite reference logits");
    }
    return static_cast<int32_t>(std::max_element(logits, logits + vocabulary) - logits);
}

int main(int argc, char ** argv) {
    if (argc < 5 || argc > 7) {
        std::cerr << "usage: qwen3_5_llama_reference MODEL OUTDIR PROMPT_FILE DECODE_STEPS [PREFILL_CHUNK] [--summary]\n";
        return 2;
    }
    try {
        const fs::path directory(argv[2]);
        fs::create_directories(directory);
        std::ifstream prompt_file(argv[3], std::ios::binary);
        if (!prompt_file) {
            throw std::runtime_error("Cannot read prompt file");
        }
        const std::string prompt((std::istreambuf_iterator<char>(prompt_file)), {});
        const int decode_steps = std::stoi(argv[4]);
        const int chunk = argc >= 6 ? std::stoi(argv[5]) : 0;
        const bool summary_mode = argc == 7;
        if (summary_mode && std::string(argv[6]) != "--summary") {
            throw std::runtime_error("Expected --summary after PREFILL_CHUNK");
        }
        if (decode_steps < 0 || chunk < 0) {
            throw std::runtime_error("Steps and chunk must be nonnegative");
        }
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        llama_model * model = llama_model_load_from_file(argv[1], mp);
        if (!model) {
            throw std::runtime_error("Cannot load reference model");
        }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const bool add_bos = llama_vocab_get_add_bos(vocab);
        const int capacity = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, add_bos, true);
        std::vector<llama_token> tokens(capacity);
        const int count = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), capacity, add_bos, true);
        if (count <= 0) {
            throw std::runtime_error("Empty or invalid prompt");
        }
        tokens.resize(count);
        {
            std::ofstream file(directory / "tokens.txt");
            for (auto token : tokens) {
                file << token << '\n';
            }
        }
        dump_context dump{directory, "", std::ofstream()};
        if (!summary_mode) {
            dump.index.open(directory / "tensors.tsv");
            dump.index << "stage\tname\td0\td1\td2\td3\n";
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = std::max(256, count + decode_steps + 64);
        cp.n_batch = std::max(32, count);
        cp.n_ubatch = cp.n_batch;
        cp.n_seq_max = 1;
        cp.n_threads = 8;
        cp.n_threads_batch = 8;
        cp.type_k = GGML_TYPE_F32;
        cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv = false;
        cp.op_offload = false;
        cp.cb_eval = summary_mode ? nullptr : dump_tensor;
        cp.cb_eval_user_data = &dump;
        llama_context * context = llama_init_from_model(model, cp);
        if (!context) {
            throw std::runtime_error("Cannot initialize reference context");
        }
        const int vocabulary = llama_vocab_n_tokens(vocab);
        std::ofstream generated(directory / "generated.txt");
        std::ofstream decoded(directory / "generated_utf8.txt", std::ios::binary);
        llama_batch batch = llama_batch_init(cp.n_batch, 0, 1);
        auto evaluate = [&](const std::vector<llama_token> & input, int position, const std::string & stage) {
            dump.stage = stage;
            fs::create_directories(directory / stage);
            batch.n_tokens = input.size();
            for (int i = 0; i < batch.n_tokens; ++i) {
                batch.token[i] = input[i];
                batch.pos[i] = position + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = !summary_mode || i == batch.n_tokens - 1;
            }
            if (llama_decode(context, batch) != 0) {
                throw std::runtime_error("llama_decode failed at " + stage);
            }
            llama_synchronize(context);
            if (summary_mode) {
                return argmax(llama_get_logits_ith(context, batch.n_tokens - 1), vocabulary);
            }
            std::vector<float> logits(static_cast<size_t>(vocabulary) * input.size());
            for (int i = 0; i < batch.n_tokens; ++i) {
                const float * row = llama_get_logits_ith(context, i);
                std::copy(row, row + vocabulary, logits.data() + static_cast<size_t>(i) * vocabulary);
            }
            write_floats(directory / stage / "logits.bin", logits.data(), logits.size());
            dump.index << stage << "\tlogits\t" << vocabulary << '\t' << input.size() << "\t1\t1\n";
            std::cout << "REFERENCE " << stage << " position=" << position << " tokens=" << input.size() << std::endl;
            return argmax(logits.data() + static_cast<size_t>(input.size()-1)*vocabulary, vocabulary);
        };
        int32_t next = 0;
        int batch_id = 0;
        const int batch_size = chunk ? chunk : count;
        const auto started = std::chrono::steady_clock::now();
        for (int position = 0; position < count; position += batch_size) {
            const int end = std::min(count, position + batch_size);
            const std::string stage = chunk ? "prefill_" + std::to_string(batch_id++) : "prefill";
            next = evaluate({tokens.begin() + position, tokens.begin() + end}, position, stage);
        }
        const auto prefill_done = std::chrono::steady_clock::now();
        int generated_count = 0;
        for (int i = 0; i < decode_steps; ++i) {
            generated << next << '\n';
            ++generated_count;
            std::vector<char> piece(1024);
            const int length = llama_token_to_piece(vocab, next, piece.data(), piece.size(), 0, true);
            if (length > 0) {
                decoded.write(piece.data(), length);
            }
            if (summary_mode && (llama_vocab_is_eog(vocab, next) || i + 1 == decode_steps)) {
                break;
            }
            next = evaluate({next}, count + i, "decode_" + std::to_string(i));
        }
        const auto finished = std::chrono::steady_clock::now();
        if (summary_mode) {
            const double prefill_seconds = std::chrono::duration<double>(prefill_done - started).count();
            const double decode_seconds = std::chrono::duration<double>(finished - prefill_done).count();
            std::ofstream metrics(directory / "generation_metrics.json");
            metrics.precision(12);
            metrics << "{\"prefill_seconds\":" << prefill_seconds
                    << ",\"decode_seconds\":" << decode_seconds
                    << ",\"generation_seconds\":" << prefill_seconds + decode_seconds
                    << ",\"generated_tokens\":" << generated_count << "}\n";
        }
        llama_batch_free(batch);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        std::cout << "REFERENCE_OK prompt_tokens=" << count << " decode_steps=" << decode_steps << std::endl;
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
