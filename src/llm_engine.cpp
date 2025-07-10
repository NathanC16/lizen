#include "cpu_llm_project/llm_engine.hpp"
// llama.h já está incluído via llm_engine.hpp

#include <iostream>
#include <vector>
#include <sstream> // Para std::ostringstream
#include <stdexcept>
#include <algorithm> // Para std::fill, std::min, std::copy
#include <string.h>  // Para memcpy
#include <thread>    // Para std::thread::hardware_concurrency()

// Callback de log do llama.cpp
static void LlmEngine_static_llama_log_callback(ggml_log_level level, const char *text, void *user_data) {
    (void)user_data;
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fprintf(stderr, "[LlamaLog] %s", text); // text já contém newline
        fflush(stderr);
    }
}

namespace cpu_llm_project {

LlmEngine::LlmEngine() {
    llama_log_set(LlmEngine_static_llama_log_callback, nullptr);
    llama_backend_init(); // Inicializa o backend do llama.cpp (necessário uma vez)
    // Não inicializar n_threads aqui, pois depende da configuração do modelo
    std::cout << "LlmEngine: Initialized. Llama backend initialized." << std::endl;
}

LlmEngine::~LlmEngine() {
    unload_model();
    llama_backend_free(); // Libera o backend do llama.cpp
    std::cout << "LlmEngine: Destroyed. Llama backend freed." << std::endl;
}

bool LlmEngine::load_model(const std::string& model_path, int n_ctx_req, int n_gpu_layers, int num_threads_param) {
    if (is_model_loaded()) {
        std::cerr << "LlmEngine::load_model: Model already loaded. Unload first." << std::endl;
        return false;
    }

    model_path_ = model_path;
    n_ctx_ = n_ctx_req > 0 ? n_ctx_req : 2048; // Valor padrão se n_ctx_req for 0 ou negativo

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers; // Suporte a GPU (0 para CPU)

    model_ = llama_load_model_from_file(model_path_.c_str(), model_params);

    if (!model_) {
        std::cerr << "LlmEngine::load_model: Failed to load model from '" << model_path_ << "'." << std::endl;
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx_;
    ctx_params.n_batch = 512;

    if (num_threads_param > 0) {
        ctx_params.n_threads = num_threads_param;
        ctx_params.n_threads_batch = num_threads_param;
        std::cout << "LlmEngine: Usando " << num_threads_param << " threads (definido pelo usuário)." << std::endl;
    } else {
        unsigned int hardware_threads = std::thread::hardware_concurrency();
        ctx_params.n_threads = hardware_threads > 0 ? std::min(hardware_threads, 8U) : 4U;
        ctx_params.n_threads_batch = ctx_params.n_threads;
        std::cout << "LlmEngine: Usando " << ctx_params.n_threads << " threads (detectado automaticamente/padrão)." << std::endl;
    }
    if (ctx_params.n_threads == 0) ctx_params.n_threads = 1;
    if (ctx_params.n_threads_batch == 0) ctx_params.n_threads_batch = 1;

    // Usar llama_init_from_model em vez de llama_new_context_with_model
    ctx_ = llama_init_from_model(model_, ctx_params);

    if (!ctx_) {
        std::cerr << "LlmEngine::load_model: Failed to create llama_context." << std::endl;
        // Usar llama_model_free em vez de llama_free_model
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    std::cout << "LlmEngine: Model and context loaded successfully. Context size: " << llama_n_ctx(ctx_) << std::endl;
    return true;
}

void LlmEngine::unload_model() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        // Usar llama_model_free em vez de llama_free_model
        llama_model_free(model_);
        model_ = nullptr;
    }
    model_path_.clear();
    n_ctx_ = 0;
    std::cout << "LlmEngine: Model unloaded." << std::endl;
}

bool LlmEngine::is_model_loaded() const {
    return model_ != nullptr && ctx_ != nullptr;
}

std::string LlmEngine::get_model_path() const {
    return model_path_;
}

std::string LlmEngine::predict(const std::string& user_prompt,
                               const std::string& system_prompt,
                               int max_tokens_to_generate,
                               float temp_param,
                               int top_k_param,
                               float top_p_param,
                               float repeat_penalty_param) {
    if (!is_model_loaded()) {
        return "[Error: Model not loaded]";
    }

    std::ostringstream oss_prompt;
    if (!system_prompt.empty()) {
        oss_prompt << system_prompt << "\n";
    }
    oss_prompt << "<start_of_turn>user\n"
               << user_prompt << "<end_of_turn>\n"
               << "<start_of_turn>model"; // O modelo deve começar a gerar a partir daqui

    std::string final_prompt_text = oss_prompt.str();
    const int current_n_ctx = llama_n_ctx(ctx_);

    // Tokenizar o prompt
    std::vector<llama_token> prompt_tokens_vec(final_prompt_text.length() + 16);
    int n_prompt_tokens = llama_tokenize(
        model_, final_prompt_text.c_str(), (int32_t)final_prompt_text.length(),
        prompt_tokens_vec.data(), (int32_t)prompt_tokens_vec.size(),
        llama_model_should_add_bos_token(model_), // Correção: Usar llama_model_should_add_bos_token
        true
    );

    if (n_prompt_tokens < 0) {
        prompt_tokens_vec.resize(-n_prompt_tokens);
        n_prompt_tokens = llama_tokenize(
            model_, final_prompt_text.c_str(), (int32_t)final_prompt_text.length(),
            prompt_tokens_vec.data(), (int32_t)prompt_tokens_vec.size(),
            llama_model_should_add_bos_token(model_), true // Correção
        );
        if (n_prompt_tokens < 0) {
            std::cerr << "LlmEngine::predict: Failed to tokenize prompt (code " << n_prompt_tokens << ")." << std::endl;
            return "[Error: Failed to tokenize prompt]";
        }
    }
    prompt_tokens_vec.resize(n_prompt_tokens);

    if (n_prompt_tokens >= current_n_ctx) {
        std::cerr << "LlmEngine::predict: Prompt is too long (" << n_prompt_tokens
                  << " tokens) for context size (" << current_n_ctx << ")." << std::endl;
        return "[Error: Prompt too long for context]";
    }

    // Correção: Usar llama_kv_cache_clear(ctx_) para limpar todo o cache KV
    llama_kv_cache_clear(ctx_);

    llama_batch batch = llama_batch_init(std::max(n_prompt_tokens, 1), 0, 1);

    for (int i = 0; i < n_prompt_tokens; ++i) {
        // Correção: Usar llama_batch_add para adicionar token
        llama_batch_add(batch, prompt_tokens_vec[i], i, {0}, (i == n_prompt_tokens - 1));
    }

    if (batch.n_tokens == 0) {
         if (!user_prompt.empty()) {
             std::cerr << "LlmEngine::predict: Prompt tokenized to zero tokens, though user_prompt was not empty." << std::endl;
             llama_batch_free(batch);
             return "[Error: Prompt tokenized to zero tokens]";
         } else {
             llama_batch_free(batch);
             return "";
         }
    }

    if (llama_decode(ctx_, batch) != 0) {
        std::cerr << "LlmEngine::predict: llama_decode failed for prompt." << std::endl;
        llama_batch_free(batch);
        return "[Error: llama_decode failed for prompt]";
    }

    std::string result_text = "";
    int n_cur = n_prompt_tokens;
    int n_decoded = 0;

    // Usar nomes de API de sampler conforme sugestões
    llama_sampling_params sparams = llama_sampling_default_params();
    sparams.temp            = temp_param;
    sparams.top_k           = top_k_param <= 0 ? 0 : top_k_param;
    sparams.top_p           = top_p_param;
    sparams.penalty_repeat  = repeat_penalty_param;
    sparams.penalty_last_n  = current_n_ctx > 0 ? std::min(current_n_ctx, 256) : 256;

    // Correção: Usar llama_sampler_init e tipo llama_sampling_context* (se este for o tipo retornado)
    // Se llama_sampler_init retorna um tipo diferente, precisaremos ajustar sampling_ctx.
    // Assumindo que llama_sampling_context* ainda é o tipo, apesar das sugestões de nome de função.
    // Se o tipo for realmente llama_sampler_context_t, precisaremos mudar a declaração.
    // Por ora, vamos tentar com os nomes de função corrigidos.
    struct llama_sampling_context * sampling_ctx = llama_sampler_init(sparams);
    if (!sampling_ctx) {
        std::cerr << "LlmEngine::predict: Failed to initialize sampling context." << std::endl;
        llama_batch_free(batch);
        return "[Error: Failed to initialize sampling context]";
    }

    for (int i = 0; i < n_prompt_tokens; ++i) {
        // Correção: Usar llama_sampler_accept
        llama_sampler_accept(sampling_ctx, ctx_, prompt_tokens_vec[i], false);
    }

    // Correção: Limpar batch com batch.n_tokens = 0
    batch.n_tokens = 0;

    while (n_cur < current_n_ctx && n_decoded < max_tokens_to_generate) {
        // Correção: Usar llama_sampler_sample
        llama_token new_token_id = llama_sampler_sample(sampling_ctx, ctx_, nullptr); // idx não é mais necessário aqui

        // Correção: Usar llama_sampler_accept
        llama_sampler_accept(sampling_ctx, ctx_, new_token_id, true);

        // Correção: Usar llama_model_token_eos
        if (new_token_id == llama_model_token_eos(model_)) {
            break;
        }

        char piece_buffer[64];
        // Correção: Usar llama_model_token_to_piece
        int len = llama_model_token_to_piece(model_, new_token_id, piece_buffer, sizeof(piece_buffer));

        if (len > 0) {
            result_text.append(piece_buffer, len);
            const size_t min_len_for_stop_check = 6;
            const std::vector<std::string> stop_sequences = {
                "\nUSER:", "\nASSISTANT:", " USER:", " ASSISTANT:", "<end_of_turn>"
            };
            if (result_text.length() >= min_len_for_stop_check) {
                bool stopped = false;
                for (const auto& seq : stop_sequences) {
                    if (result_text.length() >= seq.length() &&
                        result_text.rfind(seq) == result_text.length() - seq.length()) {
                        result_text.erase(result_text.length() - seq.length());
                        stopped = true;
                        break;
                    }
                }
                if (stopped) {
                    break;
                }
            }
        } else if (len < 0) {
            std::cerr << "LlmEngine::predict: llama_token_to_piece failed for token " << new_token_id << ". Returned: " << len << std::endl;
        }

        // Correção: Limpar batch com batch.n_tokens = 0
        batch.n_tokens = 0;
        // Correção: Usar llama_batch_add
        llama_batch_add(batch, new_token_id, n_cur, {0}, true);

        n_decoded++;
        n_cur++;

        if (llama_decode(ctx_, batch) != 0) {
            std::cerr << "LlmEngine::predict: llama_decode failed for generated token " << new_token_id << std::endl;
            llama_sampler_free(sampling_ctx); // Correção
            llama_batch_free(batch);
            return result_text + "[Error: llama_decode failed during generation]";
        }
    }

    llama_sampler_free(sampling_ctx); // Correção
    llama_batch_free(batch);
    return result_text;
}

} // namespace cpu_llm_project
>>>>>>> REPLACE
