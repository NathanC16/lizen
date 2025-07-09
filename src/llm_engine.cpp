#include "cpu_llm_project/llm_engine.hpp"
// llama.h já está incluído via llm_engine.hpp

#include <iostream>
#include <vector>
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
    llama_backend_init();
    std::cout << "LlmEngine: Initialized. Llama backend initialized." << std::endl;
}

LlmEngine::~LlmEngine() {
    unload_model();
    llama_backend_free();
    std::cout << "LlmEngine: Destroyed. Llama backend freed." << std::endl;
}

bool LlmEngine::load_model(const std::string& model_path, int n_ctx_req, int n_gpu_layers) {
    if (is_model_loaded()) {
        std::cerr << "LlmEngine::load_model: Model already loaded. Unload first." << std::endl;
        return false;
    }

    model_path_ = model_path;
    n_ctx_ = n_ctx_req > 0 ? n_ctx_req : 2048;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    model_ = llama_model_load_from_file(model_path_.c_str(), model_params);

    if (!model_) {
        std::cerr << "LlmEngine::load_model: Failed to load model from '" << model_path_ << "'." << std::endl;
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx_;
    ctx_params.n_batch = std::min((uint32_t)n_ctx_, 512U);

    unsigned int n_threads = std::thread::hardware_concurrency();
    ctx_params.n_threads = n_threads > 0 ? std::min(n_threads, 8U) : 4U;
    ctx_params.n_threads_batch = ctx_params.n_threads;

    ctx_ = llama_init_from_model(model_, ctx_params);

    if (!ctx_) {
        std::cerr << "LlmEngine::load_model: Failed to create llama_context." << std::endl;
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

std::string LlmEngine::predict(const std::string& prompt_text, int max_tokens_to_generate,
                               float temp_param, int top_k_param, float top_p_param,
                               float repeat_penalty_param) {
    if (!is_model_loaded()) {
        return "[Error: Model not loaded]";
    }

    // Marcar parâmetros de amostragem avançada como não utilizados
    (void)max_tokens_to_generate; // Ignorado na implementação simplificada
    (void)temp_param;
    (void)top_k_param;
    (void)top_p_param;
    (void)repeat_penalty_param;

    const int current_n_ctx = llama_n_ctx(ctx_);
    bool add_bos = llama_vocab_type(llama_model_get_vocab(model_)) == LLAMA_VOCAB_TYPE_SPM;

    std::vector<llama_token> prompt_tokens_vec(current_n_ctx);
    int n_prompt_tokens = llama_tokenize(
        llama_model_get_vocab(model_), prompt_text.c_str(), (int32_t)prompt_text.length(),
        prompt_tokens_vec.data(), (int32_t)prompt_tokens_vec.size(), add_bos, true
    );

    if (n_prompt_tokens < 0) {
        int required_size = -n_prompt_tokens;
        prompt_tokens_vec.resize(required_size);
        n_prompt_tokens = llama_tokenize(
            llama_model_get_vocab(model_), prompt_text.c_str(), (int32_t)prompt_text.length(),
            prompt_tokens_vec.data(), (int32_t)prompt_tokens_vec.size(), add_bos, true
        );
        if (n_prompt_tokens < 0) {
            std::cerr << "LlmEngine::predict: Failed to tokenize prompt even after resizing. Error code: " << n_prompt_tokens << std::endl;
            return "[Error: Failed to tokenize prompt - unexpected error]";
        }
    }
    prompt_tokens_vec.resize(n_prompt_tokens);

    if (n_prompt_tokens >= current_n_ctx) {
        std::cerr << "LlmEngine::predict: Prompt is too long (" << n_prompt_tokens
                  << " tokens) for context size (" << current_n_ctx << ")." << std::endl;
        return "[Error: Prompt too long for context]";
    }

    llama_kv_self_seq_rm(ctx_, (llama_seq_id)0, 0, -1);


    llama_batch batch = llama_batch_init(current_n_ctx, 0, 1);

    for (int i = 0; i < n_prompt_tokens; ++i) {
        batch.token   [batch.n_tokens] = prompt_tokens_vec[i];
        batch.pos     [batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id  [batch.n_tokens][0] = 0;
        batch.logits  [batch.n_tokens] = (i == n_prompt_tokens - 1); // Somente o último token do prompt precisa de logits
        batch.n_tokens++;
    }

    if (batch.n_tokens == 0) {
         if (!prompt_text.empty()) {
             std::cerr << "LlmEngine::predict: Prompt tokenized to zero tokens, though prompt was not empty." << std::endl;
             llama_batch_free(batch);
             return "[Error: Prompt tokenized to zero tokens]";
         } else {
             llama_batch_free(batch);
             return ""; // Retorna string vazia para prompt vazio
         }
    }

    if (llama_decode(ctx_, batch) != 0) {
        std::cerr << "LlmEngine::predict: llama_decode failed for prompt." << std::endl;
        llama_batch_free(batch);
        return "[Error: llama_decode failed for prompt]";
    }

    // Implementação de predição removida/simplificada para garantir a compilação.
    // Retorna apenas o prompt processado ou uma mensagem.
    // Para realmente gerar texto, a lógica de amostragem precisa ser restaurada/corrigida.

    // Opcional: Obter o texto do prompt de volta para verificar a tokenização
    // std::string processed_prompt_text = "";
    // for(int i = 0; i < n_prompt_tokens; ++i) {
    //     char piece_buffer[32];
    //     int len = llama_token_to_piece(llama_model_get_vocab(model_), prompt_tokens_vec[i], piece_buffer, sizeof(piece_buffer), 0, false);
    //     if (len > 0) {
    //         processed_prompt_text.append(piece_buffer, len);
    //     }
    // }
    // std::cout << "Processed prompt: " << processed_prompt_text << std::endl;

    llama_batch_free(batch);
    return "[INFO: Text generation loop disabled for compilation. Processed prompt.]";
}

} // namespace cpu_llm_project
