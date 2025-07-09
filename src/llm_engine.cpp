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
    // llama_numa_init(GGML_NUMA_STRATEGY_DISABLED); // Opcional
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

    std::cout << "LlmEngine: Loading model from " << model_path_ << "..." << std::endl;
    model_ = llama_load_model_from_file(model_path_.c_str(), model_params);

    if (!model_) {
        std::cerr << "LlmEngine::load_model: Failed to load model from '" << model_path_ << "'." << std::endl;
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx_;
    ctx_params.n_batch = std::min((uint32_t)n_ctx_, 512U); // n_batch é uint32_t, fazer cast em n_ctx_

    // Configurar threads de forma mais conservadora
    unsigned int n_threads = std::thread::hardware_concurrency();
    ctx_params.n_threads = n_threads > 0 ? std::min(n_threads, 8U) : 4U; // Limitar a 8 ou usar 4
    ctx_params.n_threads_batch = ctx_params.n_threads;


    ctx_ = llama_new_context_with_model(model_, ctx_params);

    if (!ctx_) {
        std::cerr << "LlmEngine::load_model: Failed to create llama_context." << std::endl;
        llama_free_model(model_);
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
        llama_free_model(model_);
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

    const int current_n_ctx = llama_n_ctx(ctx_);

    // `add_bos` é true por padrão se o modelo não especificar o contrário.
    // A função `llama_tokenize` com `llama_model*` lida com isso internamente.
    bool add_bos = true;
    // Alternativamente, para ser mais explícito com modelos que *não* querem BOS:
    // if (llama_model_meta_val_str(model_, LLAMA_MODEL_META_ADD_BOS_TOKEN, add_bos_str_val, sizeof(add_bos_str_val)) == sizeof(add_bos_str_val) -1) {
    //    if (strcmp(add_bos_str_val, "false") == 0) add_bos = false;
    // }
    // Ou, mais simples, se a API `llama_model_add_bos_token` estiver disponível e for confiável:
    // add_bos = llama_model_add_bos_token(model_);


    std::vector<llama_token> prompt_tokens(current_n_ctx);
    int n_prompt_tokens = llama_tokenize(
        model_, prompt_text.c_str(), prompt_text.length(),
        prompt_tokens.data(), prompt_tokens.size(), add_bos, true /* special tokens */
    );

    if (n_prompt_tokens < 0) {
        // Se n_prompt_tokens for negativo, -(n_prompt_tokens) é o tamanho necessário.
        // Aqui, estamos assumindo que current_n_ctx é grande o suficiente.
        // Se o problema for realmente o buffer `prompt_tokens.size()`, então o erro é diferente.
        std::cerr << "LlmEngine::predict: Failed to tokenize prompt. Error code or required size: " << n_prompt_tokens << std::endl;
        return "[Error: Failed to tokenize prompt - possibly too long for internal buffers or other error]";
    }
    prompt_tokens.resize(n_prompt_tokens);

    if (n_prompt_tokens >= current_n_ctx) {
        std::cerr << "LlmEngine::predict: Prompt is too long (" << n_prompt_tokens
                  << " tokens) for context size (" << current_n_ctx << ")." << std::endl;
        return "[Error: Prompt too long for context]";
    }

    llama_kv_cache_clear(ctx_); // Limpa todo o KV cache do contexto

    // O tamanho do batch para llama_batch_init deve ser capaz de conter o prompt e os tokens gerados.
    // No entanto, para a decodificação token a token, o n_tokens no batch só precisa ser 1 para o token atual.
    // O primeiro argumento de llama_batch_init é o tamanho máximo do batch.
    // O tamanho máximo do batch é o tamanho do contexto.
    // n_seq_max é 1 porque estamos processando uma única sequência.
    llama_batch batch = llama_batch_init(current_n_ctx, 0, 1);

    // Adicionar o prompt ao batch
    for (int i = 0; i < n_prompt_tokens; ++i) {
        // A verificação de batch.n_alloc não é necessária aqui, pois llama_batch_init
        // já alocou espaço suficiente para current_n_ctx tokens.
        // Se n_prompt_tokens > current_n_ctx, já teríamos retornado um erro.
        batch.token   [batch.n_tokens] = prompt_tokens[i];
        batch.pos     [batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id  [batch.n_tokens][0] = 0; // ID da sequência 0
        batch.logits  [batch.n_tokens] = (i == n_prompt_tokens - 1); // Logits apenas para o último token do prompt
        batch.n_tokens++;
    }

    if (batch.n_tokens == 0) { // Se o prompt era vazio ou não tokenizou para nada
         if (!prompt_text.empty()) {
             std::cerr << "LlmEngine::predict: Prompt tokenized to zero tokens, though prompt was not empty." << std::endl;
             llama_batch_free(batch);
             return "[Error: Prompt tokenized to zero tokens]";
         } else { // Prompt realmente vazio
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
    int n_cur = batch.n_tokens; // Posição atual na sequência é o número de tokens do prompt
    int n_decoded = 0;

    std::vector<llama_token> last_n_tokens_ring_buffer(current_n_ctx, 0);
    if (n_prompt_tokens > 0) {
         std::copy(prompt_tokens.begin(), prompt_tokens.end(),
                   last_n_tokens_ring_buffer.end() - n_prompt_tokens);
    }

    const int n_vocab = llama_n_vocab(model_);

    while (n_cur < current_n_ctx && n_decoded < max_tokens_to_generate) {
        // Obter logits para o último token processado no batch
        float *logits = llama_get_logits_ith(ctx_, batch.n_tokens - 1);

        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; ++token_id) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }
        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

        // Aplicar samplers
        llama_sample_repetition_penalties(ctx_, &candidates_p,
                                           last_n_tokens_ring_buffer.data(),
                                           last_n_tokens_ring_buffer.size(),
                                           repeat_penalty_param, 1.0f, 1.0f);
        llama_sample_top_k(ctx_, &candidates_p, top_k_param <= 0 ? n_vocab : top_k_param, 1);
        llama_sample_top_p(ctx_, &candidates_p, top_p_param, 1);
        llama_sample_temp (ctx_, &candidates_p, temp_param);

        llama_token new_token_id = llama_sample_token(ctx_, &candidates_p);

        if (new_token_id == llama_token_eos(model_)) {
            break;
        }

        char piece_buffer[32];
        // O último argumento de llama_token_to_piece é `special`:
        // Para a saída do LLM, geralmente queremos `false` para não imprimir tokens como <eos>
        int len = llama_token_to_piece(model_, new_token_id, piece_buffer, sizeof(piece_buffer), false);
        if (len > 0) {
            result_text.append(piece_buffer, len);
        } else if (len < 0) {
            std::cerr << "LlmEngine::predict: llama_token_to_piece failed for token " << new_token_id << std::endl;
        }

        if (!last_n_tokens_ring_buffer.empty()) {
            last_n_tokens_ring_buffer.erase(last_n_tokens_ring_buffer.begin());
            last_n_tokens_ring_buffer.push_back(new_token_id);
        }

        // Preparar para o próximo ciclo de decodificação
        batch.n_tokens = 0; // Limpar o batch para o próximo token

        // Adicionar o novo token ao batch para a próxima iteração
        // Não precisamos verificar n_alloc aqui, pois o batch foi inicializado com current_n_ctx,
        // e o loop while (n_cur < current_n_ctx) garante que não excederemos.
        batch.token   [batch.n_tokens] = new_token_id;
        batch.pos     [batch.n_tokens] = n_cur; // Posição atual na sequência
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id  [batch.n_tokens][0] = 0;
        batch.logits  [batch.n_tokens] = true; // Queremos logits para este novo token
        batch.n_tokens++;

        n_decoded++;
        n_cur++;

        if (llama_decode(ctx_, batch) != 0) {
            std::cerr << "LlmEngine::predict: llama_decode failed for generated token " << new_token_id << std::endl;
            llama_batch_free(batch); // Liberar batch em caso de erro aqui
            return result_text + "[Error: llama_decode failed during generation]";
        }
    }

    llama_batch_free(batch);
    return result_text;
}

} // namespace cpu_llm_project
