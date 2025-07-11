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
    (void)user_data; // Evitar aviso de não utilizado
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fprintf(stderr, "[LlamaLog] %s", text); // text já contém newline
        fflush(stderr); // Garantir que o log seja exibido imediatamente
    }
}

namespace cpu_llm_project {

LlmEngine::LlmEngine() {
    llama_log_set(LlmEngine_static_llama_log_callback, nullptr);
    llama_backend_init();
    std::cout << "LlmEngine: Inicializado. Backend Llama inicializado." << std::endl;
}

LlmEngine::~LlmEngine() {
    unload_model();
    llama_backend_free();
    std::cout << "LlmEngine: Destruído. Backend Llama liberado." << std::endl;
}

bool LlmEngine::load_model(const std::string& model_path, int n_ctx_req, int n_gpu_layers, int num_threads_param) {
    if (is_model_loaded()) {
        std::cerr << "LlmEngine::load_model: Modelo já carregado. Descarregue primeiro." << std::endl;
        return false;
    }

    model_path_ = model_path;
    n_ctx_ = n_ctx_req > 0 ? n_ctx_req : 2048;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    model_ = llama_load_model_from_file(model_path_.c_str(), model_params);

    if (!model_) {
        std::cerr << "LlmEngine::load_model: Falha ao carregar modelo de '" << model_path_ << "'." << std::endl;
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
    if (ctx_params.n_threads == 0) ctx_params.n_threads = 1; // Garantir pelo menos 1 thread
    if (ctx_params.n_threads_batch == 0) ctx_params.n_threads_batch = 1; // Garantir pelo menos 1 thread para batch

    ctx_ = llama_init_from_model(model_, ctx_params);

    if (!ctx_) {
        std::cerr << "LlmEngine::load_model: Falha ao criar contexto llama." << std::endl;
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    std::cout << "LlmEngine: Modelo e contexto carregados com sucesso. Tamanho do contexto: " << llama_n_ctx(ctx_) << std::endl;
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
    std::cout << "LlmEngine: Modelo descarregado." << std::endl;
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
        return "[Erro: Modelo não carregado]";
    }

    std::ostringstream oss_prompt;
    if (!system_prompt.empty()) {
        oss_prompt << system_prompt << "\n";
    }
    oss_prompt << "<start_of_turn>user\n"
               << user_prompt << "<end_of_turn>\n"
               << "<start_of_turn>model";

    std::string final_prompt_text = oss_prompt.str();
    const int current_n_ctx = llama_n_ctx(ctx_);

    std::vector<llama_token> prompt_tokens_vec(final_prompt_text.length() + 16); // Margem de segurança
    int n_prompt_tokens = llama_tokenize(
        model_, final_prompt_text.c_str(), (int32_t)final_prompt_text.length(),
        prompt_tokens_vec.data(), (int32_t)prompt_tokens_vec.size(),
        llama_vocab_get_add_bos(llama_get_vocab(model_)), // Usar vocab do modelo
        true // Processar tokens especiais
    );

    if (n_prompt_tokens < 0) {
        prompt_tokens_vec.resize(-n_prompt_tokens);
        n_prompt_tokens = llama_tokenize(
            model_, final_prompt_text.c_str(), (int32_t)final_prompt_text.length(),
            prompt_tokens_vec.data(), (int32_t)prompt_tokens_vec.size(),
            llama_vocab_get_add_bos(llama_get_vocab(model_)), true
        );
        if (n_prompt_tokens < 0) {
            std::cerr << "LlmEngine::predict: Falha ao tokenizar prompt (código " << n_prompt_tokens << ")." << std::endl;
            return "[Erro: Falha ao tokenizar prompt]";
        }
    }
    prompt_tokens_vec.resize(n_prompt_tokens);

    if (n_prompt_tokens >= current_n_ctx) {
        std::cerr << "LlmEngine::predict: Prompt muito longo (" << n_prompt_tokens
                  << " tokens) para o tamanho do contexto (" << current_n_ctx << ")." << std::endl;
        return "[Erro: Prompt muito longo para o contexto]";
    }

    llama_kv_cache_clear(ctx_); // Limpar cache KV para a nova sequência

    llama_batch batch = llama_batch_init(std::max(n_prompt_tokens, 1), 0, 1);

    for (int i = 0; i < n_prompt_tokens; ++i) {
        batch.token[batch.n_tokens] = prompt_tokens_vec[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = (i == n_prompt_tokens - 1); // Calcular logits apenas para o último token
        batch.n_tokens++;
    }

    if (batch.n_tokens == 0) {
         if (!user_prompt.empty()) {
             std::cerr << "LlmEngine::predict: Prompt tokenizado para zero tokens, embora user_prompt não estivesse vazio." << std::endl;
             llama_batch_free(batch);
             return "[Erro: Prompt tokenizado para zero tokens]";
         } else {
             llama_batch_free(batch);
             return ""; // Retornar string vazia para prompt vazio
         }
    }

    if (llama_decode(ctx_, batch) != 0) {
        std::cerr << "LlmEngine::predict: Falha no llama_decode para o prompt." << std::endl;
        llama_batch_free(batch);
        return "[Erro: Falha no llama_decode para o prompt]";
    }

    std::string result_text = "";
    int n_cur = n_prompt_tokens;
    int n_decoded = 0;

    // Usar llama_sampling_params e não llama_sampler_params
    llama_sampling_params sparams = llama_sampling_default_params();
    sparams.temp            = temp_param;
    sparams.top_k           = top_k_param <= 0 ? 0 : top_k_param; // 0 desabilita top_k
    sparams.top_p           = top_p_param;
    sparams.penalty_repeat  = repeat_penalty_param;
    sparams.penalty_last_n  = current_n_ctx > 0 ? std::min(current_n_ctx, 256) : 256;

    struct llama_sampler *sampler = llama_sampler_init(sparams);
    if (!sampler) {
        std::cerr << "LlmEngine::predict: Falha ao inicializar o sampler." << std::endl;
        llama_batch_free(batch);
        return "[Erro: Falha ao inicializar o sampler]";
    }

    for (int i = 0; i < n_prompt_tokens; ++i) {
        llama_sampler_accept(sampler, prompt_tokens_vec[i]);
    }

    batch.n_tokens = 0; // Limpar batch para geração de novos tokens

    while (n_cur < current_n_ctx && n_decoded < max_tokens_to_generate) {
        llama_token new_token_id = llama_sampler_sample(sampler, ctx_, 0); // idx=0 para o único logit no batch

        llama_sampler_accept(sampler, new_token_id);

        // Usar llama_token_eos(llama_get_vocab(model_))
        if (new_token_id == llama_token_eos(llama_get_vocab(model_))) {
            break;
        }

        char piece_buffer[64];
        // Usar llama_token_to_piece(llama_get_vocab(model_), ...)
        int len = llama_token_to_piece(llama_get_vocab(model_), new_token_id, piece_buffer, sizeof(piece_buffer));

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
            std::cerr << "LlmEngine::predict: Falha em llama_token_to_piece para o token " << new_token_id << ". Retornou: " << len << std::endl;
        }

        batch.n_tokens = 0; // Limpar batch para o próximo token
        batch.token[batch.n_tokens] = new_token_id;
        batch.pos[batch.n_tokens] = n_cur;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = true; // Calcular logits para o token amostrado
        batch.n_tokens++;

        n_decoded++;
        n_cur++;

        if (llama_decode(ctx_, batch) != 0) {
            std::cerr << "LlmEngine::predict: Falha no llama_decode para o token gerado " << new_token_id << std::endl;
            llama_sampler_free(sampler);
            llama_batch_free(batch);
            return result_text + "[Erro: Falha no llama_decode durante a geração]";
        }
    }

    llama_sampler_free(sampler);
    llama_batch_free(batch);
    return result_text;
}

} // namespace cpu_llm_project
