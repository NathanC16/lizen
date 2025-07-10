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

// Definição da função load_model atualizada para incluir num_threads
bool LlmEngine::load_model(const std::string& model_path, int n_ctx_req, int n_gpu_layers, int num_threads_param) {
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

    if (num_threads_param > 0) {
        ctx_params.n_threads = num_threads_param;
        ctx_params.n_threads_batch = num_threads_param; // Pode ser diferente, mas geralmente é o mesmo para CPU
        std::cout << "LlmEngine: Usando " << num_threads_param << " threads (definido pelo usuário)." << std::endl;
    } else {
        unsigned int hardware_threads = std::thread::hardware_concurrency();
        // Lógica padrão: usar no máximo 8 threads ou o total de threads de hardware, o que for menor. Mínimo de 4.
        ctx_params.n_threads = hardware_threads > 0 ? std::min(hardware_threads, 8U) : 4U;
        ctx_params.n_threads_batch = ctx_params.n_threads;
        std::cout << "LlmEngine: Usando " << ctx_params.n_threads << " threads (detectado automaticamente/padrão)." << std::endl;
    }

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

// Atualizada a assinatura para incluir system_prompt e os parâmetros de amostragem
std::string LlmEngine::predict(const std::string& user_prompt,
                               const std::string& system_prompt, // Novo parâmetro
                               int max_tokens_to_generate,
                               float temp_param,
                               int top_k_param,
                               float top_p_param,
                               float repeat_penalty_param) {
    if (!is_model_loaded()) {
        return "[Error: Model not loaded]";
    }

    // Marcar parâmetros de amostragem avançada como não utilizados POR ENQUANTO (serão usados quando a amostragem for restaurada)
    // (void)temp_param; // Será usado
    // (void)top_k_param; // Será usado
    // (void)top_p_param; // Será usado
    // (void)repeat_penalty_param; // Será usado

    // Formato de prompt específico para Gemma
    // Documentação de referência: https://ai.google.dev/gemma/docs/formatting
    std::ostringstream oss_prompt;
    if (!system_prompt.empty()) {
        // Embora Gemma não tenha um "system prompt" formal como outros modelos,
        // é comum prefixar instruções gerais antes do primeiro turno do usuário.
        // Poderíamos envolvê-lo em tokens de modelo se quiséssemos que parecesse uma instrução do sistema,
        // mas para Gemma, geralmente é apenas texto antes do primeiro <start_of_turn>user.
        // Vamos mantê-lo simples por enquanto e apenas prefixá-lo.
        // Outra abordagem seria:
        // oss_prompt << "<start_of_turn>model\n" << system_prompt << "<end_of_turn>\n";
        // Mas isso pode ser interpretado como um turno de diálogo do modelo.
        // Por enquanto, apenas texto simples antes do diálogo.
        oss_prompt << system_prompt << "\n"; // Adiciona uma nova linha para separação clara
    }
    oss_prompt << "<start_of_turn>user\n"
               << user_prompt << "<end_of_turn>\n"
               << "<start_of_turn>model"; // O modelo deve começar a gerar a partir daqui

    std::string final_prompt_text = oss_prompt.str();
    // std::cout << "Debug: Prompt Formatado para Gemma:\n\"" << final_prompt_text << "\"" << std::endl;


    const int current_n_ctx = llama_n_ctx(ctx_);
    bool add_bos = llama_vocab_type(llama_model_get_vocab(model_)) == LLAMA_VOCAB_TYPE_SPM;

    std::vector<llama_token> prompt_tokens_vec(current_n_ctx);
    // Tokenizar o final_prompt_text
    int n_prompt_tokens = llama_tokenize(
        llama_model_get_vocab(model_), final_prompt_text.c_str(), (int32_t)final_prompt_text.length(),
        prompt_tokens_vec.data(), (int32_t)prompt_tokens_vec.size(), add_bos, true
    );

    if (n_prompt_tokens < 0) {
        int required_size = -n_prompt_tokens;
        prompt_tokens_vec.resize(required_size);
        // Corrigir para usar final_prompt_text na segunda tentativa também
        n_prompt_tokens = llama_tokenize(
            llama_model_get_vocab(model_), final_prompt_text.c_str(), (int32_t)final_prompt_text.length(),
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
         if (!user_prompt.empty()) { // Verificar user_prompt em vez de prompt_text
             std::cerr << "LlmEngine::predict: Prompt tokenized to zero tokens, though user_prompt was not empty." << std::endl;
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

    std::string result_text = "";
    int n_cur = batch.n_tokens; // n_cur é a posição atual na sequência
    int n_decoded = 0;

    // Histórico de tokens para penalidades
    std::vector<llama_token> penalty_tokens;
    penalty_tokens.reserve(current_n_ctx);
    for(int i = 0; i < n_prompt_tokens; ++i) {
        penalty_tokens.push_back(prompt_tokens_vec[i]);
    }
    const size_t penalty_max_size = current_n_ctx > 0 ? (size_t)current_n_ctx : 512;

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_));

    while (n_cur < current_n_ctx && n_decoded < max_tokens_to_generate) {
        float *logits = llama_get_logits_ith(ctx_, batch.n_tokens - 1);

        std::vector<llama_token_data> candidates_vec_loop;
        candidates_vec_loop.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; ++token_id) {
            candidates_vec_loop.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }
        llama_token_data_array candidates_p = { candidates_vec_loop.data(), candidates_vec_loop.size(), false };

        // Restaurar lógica de amostragem avançada
        if (!penalty_tokens.empty()) {
             llama_sample_repetition_penalty(ctx_, &candidates_p,
                                          penalty_tokens.data(),
                                          penalty_tokens.size(),
                                          repeat_penalty_param);
        }
        llama_sample_top_k(ctx_, &candidates_p, top_k_param <= 0 ? llama_vocab_n_tokens(llama_model_get_vocab(model_)) : top_k_param, 1);
        llama_sample_top_p(ctx_, &candidates_p, top_p_param, 1);
        llama_sample_temp(ctx_, &candidates_p, temp_param);
        // A função llama_sample_token_greedy foi removida em versões mais recentes de llama.cpp
        // Em vez disso, usa-se llama_sample_token que aplica a amostragem definida anteriormente.
        // Se llama_sample_token_greedy fosse estritamente necessária, seria preciso verificar a versão do llama.cpp
        // e possivelmente reimplementar uma lógica greedy simples se as outras funções de sample modificam os logits para softmax.
        // Por agora, vamos usar llama_sample_token, que é o padrão para aplicar as transformações de amostragem.
        llama_token new_token_id = llama_sample_token(ctx_, &candidates_p);


        penalty_tokens.push_back(new_token_id);
        if (penalty_tokens.size() > penalty_max_size) {
            penalty_tokens.erase(penalty_tokens.begin(), penalty_tokens.begin() + (penalty_tokens.size() - penalty_max_size));
        }

        if (new_token_id == llama_token_eos(llama_model_get_vocab(model_))) {
            std::cout << " [EOS]" << std::endl;
            break;
        }

        char piece_buffer[64];
        int len = llama_token_to_piece(llama_model_get_vocab(model_), new_token_id, piece_buffer, sizeof(piece_buffer), 0, false);

        if (len > 0) {
            result_text.append(piece_buffer, len);
            // std::cout << std::string(piece_buffer, len); // Imprimir token por token (opcional)
            // fflush(stdout);

            // Verificar sequências de parada
            // Definir um tamanho mínimo para evitar verificações em strings muito curtas
            const size_t min_len_for_stop_check = 6; // Ex: " USER:"
            if (result_text.length() >= min_len_for_stop_check) {
                const std::vector<std::string> stop_sequences = {
                    "\nUSER:", "\nASSISTANT:", " USER:", " ASSISTANT:"
                    // Adicionar "<|user|>", "<|assistant|>" ou outros tokens especiais se o modelo usar
                };
                bool stopped = false;
                for (const auto& seq : stop_sequences) {
                    if (result_text.rfind(seq) == result_text.length() - seq.length()) {
                        std::cout << " [Stop sequence detected: '" << seq << "']" << std::endl;
                        // Remover a sequência de parada da result_text
                        result_text.erase(result_text.length() - seq.length());
                        stopped = true;
                        break;
                    }
                }
                if (stopped) {
                    break; // Sair do loop de geração
                }
            }

        } else if (len < 0) {
            std::cerr << "LlmEngine::predict: llama_token_to_piece failed for token " << new_token_id << ". Returned: " << len << std::endl;
        }

        batch.n_tokens = 0;
        batch.token   [batch.n_tokens] = new_token_id;
        batch.pos     [batch.n_tokens] = n_cur;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id  [batch.n_tokens][0] = 0;
        batch.logits  [batch.n_tokens] = true;
        batch.n_tokens++;

        n_decoded++;
        n_cur++;

        if (llama_decode(ctx_, batch) != 0) {
            std::cerr << "LlmEngine::predict: llama_decode failed for generated token " << new_token_id << std::endl;
            llama_batch_free(batch);
            return result_text + "[Error: llama_decode failed during generation]";
        }
    }

    llama_batch_free(batch);
    // return "[INFO: Text generation loop disabled for compilation. Processed prompt.]";
    return result_text;
}

} // namespace cpu_llm_project
