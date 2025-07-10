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
    // ctx_params.n_batch = std::min((uint32_t)n_ctx_, 512U); // Um valor comum para n_batch
    ctx_params.n_batch = 512; // Default n_batch from llama.cpp, can be overridden by user.
                               // Let's keep it simple, or allow this to be configurable.
                               // Needs to be <= n_ctx.

    if (num_threads_param > 0) {
        ctx_params.n_threads = num_threads_param;
        ctx_params.n_threads_batch = num_threads_param; // Para CPU, geralmente são iguais
        std::cout << "LlmEngine: Usando " << num_threads_param << " threads (definido pelo usuário)." << std::endl;
    } else {
        unsigned int hardware_threads = std::thread::hardware_concurrency();
        // Lógica padrão: usar no máximo 8 threads ou o total de threads de hardware, o que for menor. Mínimo de 4.
        // Esta lógica pode ser ajustada conforme necessidade.
        ctx_params.n_threads = hardware_threads > 0 ? std::min(hardware_threads, 8U) : 4U;
        ctx_params.n_threads_batch = ctx_params.n_threads;
        std::cout << "LlmEngine: Usando " << ctx_params.n_threads << " threads (detectado automaticamente/padrão)." << std::endl;
    }
     // Ensure n_threads and n_threads_batch are at least 1
    if (ctx_params.n_threads == 0) ctx_params.n_threads = 1;
    if (ctx_params.n_threads_batch == 0) ctx_params.n_threads_batch = 1;


    ctx_ = llama_new_context_with_model(model_, ctx_params);

    if (!ctx_) {
        std::cerr << "LlmEngine::load_model: Failed to create llama_context." << std::endl;
        llama_free_model(model_); // Use llama_free_model para o objeto model_
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
        llama_free_model(model_); // Use llama_free_model para o objeto model_
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
    // A função llama_tokenize agora aceita llama_model* diretamente.
    // Adicionar um pequeno buffer para segurança, ou calcular o tamanho máximo possível.
    std::vector<llama_token> prompt_tokens_vec(final_prompt_text.length() + 16);
    int n_prompt_tokens = llama_tokenize(
        model_, final_prompt_text.c_str(), (int32_t)final_prompt_text.length(),
        prompt_tokens_vec.data(), (int32_t)prompt_tokens_vec.size(),
        llama_should_add_bos_token(model_), // Use llama_should_add_bos_token para determinar se BOS é necessário
        true // special = true para processar tokens de controle como <start_of_turn>
    );

    if (n_prompt_tokens < 0) { // Se negativo, significa que o buffer era muito pequeno
        prompt_tokens_vec.resize(-n_prompt_tokens); // n_prompt_tokens é o tamanho necessário
        n_prompt_tokens = llama_tokenize(
            model_, final_prompt_text.c_str(), (int32_t)final_prompt_text.length(),
            prompt_tokens_vec.data(), (int32_t)prompt_tokens_vec.size(),
            llama_should_add_bos_token(model_), true
        );
        if (n_prompt_tokens < 0) { // Ainda falhou
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

    // Limpar o cache KV para a sequência 0 (assumindo uma única sequência por predição)
    llama_kv_cache_seq_clear(ctx_, 0);

    // Criar o batch para processar o prompt
    // O tamanho do batch pode ser n_prompt_tokens para o prompt inicial.
    // Para geração, o batch terá apenas 1 token.
    llama_batch batch = llama_batch_init(std::max(n_prompt_tokens, 1), 0, 1); // Max com 1 para evitar batch de 0

    // Adicionar tokens do prompt ao batch
    for (int i = 0; i < n_prompt_tokens; ++i) {
        // Adiciona token, posição, IDs de sequência, e flag de logits
        llama_batch_add_token(batch, prompt_tokens_vec[i], i, {0}, (i == n_prompt_tokens - 1));
    }
    // batch.logits[batch.n_tokens - 1] = true; // Assegura que os logits são calculados para o último token do prompt.
                                             // llama_batch_add_token já faz isso com o último parâmetro.

    if (batch.n_tokens == 0) {
         if (!user_prompt.empty()) {
             std::cerr << "LlmEngine::predict: Prompt tokenized to zero tokens, though user_prompt was not empty." << std::endl;
             llama_batch_free(batch);
             return "[Error: Prompt tokenized to zero tokens]";
         } else { // Prompt vazio resultou em zero tokens, o que é esperado.
             llama_batch_free(batch);
             return "";
         }
    }

    // Processar o prompt
    if (llama_decode(ctx_, batch) != 0) {
        std::cerr << "LlmEngine::predict: llama_decode failed for prompt." << std::endl;
        llama_batch_free(batch);
        return "[Error: llama_decode failed for prompt]";
    }

    std::string result_text = "";
    int n_cur = n_prompt_tokens; // Posição atual na sequência
    int n_decoded = 0;           // Número de tokens decodificados nesta chamada

    // Configurar parâmetros de amostragem para llama_sampling_context
    llama_sampling_params sparams = llama_sampling_default_params(); // Começar com defaults
    sparams.temp            = temp_param;
    sparams.top_k           = top_k_param <= 0 ? 0 : top_k_param; // 0 na API de sampling_params desabilita top_k.
                                                              // llama_vocab_n_tokens(llama_model_get_vocab(model_)) faria todos os tokens serem considerados.
                                                              // Para desabilitar, 0 é o correto aqui.
    sparams.top_p           = top_p_param;
    sparams.penalty_repeat  = repeat_penalty_param; // Nome do parâmetro mudou de repeat_penalty para penalty_repeat
    sparams.penalty_last_n  = current_n_ctx > 0 ? std::min(current_n_ctx, 256) : 256; // Exemplo: últimos 256 tokens
    // sparams.penalty_freq    = 0.0f; // Default
    // sparams.penalty_present = 0.0f; // Default

    llama_sampling_context * sampling_ctx = llama_sampling_init(sparams);
    if (!sampling_ctx) {
        std::cerr << "LlmEngine::predict: Failed to initialize sampling context." << std::endl;
        llama_batch_free(batch);
        return "[Error: Failed to initialize sampling context]";
    }

    // Alimentar tokens do prompt no contexto de amostragem para que ele conheça o histórico para penalidades
    for (int i = 0; i < n_prompt_tokens; ++i) {
        llama_sampling_accept(sampling_ctx, ctx_, prompt_tokens_vec[i], false);
    }

    // Loop de geração de tokens
    llama_batch_clear(batch); // Limpar o batch, agora vamos adicionar 1 token de cada vez

    while (n_cur < current_n_ctx && n_decoded < max_tokens_to_generate) {
        // Amostrar o próximo token
        llama_token new_token_id = llama_sampling_sample(sampling_ctx, ctx_, nullptr, 0); // idx = 0 para o único logit no batch

        // Aceitar o token amostrado no contexto de amostragem (para penalidades futuras)
        llama_sampling_accept(sampling_ctx, ctx_, new_token_id, true);

        if (new_token_id == llama_token_eos(model_)) { // Usar llama_token_eos(model*)
            // result_text += " [EOS]"; // Opcional: adicionar marcador EOS para debug
            break;
        }

        char piece_buffer[64]; // Buffer para o texto do token
        // llama_token_to_piece agora usa llama_model*
        int len = llama_token_to_piece(model_, new_token_id, piece_buffer, sizeof(piece_buffer));

        if (len > 0) {
            result_text.append(piece_buffer, len);

            // Verificar sequências de parada (incluindo <end_of_turn> específico do Gemma)
            const size_t min_len_for_stop_check = 6; // Ajustar conforme necessário
            const std::vector<std::string> stop_sequences = {
                "\nUSER:", "\nASSISTANT:", " USER:", " ASSISTANT:", "<end_of_turn>"
            };
            if (result_text.length() >= min_len_for_stop_check) { // Evitar verificações em strings muito curtas
                bool stopped = false;
                for (const auto& seq : stop_sequences) {
                    // Verificar se result_text termina com a sequência de parada
                    if (result_text.length() >= seq.length() &&
                        result_text.rfind(seq) == result_text.length() - seq.length()) {
                        //std::cout << " [Stop sequence detected: '" << seq << "']" << std::endl;
                        result_text.erase(result_text.length() - seq.length()); // Remover a sequência
                        stopped = true;
                        break;
                    }
                }
                if (stopped) {
                    break; // Sair do loop de geração
                }
            }
        } else if (len < 0) { // Erro na conversão token para texto
            std::cerr << "LlmEngine::predict: llama_token_to_piece failed for token " << new_token_id << ". Returned: " << len << std::endl;
            // Continuar ou parar? Por enquanto, continuar, mas registrar o erro.
        }

        // Preparar o batch para o próximo token
        llama_batch_clear(batch);
        llama_batch_add_token(batch, new_token_id, n_cur, {0}, true); // Logits para o próximo token

        n_decoded++;
        n_cur++;

        // Decodificar o novo token
        if (llama_decode(ctx_, batch) != 0) {
            std::cerr << "LlmEngine::predict: llama_decode failed for generated token " << new_token_id << std::endl;
            llama_sampling_free(sampling_ctx);
            llama_batch_free(batch);
            return result_text + "[Error: llama_decode failed during generation]";
        }
    }

    llama_sampling_free(sampling_ctx); // Liberar o contexto de amostragem
    llama_batch_free(batch);           // Liberar o batch
    return result_text;
}

} // namespace cpu_llm_project
>>>>>>> REPLACE
