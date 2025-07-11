#include "cpu_llm_project/llm_engine.hpp"
#include <iostream>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <string.h>
#include <thread>

#include "llama.h" // Incluir o header principal do llama.cpp diretamente aqui

static void LlmEngine_static_llama_log_callback(ggml_log_level level, const char *text, void *user_data) {
    (void)user_data;
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fprintf(stderr, "[LlamaLog] %s", text);
        fflush(stderr);
    }
}

namespace cpu_llm_project {

LlmEngine::LlmEngine() {
    llama_log_set(LlmEngine_static_llama_log_callback, nullptr);
    llama_backend_init();
}

LlmEngine::~LlmEngine() {
    unload_model();
    llama_backend_free();
}

bool LlmEngine::load_model(const std::string& model_path, int n_ctx_req, int n_gpu_layers, int num_threads_param) {
    if (is_model_loaded()) { return false; }
    model_path_ = model_path;
    n_ctx_ = n_ctx_req > 0 ? n_ctx_req : 2048;
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    model_ = llama_model_load_from_file(model_path_.c_str(), model_params);
    if (!model_) {
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx_;
    ctx_params.n_batch = 512;
    ctx_params.n_threads = num_threads_param > 0 ? num_threads_param : std::thread::hardware_concurrency();
    ctx_params.n_threads_batch = ctx_params.n_threads;
    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }
    return true;
}

void LlmEngine::unload_model() {
    if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
    if (model_) { llama_model_free(model_); model_ = nullptr; }
}

bool LlmEngine::is_model_loaded() const {
    return model_ != nullptr;
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
    if (!is_model_loaded()) { return "[Error: Model not loaded]"; }

    std::string final_prompt_text = "<start_of_turn>user\n" + user_prompt + "<end_of_turn>\n<start_of_turn>model";
    if (!system_prompt.empty()) {
        final_prompt_text = system_prompt + "\n" + final_prompt_text;
    }

    const auto * vocab = llama_model_get_vocab(model_);
    std::vector<llama_token> prompt_tokens(final_prompt_text.length() + 16);
    int n_prompt_tokens = llama_tokenize(
        vocab,
        final_prompt_text.c_str(), final_prompt_text.length(),
        prompt_tokens.data(), prompt_tokens.size(), true, true);

    if (n_prompt_tokens < 0) { return "[Error: Tokenization failed]"; }
    prompt_tokens.resize(n_prompt_tokens);

    llama_kv_self_clear(ctx_);

    llama_batch batch = llama_batch_init(512, 0, 1);
    for (int i = 0; i < n_prompt_tokens; ++i) {
        batch.token[batch.n_tokens] = prompt_tokens[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = (i == n_prompt_tokens - 1);
        batch.n_tokens++;
    }

    if (llama_decode(ctx_, batch) != 0) { return "[Error: Decode failed]"; }

    std::string result_text = "";
    int n_cur = n_prompt_tokens;

    // Usar a API llama_sampler_chain
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler * sampler = llama_sampler_chain_init(sparams);

    // Adicionar samplers individuais
    if (temp_param > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp_param));
    }
    if (top_k_param > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k_param));
    }
    if (top_p_param > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p_param, 1)); // min_keep = 1
    }
    if (repeat_penalty_param != 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_penalties(256, repeat_penalty_param, 0.0f, 0.0f)); // penalty_last_n = 256, freq_penalty = 0, present_penalty = 0
    }
    // Adicionar um sampler final para selecionar o token (greedy ou dist)
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    for(auto token : prompt_tokens) {
        llama_sampler_accept(sampler, token);
    }

    while (n_cur <= max_tokens_to_generate) {
        llama_token new_token_id = llama_sampler_sample(sampler, ctx_, 0);
        llama_sampler_accept(sampler, new_token_id);

        if (new_token_id == llama_vocab_eos(vocab)) { break; }

        char piece_buffer[64];
        int len = llama_token_to_piece(vocab, new_token_id, piece_buffer, sizeof(piece_buffer), 0, true);
        if (len > 0) { result_text.append(piece_buffer, len); }

        llama_batch_free(batch);
        batch = llama_batch_init(512, 0, 1);
        batch.token[batch.n_tokens] = new_token_id;
        batch.pos[batch.n_tokens] = n_cur;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = true;
        batch.n_tokens++;

        if (llama_decode(ctx_, batch) != 0) { break; }
        n_cur++;
    }

    llama_sampler_free(sampler);
    llama_batch_free(batch);

    return result_text;
}

} // namespace
