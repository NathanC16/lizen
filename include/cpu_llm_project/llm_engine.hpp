#ifndef CPU_LLM_PROJECT_LLM_ENGINE_HPP
#define CPU_LLM_PROJECT_LLM_ENGINE_HPP

#include <string>
#include <vector>
#include <functional> // Para std::function, se usarmos callbacks no futuro

// Forward declarações para tipos do llama.cpp para evitar incluir headers do llama aqui diretamente
// se possível, ou apenas incluir o header principal 'llama.h' se for leve e necessário.
#include "llama.h" // Incluir o header principal do llama.cpp

// Não precisamos mais das forward declarations se incluirmos llama.h
// struct llama_model; // Já vem de llama.h
// struct llama_context; // Já vem de llama.h
// enum llama_log_level; // Removido, pois vem de ggml_log_level em llama.h/ggml.h

namespace cpu_llm_project {

class LlmEngine {
public:
    LlmEngine();
    ~LlmEngine();

    // Retorna true se o modelo foi carregado com sucesso, false caso contrário.
    // n_gpu_layers é incluído para compatibilidade com a API do llama.cpp, mas será 0 para CPU.
    bool load_model(const std::string& model_path, int n_ctx = 2048, int n_gpu_layers = 0);
    void unload_model();

    // Gera texto a partir de um prompt.
    std::string predict(const std::string& prompt,
                        int max_tokens = 128,
                        float temp = 0.8f,
                        int top_k = 40,
                        float top_p = 0.9f,
                        float repeat_penalty = 1.1f);

    // Callback para streaming de tokens, se implementarmos no futuro
    // using token_callback = std::function<void(const std::string& token)>;
    // std::string predict_streaming(const std::string& prompt, token_callback callback, ...);

    bool is_model_loaded() const;
    std::string get_model_path() const; // Getter para o model_path

private:
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;

    std::string model_path_;
    int n_ctx_ = 0;
    // Adicionar mais parâmetros conforme necessário (n_threads, n_batch, etc.)
    // int n_threads = 4; // Exemplo

    // A função de callback estática para logs do llama.cpp será definida no .cpp
    // e usará ggml_log_level diretamente. Não precisa ser membro da classe.
    // static void static_llama_log_callback(ggml_log_level level, const char *text, void *user_data);
};

} // namespace cpu_llm_project

#endif // CPU_LLM_PROJECT_LLM_ENGINE_HPP
