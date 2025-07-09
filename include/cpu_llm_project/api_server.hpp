#ifndef CPU_LLM_PROJECT_API_SERVER_HPP
#define CPU_LLM_PROJECT_API_SERVER_HPP

#include <string>
#include <memory> // Para std::unique_ptr

// Forward declare LlmEngine para evitar include completo aqui se não for necessário
// No entanto, como o ApiServer o usará diretamente, um include pode ser mais simples
// dependendo de como os métodos são implementados.
// Por enquanto, vamos com forward declaration e incluir no .cpp.

// Forward declaration para classes do cpp-httplib
namespace httplib {
    class Server;
    struct Request;
    struct Response;
}

namespace cpu_llm_project {

class LlmEngine; // Forward declaration da nossa classe LlmEngine

class ApiServer {
public:
    ApiServer(LlmEngine& engine, const std::string& host = "localhost", int port = 8080);
    ~ApiServer();

    bool start(); // Retorna true se iniciou com sucesso
    void stop();

private:
    void setup_routes();

    // Handlers para as rotas da API
    void post_generate(const httplib::Request& req, httplib::Response& res);
    // Adicionar mais handlers conforme necessário (ex: /api/chat, /api/models)

    LlmEngine& engine_; // Referência ao motor LLM principal
    std::unique_ptr<httplib::Server> server_; // O servidor HTTP
    std::string host_;
    int port_;
    // std::thread server_thread_; // Para rodar o servidor em uma thread separada
    // bool is_running_ = false; // Para controlar o estado do servidor
};

} // namespace cpu_llm_project

#endif // CPU_LLM_PROJECT_API_SERVER_HPP
