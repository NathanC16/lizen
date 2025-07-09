#include "cpu_llm_project/api_server.hpp"
#include "cpu_llm_project/llm_engine.hpp" // Definição completa do LlmEngine

// Definir CPPHTTPLIB_OPENSSL_SUPPORT ou CPPHTTPLIB_ZLIB_SUPPORT aqui se necessário
// ANTES de incluir httplib.h, e garantir que as bibliotecas estejam linkadas.
// Por simplicidade, não usaremos HTTPS ou compressão Zlib inicialmente.
#include "httplib.h"
#include "nlohmann/json.hpp"

#include <iostream>
#include <thread>   // Para std::thread, se rodarmos o servidor em background
#include <chrono>   // Para timestamps
#include <iomanip>  // Para std::put_time

// Para conveniência
using json = nlohmann::json;

namespace cpu_llm_project {

// Função helper para obter timestamp ISO 8601
std::string get_iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto itt = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ss;
    // Nota: std::put_time pode não estar disponível em todos os compiladores C++11/14.
    // Se houver problemas, uma formatação manual seria necessária.
    // Para C++20, pode-se usar std::format.
    // Para GCC com C++11, std::put_time está em <iomanip>.
    ss << std::put_time(std::gmtime(&itt), "%FT%TZ");
    return ss.str();
}


ApiServer::ApiServer(LlmEngine& engine, const std::string& host, int port)
    : engine_(engine), host_(host), port_(port) {
    server_ = std::make_unique<httplib::Server>();
    setup_routes();
}

ApiServer::~ApiServer() {
    stop(); // Garante que o servidor pare se estiver rodando.
}

void ApiServer::setup_routes() {
    if (!server_) return;

    server_->Post("/api/generate", [this](const httplib::Request& req, httplib::Response& res) {
        this->post_generate(req, res);
    });

    server_->Get("/health", [](const httplib::Request& /*req*/, httplib::Response& res) {
        json response_json;
        response_json["status"] = "ok";
        res.set_content(response_json.dump(), "application/json");
        res.status = 200;
    });
}

bool ApiServer::start() {
    if (!server_) {
        std::cerr << "ApiServer::start: Server not initialized." << std::endl;
        return false;
    }
    if (server_->is_running()) {
        std::cout << "ApiServer::start: Server is already running." << std::endl;
        return true;
    }

    std::cout << "ApiServer: Starting to listen on http://" << host_ << ":" << port_ << " ..." << std::endl;

    // Rodar o servidor de forma bloqueante nesta thread.
    // Para rodar em background, precisaríamos de std::thread e um mecanismo de parada mais robusto.
    if (!server_->listen(host_.c_str(), port_)) {
        std::cerr << "ApiServer::start: Failed to listen on " << host_ << ":" << port_ << std::endl;
        return false;
    }
    // Se listen() retornar, o servidor parou (ou falhou ao iniciar).
    std::cout << "ApiServer: Server finished listening." << std::endl;
    return true;
}

void ApiServer::stop() {
    if (server_ && server_->is_running()) {
        std::cout << "ApiServer: Stopping server..." << std::endl;
        server_->stop();
        std::cout << "ApiServer: Server stopped." << std::endl;
    }
}

void ApiServer::post_generate(const httplib::Request& req, httplib::Response& res) {
    json request_json;
    try {
        request_json = json::parse(req.body);
    } catch (json::parse_error& e) {
        res.status = 400; // Bad Request
        json error_json = {{"error", "Invalid JSON format: " + std::string(e.what())}};
        res.set_content(error_json.dump(), "application/json");
        return;
    }

    if (!request_json.contains("prompt") || !request_json["prompt"].is_string()) {
        res.status = 400;
        json error_json = {{"error", "Missing or invalid 'prompt' (string) field in request JSON"}};
        res.set_content(error_json.dump(), "application/json");
        return;
    }

    // String para o nome/path do modelo.
    // Inicialmente, vamos assumir que o LlmEngine já tem um modelo carregado.
    // No futuro, este campo "model" poderia ser usado para selecionar/carregar modelos.
    // std::string model_name = request_json.value("model", engine_.model_path_);
    // if (model_name != engine_.model_path_ || !engine_.is_model_loaded()) {
    //     // Tentar carregar o modelo (ou recarregar se diferente)
    //     // Isso adicionaria complexidade, por agora vamos pular.
    // }


    if (!engine_.is_model_loaded()) {
        res.status = 503; // Service Unavailable
        json error_json = {{"error", "No model is currently loaded in the engine. Load a model first."}};
        res.set_content(error_json.dump(), "application/json");
        return;
    }

    std::string prompt = request_json["prompt"].get<std::string>();

    // Parâmetros de amostragem (com padrões do LlmEngine ou da requisição)
    int max_tokens = request_json.value("max_tokens", 128);
    float temp = request_json.value("temperature", 0.8f);
    int top_k = request_json.value("top_k", 40);
    float top_p = request_json.value("top_p", 0.9f);
    float repeat_penalty = request_json.value("repeat_penalty", 1.1f);
    // bool stream = request_json.value("stream", false); // Streaming não implementado ainda

    std::cout << "ApiServer::post_generate: Received prompt: \"" << prompt << "\"" << std::endl;
    std::string generated_text = engine_.predict(prompt, max_tokens, temp, top_k, top_p, repeat_penalty);
    std::cout << "ApiServer::post_generate: Generated response: \"" << generated_text << "\"" << std::endl;

    json response_data;
    response_data["model"] = engine_.get_model_path(); // Usa o getter
    response_data["created_at"] = get_iso_timestamp();
    response_data["response"] = generated_text;
    response_data["done"] = true; // Para compatibilidade com API Ollama (non-streaming)
    // TODO: Adicionar "total_duration", "load_duration", "prompt_eval_duration", "eval_duration" como Ollama faz.
    // response_data["context"] = ...; // Para follow-up se não for streaming

    res.set_content(response_data.dump(), "application/json");
    res.status = 200;
}

} // namespace cpu_llm_project
