#include <iostream>
#include <vector>
#include <string>

// Incluir header da biblioteca
#include "cpu_llm_project/dummy.hpp"
#include "cpu_llm_project/llm_engine.hpp" // Nosso novo motor LLM
#include "cpu_llm_project/api_server.hpp" // Nosso servidor API
#include <fstream>      // Para std::ifstream
#include <sstream>      // Para std::ostringstream
#include <map>          // Para std::map (usado para carregar .env)
#include <algorithm>    // Para std::remove, std::isspace

// Para checagem de AVX em tempo de execução (exemplo)
#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#elif defined(_MSC_VER)
#include <intrin.h>
#endif

// Função para checar suporte a AVX em tempo de execução
// Retirado de: https://stackoverflow.com/a/6122298/1116312
bool isAvxSupported() {
    #if defined(__GNUC__) || defined(__clang__)
    unsigned int eax, ebx, ecx, edx;
    // EAX=1: Basic CPUID Information
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return false;
    }
    // Check AVX bit (ECX bit 28)
    return (ecx & (1 << 28)) != 0;
    #elif defined(_MSC_VER)
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    // Check AVX bit (cpuInfo[2] bit 28)
    return (cpuInfo[2] & (1 << 28)) != 0;
    #else
    // Compilador não suportado para esta checagem, ou não é x86/x64
    // Assumir que não há suporte para ser seguro, ou pesquisar método específico.
    std::cerr << "AVX check not implemented for this compiler." << std::endl;
    return false;
    #endif
}

bool isAvx2Supported() {
    #if defined(__GNUC__) || defined(__clang__)
    unsigned int eax, ebx, ecx, edx;
    // EAX=7, ECX=0: Extended Features
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return false;
    }
    // Check AVX2 bit (EBX bit 5)
    return (ebx & (1 << 5)) != 0;
    #elif defined(_MSC_VER)
    int cpuInfo[4];
    __cpuidex(cpuInfo, 7, 0);
    // Check AVX2 bit (cpuInfo[1] bit 5)
    return (cpuInfo[1] & (1 << 5)) != 0;
    #else
    std::cerr << "AVX2 check not implemented for this compiler." << std::endl;
    return false;
    #endif
}

// Função auxiliar para remover espaços em branco de uma string (início e fim)
std::string trim_string(const std::string& str) {
    const std::string whitespace = " \t\n\r\f\v";
    size_t start = str.find_first_not_of(whitespace);
    if (start == std::string::npos) {
        return ""; // String vazia ou só com espaços
    }
    size_t end = str.find_last_not_of(whitespace);
    return str.substr(start, end - start + 1);
}

// Função para carregar variáveis de um arquivo .env para um map
std::map<std::string, std::string> load_env_file(const std::string& path = ".env") {
    std::map<std::string, std::string> env_vars;
    std::ifstream env_file(path);

    if (!env_file.is_open()) {
        // std::cout << "Info: Arquivo .env não encontrado em " << path << ". Usando padrões e argumentos CLI." << std::endl;
        return env_vars; // Retorna mapa vazio se o arquivo não puder ser aberto
    }

    std::string line;
    while (std::getline(env_file, line)) {
        line = trim_string(line);
        if (line.empty() || line[0] == '#') { // Ignorar linhas vazias e comentários
            continue;
        }

        size_t delimiter_pos = line.find('=');
        if (delimiter_pos != std::string::npos) {
            std::string key = line.substr(0, delimiter_pos);
            std::string value = line.substr(delimiter_pos + 1);
            key = trim_string(key);
            value = trim_string(value);

            // Remover aspas do valor, se presentes
            if (value.length() >= 2 && ((value.front() == '"' && value.back() == '"') || (value.front() == '\'' && value.back() == '\''))) {
                value = value.substr(1, value.length() - 2);
            }
            env_vars[key] = value;
        }
    }
    env_file.close();
    // std::cout << "Info: Arquivo .env carregado." << std::endl;
    return env_vars;
}


int main(int argc, char* argv[]) {
    std::cout << "CPU LLM Project - Início" << std::endl;

    if (isAvxSupported()) {
        std::cout << "Info: Suporte a AVX detectado em tempo de execução." << std::endl;
    } else {
        std::cout << "Aviso: Suporte a AVX NÃO detectado em tempo de execução." << std::endl;
        std::cout << "Este projeto requer AVX para performance otimizada." << std::endl;
        // Poderia sair aqui se AVX for um requisito estrito para rodar.
        // return 1;
    }

    if (isAvx2Supported()) {
        std::cout << "Info: Suporte a AVX2 detectado em tempo de execução." << std::endl;
    } else {
        std::cout << "Info: Suporte a AVX2 NÃO detectado em tempo de execução. (Isso é esperado se o alvo é apenas AVX)" << std::endl;
    }

    if (argc > 1) {
        std::cout << "Argumentos recebidos: " << std::endl;
        for (int i = 1; i < argc; ++i) {
            std::cout << i << ": " << argv[i] << std::endl;
        }
    } else {
        std::cout << "Nenhum argumento recebido." << std::endl;
    }

    // Chamando uma função da nossa biblioteca (exemplo antigo)
    // cpu_llm_project::print_avx_message_from_lib();
    // std::cout << "Greeting from lib: " << cpu_llm_project::get_greeting("Main") << std::endl;

    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <caminho_para_modelo_gguf> [max_tokens] [temp] [top_k] [top_p] [repeat_penalty]" << std::endl;
        std::cerr << "Exemplo: " << argv[0] << " ./model.gguf" << std::endl;
        std::cerr << "\nParâmetros de amostragem opcionais:\n"
                  << "  max_tokens:    Número máximo de tokens a gerar (padrão: 128)\n"
                  << "  temp:          Temperatura de amostragem (padrão: 0.8)\n"
                  << "  top_k:         Top-k sampling (padrão: 40)\n"
                  << "  top_p:         Top-p (nucleus) sampling (padrão: 0.9)\n"
                  << "  repeat_penalty: Penalidade de repetição (padrão: 1.1)\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::ifstream model_file(model_path);
    if (!model_file.good()) {
        std::cerr << "Erro: Não foi possível abrir o arquivo do modelo em: " << model_path << std::endl;
        return 1;
    }
    model_file.close();

    // Carregar configurações do arquivo .env
    std::map<std::string, std::string> env_config = load_env_file();

    // Parâmetros com valores padrão (que podem ser sobrescritos pelo .env e depois por args CLI)
    std::string host = "localhost";
    int port = 8080;
    int n_ctx = 2048;
    int num_threads = 0;
    // Parâmetros de amostragem e system prompt
    float model_temperature = 0.8f;
    int model_top_k = 40;
    float model_top_p = 0.9f;
    float model_repeat_penalty = 1.1f;
    int max_tokens = 128; // Padrão para predict
    std::string system_prompt = "Você é um assistente de IA prestativo e conciso.";
    // std::string default_model_path_env = ""; // Não usado ativamente ainda


    // Aplicar configurações do .env se presentes
    if (env_config.count("API_HOST")) host = env_config["API_HOST"];
    if (env_config.count("API_PORT")) {
        try { port = std::stoi(env_config["API_PORT"]); } catch (...) { /* Ignorar erro, usar padrão */ }
    }
    if (env_config.count("MODEL_N_CTX")) {
        try { n_ctx = std::stoi(env_config["MODEL_N_CTX"]); } catch (...) { /* Ignorar erro, usar padrão */ }
    }
    if (env_config.count("NUM_THREADS")) {
        try { num_threads = std::stoi(env_config["NUM_THREADS"]); } catch (...) { /* Ignorar erro, usar padrão */ }
    }
    if (env_config.count("MODEL_TEMPERATURE")) {
        try { model_temperature = std::stof(env_config["MODEL_TEMPERATURE"]); } catch (...) { /* Ignorar erro */ }
    }
    if (env_config.count("MODEL_TOP_K")) {
        try { model_top_k = std::stoi(env_config["MODEL_TOP_K"]); } catch (...) { /* Ignorar erro */ }
    }
    if (env_config.count("MODEL_TOP_P")) {
        try { model_top_p = std::stof(env_config["MODEL_TOP_P"]); } catch (...) { /* Ignorar erro */ }
    }
    if (env_config.count("MODEL_REPEAT_PENALTY")) {
        try { model_repeat_penalty = std::stof(env_config["MODEL_REPEAT_PENALTY"]); } catch (...) { /* Ignorar erro */ }
    }
    if (env_config.count("SYSTEM_PROMPT")) system_prompt = env_config["SYSTEM_PROMPT"];
    // if (env_config.count("DEFAULT_MODEL_PATH")) default_model_path_env = env_config["DEFAULT_MODEL_PATH"];

    // Nota: MAX_TOKENS_TO_GENERATE não está sendo carregado do .env por enquanto,
    // mas poderia ser adicionado se necessário para o modo interativo.
    // A API tem seu próprio parâmetro max_tokens.

    bool run_server_mode = false;
    bool interactive_flag_explicitly_passed = false;

    // Analisar argumentos da linha de comando (eles sobrescrevem o .env)
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--interactive") {
            interactive_flag_explicitly_passed = true;
        } else if (arg == "--threads") {
            if (i + 1 < argc) {
                try {
                    num_threads = std::stoi(argv[++i]);
                    if (num_threads <= 0) {
                         std::cerr << "Aviso: Número de threads deve ser positivo. Usando padrão." << std::endl;
                         num_threads = 0; // Reseta para padrão do engine
                    }
                } catch (const std::invalid_argument& ia) {
                    std::cerr << "Aviso: Argumento de threads inválido '" << argv[i] << "'. Usando padrão." << std::endl;
                    num_threads = 0;
                } catch (const std::out_of_range& oor) {
                    std::cerr << "Aviso: Argumento de threads fora do intervalo '" << argv[i] << "'. Usando padrão." << std::endl;
                    num_threads = 0;
                }
            } else {
                std::cerr << "Aviso: Flag --threads requer um argumento numérico." << std::endl;
            }
        } else if (arg == "--host") {
            if (i + 1 < argc) {
                host = argv[++i];
                run_server_mode = true; // Se --host é passado, provavelmente queremos modo servidor
            } else {
                std::cerr << "Aviso: Flag --host requer um argumento." << std::endl;
            }
        } else if (arg == "--port") {
            if (i + 1 < argc) {
                try {
                    port = std::stoi(argv[++i]);
                } catch (const std::invalid_argument& ia) {
                    std::cerr << "Aviso: Argumento de porta inválido '" << argv[i] << "'. Usando porta padrão: " << port << std::endl;
                } catch (const std::out_of_range& oor) {
                    std::cerr << "Aviso: Argumento de porta fora do intervalo '" << argv[i] << "'. Usando porta padrão: " << port << std::endl;
                }
                run_server_mode = true; // Se --port é passado, provavelmente queremos modo servidor
            } else {
                std::cerr << "Aviso: Flag --port requer um argumento numérico." << std::endl;
            }
        } else if (arg == "--n_ctx") {
            if (i + 1 < argc) {
                try {
                    n_ctx = std::stoi(argv[++i]);
                     if (n_ctx <= 0) {
                         std::cerr << "Aviso: Número de n_ctx deve ser positivo. Usando padrão: " << 2048 << std::endl;
                         n_ctx = 2048;
                    }
                } catch (const std::invalid_argument& ia) {
                    std::cerr << "Aviso: Argumento n_ctx inválido '" << argv[i] << "'. Usando n_ctx padrão: " << n_ctx << std::endl;
                } catch (const std::out_of_range& oor) {
                    std::cerr << "Aviso: Argumento n_ctx fora do intervalo '" << argv[i] << "'. Usando n_ctx padrão: " << n_ctx << std::endl;
                }
            } else {
                std::cerr << "Aviso: Flag --n_ctx requer um argumento numérico." << std::endl;
            }
        } else {
            // Se não for uma flag conhecida, e ainda não definimos o host (assumindo modo servidor posicional)
            // Esta lógica posicional é um pouco frágil se misturada com flags nomeadas extensivamente.
            // Por simplicidade, se --host, --port não foram usados, os primeiros args extras ainda podem ser host e port.
            if (i == 2 && !interactive_flag_explicitly_passed) { // Potencialmente o host
                host = arg;
                run_server_mode = true;
            } else if (i == 3 && run_server_mode && !interactive_flag_explicitly_passed) { // Potencialmente a porta
                 try {
                    port = std::stoi(arg);
                } catch (const std::invalid_argument& ia) {
                    std::cerr << "Aviso: Argumento de porta posicional inválido '" << arg << "'. Usando porta padrão: " << port << std::endl;
                } catch (const std::out_of_range& oor) {
                    std::cerr << "Aviso: Argumento de porta posicional fora do intervalo '" << arg << "'. Usando porta padrão: " << port << std::endl;
                }
            } else if (i == 4 && run_server_mode && !interactive_flag_explicitly_passed) { // Potencialmente n_ctx
                 try {
                    n_ctx = std::stoi(arg);
                } catch (const std::invalid_argument& ia) {
                    std::cerr << "Aviso: Argumento n_ctx posicional inválido '" << arg << "'. Usando n_ctx padrão: " << n_ctx << std::endl;
                } catch (const std::out_of_range& oor) {
                    std::cerr << "Aviso: Argumento n_ctx posicional fora do intervalo '" << arg << "'. Usando n_ctx padrão: " << n_ctx << std::endl;
                }
            } else if (arg.rfind("-", 0) == 0) { // Começa com '-' ou '--' mas não é uma flag conhecida
                 std::cerr << "Aviso: Flag desconhecida '" << arg << "' ignorada." << std::endl;
            }
        }
    }

    // Decisão final sobre o modo: --interactive tem prioridade máxima.
    // Se não, e nenhum argumento de servidor foi inferido/passado, modo interativo.
    if (interactive_flag_explicitly_passed) {
        run_server_mode = false;
    } else if (argc == 2) { // Somente ./programa <modelo>
        run_server_mode = false; // Modo interativo por padrão
    }
    // Se run_server_mode foi setado para true por flags como --host ou argumentos posicionais, ele permanece true.

    cpu_llm_project::LlmEngine engine;
    // Passar num_threads para load_model. LlmEngine tratará 0 como "automático".
    if (!engine.load_model(model_path, n_ctx, 0, num_threads)) { // Adicionado num_threads
        std::cerr << "Erro fatal: Não foi possível carregar o modelo: " << model_path << std::endl;
        return 1;
    }
    std::cout << "Modelo " << model_path << " carregado com sucesso no LlmEngine." << std::endl;

    if (run_server_mode) {
        // Iniciar o servidor API
        cpu_llm_project::ApiServer server(engine, host, port);
        std::cout << "Iniciando servidor API em " << host << ":" << port << std::endl;

        // Adicionar um handler de sinal para parar o servidor graciosamente (opcional, mas bom para Ctrl+C)
        // std::signal(SIGINT, [](int signal){ server.stop(); }); // Precisa que 'server' seja acessível

        if (!server.start()) { // start() agora é bloqueante
            std::cerr << "Erro fatal: Falha ao iniciar o servidor API." << std::endl;
            return 1;
        }
        std::cout << "CPU LLM Project - Servidor API encerrado." << std::endl;
    } else {
        // Entrar em modo interativo
        std::cout << "\nModo Interativo. Digite 'sair', 'exit' ou 'quit' para terminar." << std::endl;
        std::string line;
        while (true) {
            std::cout << "\nPrompt: ";
            if (!std::getline(std::cin, line)) {
                // EOF (Ctrl+D) ou erro de leitura
                break;
            }

            // Comandos interativos com //
            if (line.rfind("//", 0) == 0) { // Verifica se a linha começa com "//"
                std::string command = line.substr(2); // Extrai o comando
                if (command == "sair" || command == "exit" || command == "quit") {
                    break;
                } else {
                    std::cout << "Comando desconhecido: " << command << std::endl;
                }
                continue; // Volta para o início do loop para o próximo prompt/comando
            }

            // Comandos de saída legados (sem //) - podem ser removidos gradualmente
            if (line == "sair" || line == "exit" || line == "quit") {
                std::cout << "Usando comando de saída legado. Considere usar '//sair' no futuro." << std::endl;
                break;
            }

            if (line.empty()) {
                continue;
            }

            std::cout << "Processando..." << std::endl;
            // Passar o system_prompt e os parâmetros de amostragem carregados/padrão
            std::string response = engine.predict(line,
                                                  system_prompt,
                                                  max_tokens, // Usar o padrão de predict ou carregar do .env
                                                  model_temperature,
                                                  model_top_k,
                                                  model_top_p,
                                                  model_repeat_penalty);
            std::cout << "Resposta: " << response << std::endl;
        }
        std::cout << "CPU LLM Project - Modo interativo encerrado." << std::endl;
    }

    return 0;
}
