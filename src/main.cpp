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
#include "yaml-cpp/yaml.h" // Para parsing de YAML
#include <cstdlib>     // Para getenv

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
// Removida - load_env_file e sua lógica serão substituídas pelo parsing de YAML.


// Estrutura para armazenar as configurações carregadas do YAML ou definidas por padrão/CLI
struct AppConfig {
    std::string model_gguf_path;
    std::string system_prompt = "Você é um assistente de IA prestativo e conciso.";
    int n_ctx = 2048;
    int num_threads = 0; // 0 para LlmEngine usar lógica padrão
    float model_temperature = 0.8f;
    int model_top_k = 40;
    float model_top_p = 0.9f;
    float model_repeat_penalty = 1.1f;
    int max_tokens = 128;

    std::string api_host = "localhost";
    int api_port = 8080;

    // Adicionar outros campos conforme necessário (nome da persona, etc.)
    std::string persona_name;
};

// Função para carregar e parsear o arquivo YAML de configuração da persona/modelo
bool load_config_from_yaml(const std::string& yaml_path, AppConfig& config) {
    try {
        YAML::Node yaml_config = YAML::LoadFile(yaml_path);

        if (yaml_config["model_gguf_path"]) {
            config.model_gguf_path = yaml_config["model_gguf_path"].as<std::string>();
        } else {
            std::cerr << "Erro: 'model_gguf_path' não encontrado no arquivo YAML: " << yaml_path << std::endl;
            return false;
        }

        if (yaml_config["name"]) config.persona_name = yaml_config["name"].as<std::string>();
        if (yaml_config["n_ctx"]) config.n_ctx = yaml_config["n_ctx"].as<int>(config.n_ctx);
        if (yaml_config["num_threads"]) config.num_threads = yaml_config["num_threads"].as<int>(config.num_threads);
        if (yaml_config["system_prompt"]) config.system_prompt = yaml_config["system_prompt"].as<std::string>();
        if (yaml_config["max_tokens"]) config.max_tokens = yaml_config["max_tokens"].as<int>(config.max_tokens);
        if (yaml_config["temperature"]) config.model_temperature = yaml_config["temperature"].as<float>(config.model_temperature);
        if (yaml_config["top_k"]) config.model_top_k = yaml_config["top_k"].as<int>(config.model_top_k);
        if (yaml_config["top_p"]) config.model_top_p = yaml_config["top_p"].as<float>(config.model_top_p);
        if (yaml_config["repeat_penalty"]) config.model_repeat_penalty = yaml_config["repeat_penalty"].as<float>(config.model_repeat_penalty);

        // API_HOST e API_PORT não são tipicamente por persona, mas podem ser lidos se presentes
        if (yaml_config["api_host"]) config.api_host = yaml_config["api_host"].as<std::string>(config.api_host);
        if (yaml_config["api_port"]) config.api_port = yaml_config["api_port"].as<int>(config.api_port);


        std::cout << "Info: Configuração YAML '" << yaml_path << "' carregada." << std::endl;
        if (!config.persona_name.empty()) {
            std::cout << "Info: Persona carregada: " << config.persona_name << std::endl;
        }
        return true;
    } catch (const YAML::Exception& e) {
        std::cerr << "Erro ao parsear arquivo YAML '" << yaml_path << "': " << e.what() << std::endl;
        return false;
    }
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

    AppConfig config; // Struct para manter todas as configurações
    std::string model_specifier; // Pode ser um caminho .gguf, .yaml ou nome de persona

    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " (<caminho_para_config.yaml> | <caminho_para_modelo.gguf> | --run <nome_persona>) [opções...]" << std::endl;
        std::cerr << "Opções: --interactive, --threads N, --host HOST, --port P, --n_ctx N" << std::endl;
        // Adicionar mais detalhes sobre --list e --create no futuro
        return 1;
    }

    model_specifier = argv[1]; // O primeiro argumento é sempre o especificador do modelo/configuração

    // Parâmetros de CLI que podem sobrescrever YAML/padrões
    std::string cli_host;
    int cli_port = -1; // -1 indica não definido pela CLI
    int cli_n_ctx = -1;
    int cli_num_threads = -1;
    bool interactive_flag_explicitly_passed = false;
    bool run_server_mode = false; // Determinado após análise de args e config

    std::string persona_to_run;
    std::string yaml_path_from_arg;

    // Analisar argumentos da linha de comando para flags e seus valores
    for (int i = 1; i < argc; ++i) { // Começar em 1 para pegar --run ou caminho do yaml
        std::string arg = argv[i];
        if (arg == "--interactive") {
            interactive_flag_explicitly_passed = true;
        } else if (arg == "--threads") {
            if (i + 1 < argc) {
                try { cli_num_threads = std::stoi(argv[++i]); } catch (...) { std::cerr << "Aviso: Valor inválido para --threads: " << argv[i] << std::endl; }
            } else { std::cerr << "Aviso: Flag --threads requer um argumento." << std::endl; }
        } else if (arg == "--host") {
            if (i + 1 < argc) { cli_host = argv[++i]; } else { std::cerr << "Aviso: Flag --host requer um argumento." << std::endl; }
        } else if (arg == "--port") {
            if (i + 1 < argc) {
                try { cli_port = std::stoi(argv[++i]); } catch (...) { std::cerr << "Aviso: Valor inválido para --port: " << argv[i] << std::endl; }
            } else { std::cerr << "Aviso: Flag --port requer um argumento." << std::endl; }
        } else if (arg == "--n_ctx") {
            if (i + 1 < argc) {
                try { cli_n_ctx = std::stoi(argv[++i]); } catch (...) { std::cerr << "Aviso: Valor inválido para --n_ctx: " << argv[i] << std::endl; }
            } else { std::cerr << "Aviso: Flag --n_ctx requer um argumento." << std::endl; }
        } else if (arg == "--run") {
            if (i + 1 < argc) { persona_to_run = argv[++i]; } else { std::cerr << "Aviso: Flag --run requer um nome de persona." << std::endl; }
        } else {
            if (i == 1 && arg.rfind("--", 0) != 0) { // É o primeiro argumento e não parece uma flag
                // Pode ser um caminho para .yaml ou .gguf
                if (arg.length() > 5 && arg.substr(arg.length() - 5) == ".yaml") {
                    yaml_path_from_arg = arg;
                } else if (arg.length() > 5 && arg.substr(arg.length() - 5) == ".gguf") {
                    config.model_gguf_path = arg; // Carregamento direto de GGUF
                } else {
                     // Se não for --run e não for .yaml/.gguf no primeiro argumento, mas houver outros args,
                     // pode ser um erro de uso se não for --list ou --create no futuro.
                     // Por agora, se não for --run e o primeiro arg não for um arquivo conhecido, é um erro.
                     // A menos que seja --list ou --create.
                }
            } else if (arg.rfind("--",0) == 0) {
                 std::cerr << "Aviso: Flag desconhecida '" << arg << "' ignorada." << std::endl;
            }
        }
    }

    // Determinar o caminho do arquivo de configuração YAML
    std::string effective_yaml_path;
    if (!persona_to_run.empty()) {
        effective_yaml_path = "./personas/" + persona_to_run + ".yaml"; // Diretório padrão para personas
        std::cout << "Info: Tentando carregar persona '" << persona_to_run << "' de " << effective_yaml_path << std::endl;
    } else if (!yaml_path_from_arg.empty()) {
        effective_yaml_path = yaml_path_from_arg;
    }

    // Carregar do YAML se um caminho foi determinado
    if (!effective_yaml_path.empty()) {
        if (!load_config_from_yaml(effective_yaml_path, config)) {
            std::cerr << "Erro: Falha ao carregar configuração de " << effective_yaml_path << std::endl;
            return 1;
        }
    } else if (config.model_gguf_path.empty()) { // Se nem --run, nem .yaml, nem .gguf direto foi passado
        std::cerr << "Erro: Nenhum modelo ou arquivo de configuração especificado." << std::endl;
        std::cerr << "Uso: " << argv[0] << " (<caminho_para_config.yaml> | <caminho_para_modelo.gguf> | --run <nome_persona>) [opções...]" << std::endl;
        return 1;
    }

    // Sobrescrever configurações do YAML com flags da CLI, se fornecidas
    if (!cli_host.empty()) config.api_host = cli_host;
    if (cli_port != -1) config.api_port = cli_port;
    if (cli_n_ctx != -1) config.n_ctx = cli_n_ctx > 0 ? cli_n_ctx : config.n_ctx;
    if (cli_num_threads != -1) config.num_threads = cli_num_threads; // LlmEngine trata <=0 como padrão

    // Verificar se o caminho do modelo GGUF é válido após todas as análises
    if (config.model_gguf_path.empty()) {
        std::cerr << "Erro: Caminho para o modelo GGUF não especificado (nem via YAML, nem como argumento direto)." << std::endl;
        return 1;
    }

    // Expandir ~ para o diretório home do usuário, se aplicável
    if (!config.model_gguf_path.empty() && config.model_gguf_path[0] == '~') {
        const char* home_dir = getenv("HOME");
        if (home_dir) {
            config.model_gguf_path.replace(0, 1, home_dir);
            std::cout << "Info: Caminho do modelo expandido para: " << config.model_gguf_path << std::endl;
        } else {
            std::cerr << "Aviso: Não foi possível expandir '~' no caminho do modelo porque a variável de ambiente HOME não está definida. Tentando usar o caminho como está." << std::endl;
        }
    }

    std::ifstream model_check_file(config.model_gguf_path);
    if (!model_check_file.good()) {
        std::cerr << "Erro: Não foi possível abrir o arquivo do modelo GGUF em: " << config.model_gguf_path << std::endl;
        return 1;
    }
    model_check_file.close();


    // Decidir o modo de execução
    // Se host ou port foram definidos (via CLI ou YAML) E --interactive não foi passado, rodar servidor.
    // A verificação de `run_server_mode = true` na análise de --host/--port CLI é importante aqui.
    // A lógica anterior de `run_server_mode` baseada em `argc` não é mais necessária.
    if (interactive_flag_explicitly_passed) {
        run_server_mode = false;
    } else {
        // Se o host ou porta foram especificados (via CLI ou vieram do YAML e não foram o default inicial)
        // e não estamos explicitamente no modo interativo, então é modo servidor.
        bool host_is_set = !config.api_host.empty() && config.api_host != "localhost"; // Exemplo de checagem mais robusta
        bool port_is_set = config.api_port != 0 && config.api_port != 8080; // Exemplo

        // Se alguma flag de servidor foi passada na CLI, run_server_mode já deve ser true.
        // Se vieram do YAML, precisamos de uma lógica para decidir.
        // Por simplicidade agora: se --interactive não foi passada, e (cli_host foi setado OU cli_port foi setado), então modo servidor.
        // Ou se apenas o .gguf foi passado, modo interativo.
        if (!cli_host.empty() || cli_port != -1) { // Se host ou port foram explicitamente passados na CLI
             run_server_mode = true;
        } else if (yaml_path_from_arg.empty() && persona_to_run.empty() && argc > 2) {
            // Caso legado: ./programa modelo.gguf host port
            // Isso é coberto pela lógica de análise de args posicionais que foi removida.
            // Precisa ser reavaliado se queremos manter args posicionais para host/port.
            // Por agora, vamos requerer --host ou --port para modo servidor se não for interativo.
            // Se apenas model_gguf_path é dado (argc==2 implicitamente), ou yaml/run é dado sem flags de servidor, será interativo.
            if (argc == 2 || !effective_yaml_path.empty()) { // Apenas .gguf, ou um .yaml/--run sem flags de servidor
                run_server_mode = false;
            } else {
                // Se temos mais de 2 args, e não é --interactive, e não é um yaml/run que define host/port,
                // é ambiguo. Por agora, se não for explicitamente interativo, e não houver flags de servidor,
                // mas houver args extras, pode ser um erro de uso.
                // Para simplificar: se --interactive não está, e não há --host/--port, modo interativo.
                // A menos que o YAML carregado tenha host/port diferentes do default.
                // A lógica atual de `run_server_mode` na análise de flags CLI já trata isso.
                // Se nenhuma flag de servidor foi usada, `run_server_mode` permanece `false`.
            }
        }
         if (argc == 2 && persona_to_run.empty() && yaml_path_from_arg.empty()) { // Apenas ./programa <modelo.gguf>
            run_server_mode = false; // Interativo por padrão
        }
    }


    cpu_llm_project::LlmEngine engine;
    if (!engine.load_model(config.model_gguf_path, config.n_ctx, 0, config.num_threads)) {
        std::cerr << "Erro fatal: Não foi possível carregar o modelo: " << config.model_gguf_path << std::endl;
        return 1;
    }
    std::cout << "Modelo '" << config.model_gguf_path << "' carregado com sucesso no LlmEngine." << std::endl;

    if (run_server_mode) {
        // Iniciar o servidor API
        cpu_llm_project::ApiServer server(engine, config.api_host, config.api_port);
        std::cout << "Iniciando servidor API em " << config.api_host << ":" << config.api_port << std::endl;

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
            // Passar o system_prompt e os parâmetros de amostragem da struct AppConfig
            std::string response = engine.predict(line,
                                                  config.system_prompt,
                                                  config.max_tokens,
                                                  config.model_temperature,
                                                  config.model_top_k,
                                                  config.model_top_p,
                                                  config.model_repeat_penalty);
            std::cout << "Resposta: " << response << std::endl;
        }
        std::cout << "CPU LLM Project - Modo interativo encerrado." << std::endl;
    }

    return 0;
}
