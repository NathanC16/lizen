#include <iostream>
#include <vector>
#include <string>

// Incluir header da biblioteca
#include "cpu_llm_project/dummy.hpp"
#include "cpu_llm_project/llm_engine.hpp" // Nosso novo motor LLM
#include "cpu_llm_project/api_server.hpp" // Nosso servidor API
#include <fstream>      // Para std::ifstream
#include <sstream>      // Para std::ostringstream

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

    // Parâmetros de linha de comando para o servidor e modelo
    std::string host = "localhost";
    int port = 8080;
    int n_ctx = 2048; // Tamanho do contexto padrão

    // Analisar argumentos para host, porta e n_ctx
    bool run_server_mode = false;
    bool interactive_mode_requested = false;

    // Primeiro, verificar se a flag --interactive está presente em qualquer lugar após o model_path
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--interactive") {
            interactive_mode_requested = true;
            break;
        }
    }

    if (interactive_mode_requested) {
        run_server_mode = false;
    } else {
        // Se --interactive não foi passada, verificar se temos argumentos para o modo servidor
        if (argc > 2) { // Pelo menos host é fornecido
            host = argv[2]; // Assumir que argv[2] é o host
            run_server_mode = true;

            if (argc > 3) { // Porta fornecida
                try {
                    port = std::stoi(argv[3]);
                } catch (const std::invalid_argument& ia) {
                    std::cerr << "Aviso: Argumento de porta inválido '" << argv[3] << "'. Usando porta padrão: " << port << std::endl;
                } catch (const std::out_of_range& oor) {
                    std::cerr << "Aviso: Argumento de porta fora do intervalo '" << argv[3] << "'. Usando porta padrão: " << port << std::endl;
                }
            }
            if (argc > 4) { // n_ctx fornecido
                // Certificar-se de que argv[4] não é --interactive (embora já verificado acima, é uma dupla checagem)
                if (std::string(argv[4]) != "--interactive") {
                     try {
                        n_ctx = std::stoi(argv[4]);
                    } catch (const std::invalid_argument& ia) {
                        std::cerr << "Aviso: Argumento n_ctx inválido '" << argv[4] << "'. Usando n_ctx padrão: " << n_ctx << std::endl;
                    } catch (const std::out_of_range& oor) {
                        std::cerr << "Aviso: Argumento n_ctx fora do intervalo '" << argv[4] << "'. Usando n_ctx padrão: " << n_ctx << std::endl;
                    }
                }
            }
        } else {
            // argc == 2 (apenas ./programa <modelo>), modo interativo por padrão
            run_server_mode = false;
        }
    }


    cpu_llm_project::LlmEngine engine;
    if (!engine.load_model(model_path, n_ctx, 0)) {
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

            if (line == "sair" || line == "exit" || line == "quit") {
                break;
            }

            if (line.empty()) {
                continue;
            }

            std::cout << "Processando..." << std::endl;
            // Usar parâmetros padrão para a predição no modo interativo por enquanto
            // A função predict atualmente retorna uma mensagem estática.
            std::string response = engine.predict(line);
            std::cout << "Resposta: " << response << std::endl;
        }
        std::cout << "CPU LLM Project - Modo interativo encerrado." << std::endl;
    }

    return 0;
}
