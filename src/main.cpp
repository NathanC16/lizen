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

    // Parâmetros com valores padrão
    std::string host = "localhost";
    int port = 8080;
    int n_ctx = 2048;
    int num_threads = 0; // 0 para LlmEngine usar lógica padrão (hardware_concurrency)
    bool run_server_mode = false; // Por padrão, não roda o servidor (prefere interativo se poucos args)
    bool interactive_flag_explicitly_passed = false;

    // Analisar argumentos da linha de comando
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
