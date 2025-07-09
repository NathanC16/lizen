#include "cpu_llm_project/dummy.hpp"
#include <iostream> // Para std::cout em uma função de exemplo, se necessário

namespace cpu_llm_project {

std::string get_greeting(const std::string& name) {
    if (name.empty()) {
        return "Hello, there!";
    }
    return "Hello, " + name + "!";
}

// Função de exemplo para demonstrar a biblioteca
void print_avx_message_from_lib() {
    #if defined(__AVX__)
        std::cout << "[Lib] Compiled with AVX support." << std::endl;
    #else
        std::cout << "[Lib] Not compiled with AVX support (from macro check)." << std::endl;
    #endif
}

} // namespace cpu_llm_project
