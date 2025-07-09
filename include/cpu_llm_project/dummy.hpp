// Este é um arquivo header de exemplo.
// Coloque as declarações de suas classes e funções públicas aqui.
#ifndef CPU_LLM_PROJECT_DUMMY_HPP
#define CPU_LLM_PROJECT_DUMMY_HPP

#include <string>

namespace cpu_llm_project {

/**
 * @brief Uma função de exemplo.
 *
 * @param name O nome para saudar.
 * @return Uma string de saudação.
 */
std::string get_greeting(const std::string& name);

/**
 * @brief Prints a message indicating if the library was compiled with AVX support.
 *
 */
void print_avx_message_from_lib();

} // namespace cpu_llm_project

#endif // CPU_LLM_PROJECT_DUMMY_HPP
