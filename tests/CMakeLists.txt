# Define o executável de teste
add_executable(run_tests
    test_example.cpp
    test_llm_engine.cpp
)

# Linka o executável de teste com o Catch2 e a biblioteca do projeto
# cpu_llm_lib já está linkada com llama (PUBLIC), então run_tests terá acesso a tudo que precisa.
target_link_libraries(run_tests PRIVATE Catch2::Catch2WithMain cpu_llm_lib)

# Adiciona o teste ao CTest
include(CTest)
message(STATUS "[tests/CMakeLists.txt] CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")
message(STATUS "[tests/CMakeLists.txt] CMAKE_RUNTIME_OUTPUT_DIRECTORY = ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
# Especifica o caminho completo para o executável de teste, usando a variável CMAKE_RUNTIME_OUTPUT_DIRECTORY
# que foi definida no CMakeLists.txt principal.
set(TEST_COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/run_tests)
message(STATUS "[tests/CMakeLists.txt] Test command = ${TEST_COMMAND}")
add_test(NAME unit_tests COMMAND ${TEST_COMMAND})
