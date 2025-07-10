cmake_minimum_required(VERSION 3.16)
project(cpu_llm_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- Diretórios de Saída ---
# Colocar todos os executáveis em build/bin e bibliotecas em build/lib
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # Para bibliotecas estáticas
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # Para bibliotecas compartilhadas

# --- Compilação com AVX ---
# Adiciona a flag -mavx para GCC/Clang
# Para MSVC, /arch:AVX é geralmente ativado por padrão quando o hardware suporta,
# mas pode ser especificado se necessário.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mno-avx2")
    # Poderíamos adicionar -march=native, mas para garantir a compatibilidade AVX
    # e não AVX2, -mavx é mais específico e -mno-avx2 desabilita explicitamente o AVX2.
    # Para otimizações gerais:
    # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    # set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
elseif(MSVC)
    # MSVC tipicamente habilita AVX com /arch:AVX ou por padrão em compilações x64
    # se o SDK do Windows suportar e o processador também.
    # Para ser explícito, poderíamos tentar adicionar /arch:AVX,
    # mas é melhor deixar o compilador otimizar para a arquitetura alvo
    # ou requerer uma toolchain que suporte isso.
    # Por enquanto, vamos assumir que as configurações padrão do MSVC para x64
    # farão um bom trabalho ou que flags específicas serão adicionadas manualmente se necessário.
    # message(STATUS "MSVC compiler detected. AVX support typically enabled by default for x64.")
endif()

# --- Incluir diretórios ---
include_directories(include)

# --- Criar uma biblioteca para o código principal ---
# Esta biblioteca conterá a lógica que pode ser usada tanto pelo executável principal
# quanto pelos testes.
# Por enquanto, vamos adicionar um arquivo dummy à biblioteca, e main.cpp ficará no executável.
# No futuro, src/main.cpp pode se tornar menor, e a lógica principal movida para arquivos
# dentro da biblioteca.
add_library(cpu_llm_lib
    src/dummy_lib_file.cpp
    src/llm_engine.cpp
)
target_include_directories(cpu_llm_lib PUBLIC include)

# --- Adicionar o executável principal ---
find_package(Threads REQUIRED) # Adicionar para encontrar dependências de thread

# Ele linkará com a nossa biblioteca e também incluirá o código do servidor API.
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/api_server.cpp
)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        cpu_llm_lib         # Nossa biblioteca principal com LlmEngine
        httplib::httplib    # Target do cpp-httplib (se existir) - cpp-httplib geralmente é header-only
        nlohmann_json::nlohmann_json # Target do nlohmann/json
        ${CMAKE_THREAD_LIBS_INIT} # Linkar com bibliotecas de thread
)
# Para cpp-httplib, se for header-only e não criar um target linkável 'httplib::httplib',
# apenas garantir que seus headers estejam no include path é suficiente.
# FetchContent_MakeAvailable(httplib) já deve ter tornado os headers acessíveis.
# Se 'httplib::httplib' não for um target válido, podemos remover da lista de link.
# A tag v0.15.3 do cpp-httplib pode não criar um target importado 'httplib::httplib'.
# Se a compilação falhar por causa disso, removeremos 'httplib::httplib' do link.

# --- Configuração de Testes com Catch2 ---
# Habilitar o CTest para o projeto
enable_testing()

# Usaremos FetchContent para obter o Catch2
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.2 # Use uma tag de release estável
)
FetchContent_MakeAvailable(Catch2)

# --- Adicionar llama.cpp como dependência ---
FetchContent_Declare(
  llama_cpp
  GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
  GIT_TAG        master # Usar a branch master para ter o suporte mais recente a modelos
)

# Configurar opções de build para llama.cpp antes de torná-lo disponível
# Queremos AVX, mas não necessariamente AVX2, FMA, etc.
# LLAMA_STATIC para construir como biblioteca estática.
# LLAMA_BUILD_TESTS e LLAMA_BUILD_EXAMPLES podem ser OFF para economizar tempo de compilação.
# GGML_OPENMP ou GGML_SYCL etc., podem ser controlados aqui se necessário.
# Por padrão, llama.cpp tentará detectar e usar as melhores instruções.
# Para forçar AVX e desabilitar AVX2:
# -DLLAMA_AVX=ON -DLLAMA_AVX2=OFF -DLLAMA_AVX512=OFF
# -DLLAMA_FMA=OFF -DLLAMA_F16C=OFF (se quisermos ser muito específicos sobre não usar nada além de AVX puro)
# No entanto, é geralmente melhor deixar o llama.cpp usar o que ele considera melhor para AVX,
# e o nosso compilador principal já está com -mavx.
# Vamos começar com uma configuração mais padrão e ajustar se necessário.
# Desabilitar exemplos e testes do llama.cpp para acelerar nosso build.
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build llama.cpp examples" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build llama.cpp tests" FORCE)
set(LLAMA_STATIC ON CACHE BOOL "Build llama.cpp as a static library" FORCE)
# Para garantir que ele não tente pegar coisas que não queremos no nosso contexto específico.
set(LLAMA_MPI OFF CACHE BOOL "Disable MPI for llama.cpp" FORCE)
set(LLAMA_OPENBLAS OFF CACHE BOOL "Disable OpenBLAS for llama.cpp" FORCE) # A menos que queiramos gerenciar essa dependência
set(LLAMA_BLIS OFF CACHE BOOL "Disable BLIS for llama.cpp" FORCE)
# set(LLAMA_CUDA OFF CACHE BOOL "Disable CUDA for llama.cpp" FORCE) # Já deve ser o padrão se CUDA não estiver presente

# Forçar configurações para focar em AVX e desabilitar instruções mais recentes para llama.cpp
# Isso garante que o llama.cpp não tente usar AVX2, AVX512, FMA, ou F16C se não quisermos depender deles.
# O compilador do nosso projeto já está configurado com -mavx.
# Comentando temporariamente as flags explícitas de CPU para o llama.cpp para teste.
# Deixar o llama.cpp usar sua própria detecção de hardware por enquanto.
set(LLAMA_AVX ON CACHE BOOL "Enable AVX for llama.cpp" FORCE)
set(LLAMA_AVX2 OFF CACHE BOOL "Disable AVX2 for llama.cpp" FORCE)
set(LLAMA_AVX512 OFF CACHE BOOL "Disable AVX512 for llama.cpp" FORCE)
set(LLAMA_FMA OFF CACHE BOOL "Disable FMA for llama.cpp" FORCE) # FMA é frequentemente associado com AVX2
set(LLAMA_F16C OFF CACHE BOOL "Disable F16C for llama.cpp" FORCE) # F16C é frequentemente associado com AVX2/AVX-512
# set(LLAMA_ACCELERATE OFF CACHE BOOL "Disable Accelerate framework for llama.cpp" FORCE) # Manter comentado a menos que especificamente necessário
# set(LLAMA_METAL OFF CACHE BOOL "Disable Metal for llama.cpp" FORCE) # Manter comentado a menos que especificamente necessário

FetchContent_MakeAvailable(llama_cpp)

# --- Adicionar cpp-httplib como dependência (para o servidor HTTP) ---
FetchContent_Declare(
  httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG        v0.15.3 # Usar uma tag de release estável
  GIT_SHALLOW    TRUE
)
# cpp-httplib é header-only na sua forma mais simples, mas FetchContent_MakeAvailable
# pode criar um target de interface se o CMakeLists.txt dele estiver configurado para isso.
# Se não, apenas ter os headers disponíveis via FetchContent_GetProperties é suficiente.
# No entanto, para garantir que os includes sejam gerenciados corretamente, vamos usar MakeAvailable.
set(HTTPLIB_CLIENT_SUPPORT OFF CACHE BOOL "Disable httplib client support if only server needed" FORCE) # Opcional
set(HTTPLIB_OPENSSL_SUPPORT OFF CACHE BOOL "Disable OpenSSL for httplib if not needed" FORCE) # Opcional
set(HTTPLIB_ZLIB_SUPPORT OFF CACHE BOOL "Disable Zlib for httplib if not needed" FORCE) # Opcional
set(HTTPLIB_BROTLI_SUPPORT OFF CACHE BOOL "Disable Brotli for httplib if not needed" FORCE) # Opcional
FetchContent_MakeAvailable(httplib)

# --- Adicionar nlohmann/json como dependência (para manipulação de JSON na API) ---
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3 # Usar uma tag de release estável
  GIT_SHALLOW    TRUE
)
# nlohmann/json é header-only. FetchContent_MakeAvailable geralmente cria um target INTERFACE.
FetchContent_MakeAvailable(nlohmann_json)


# --- Adicionar subdiretório de testes ---
add_subdirectory(tests)

# --- Linkar nossa biblioteca com llama.cpp ---
# O target da biblioteca principal do llama.cpp é 'llama'.
# Usar PUBLIC aqui para que os diretórios de include do llama.cpp (onde está llama.h)
# sejam propagados para targets que linkam com cpu_llm_lib, como o nosso executável principal
# que inclui llm_engine.hpp (que por sua vez inclui llama.h).
target_link_libraries(cpu_llm_lib PUBLIC llama)

# --- Mensagens de Build ---
message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CXX Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}") # Debug, Release, etc.

# Opcional: Habilitar checagem de AVX em tempo de compilação (simples)
# Isso não garante que o código *use* AVX, apenas que o compilador o suporta.
# Uma verificação em tempo de execução no código C++ seria mais robusta.
# if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     execute_process(
#         COMMAND ${CMAKE_CXX_COMPILER} -mavx -dM -E - < /dev/null
#         OUTPUT_VARIABLE COMPILER_DEFINES
#         RESULT_VARIABLE COMPILER_DEFINES_RESULT
#     )
#     if(COMPILER_DEFINES_RESULT EQUAL 0 AND COMPILER_DEFINES MATCHES ".*__AVX__.*")
#         message(STATUS "AVX support detected in compiler.")
#     else()
#         message(WARNING "AVX support flag (-mavx) might not be effective or supported by the compiler.")
#     endif()
# endif()
