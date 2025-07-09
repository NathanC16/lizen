# Estágio de Build
FROM ubuntu:22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

# Instalar dependências de build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ccache \
    pkg-config \
    # Dependências para cpp-httplib com SSL (se habilitado no futuro, mas bom ter para compilação)
    # libssl-dev \
    # zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar o código fonte do projeto para o contêiner
COPY . .

# Configurar e compilar o projeto
# Usar Release build para a imagem final.
# As dependências (llama.cpp, httplib, json, catch2) serão baixadas pelo FetchContent do CMake.
# Habilitar ccache para acelerar builds repetidos dentro do Docker (se a camada for cacheada).
RUN echo "Using $(nproc) cores for build" && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache && \
    make -j$(nproc) && \
    ccache -s # Mostrar estatísticas do ccache (opcional, para debug do build)

# Estágio de Runtime
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Instalar dependências de runtime mínimas
# libstdc++6 e libgcc-s1 são geralmente parte da base, mas é bom garantir.
# Se o executável final depender de outras bibliotecas compartilhadas que não são
# estaticamente linkadas (e não são parte do sistema base), elas precisariam ser adicionadas.
# Ex: libpthread (geralmente já presente), libdl.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    libgcc-s1 \
    # ca-certificates # Para fazer chamadas HTTPS do container, se necessário
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar o executável compilado do estágio de build
COPY --from=builder /app/build/bin/cpu_llm_project /app/cpu_llm_project

# Criar um diretório para modelos (o usuário montará os modelos aqui)
RUN mkdir /app/models

# Expor a porta que o servidor usa (padrão 8080)
EXPOSE 8080

# Comando para executar o servidor.
# O usuário deve fornecer o caminho para o modelo GGUF como argumento.
# Exemplo: docker run -p 8080:8080 -v /caminho/local/para/modelos:/app/models nome_da_imagem /app/models/seu_modelo.gguf
# O servidor escutará em 0.0.0.0 por padrão se não especificado, o que é bom para Docker.
ENTRYPOINT ["/app/cpu_llm_project"]

# CMD pode fornecer argumentos padrão para o ENTRYPOINT.
# Se nenhum argumento for passado para `docker run`, o servidor mostrará a ajuda.
CMD ["--help"]
