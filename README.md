# CPU LLM Project (Nome Provisório)

Este projeto visa ser uma alternativa ao Ollama, otimizado para inferência eficiente de Modelos de Linguagem Grandes (LLMs) em CPUs que suportam instruções AVX. Ele expõe uma API HTTP para interações.

## Objetivos
*   Fornecer uma ferramenta de alta performance para rodar LLMs em CPUs comuns.
*   Ser compatível com processadores que possuem AVX (com otimizações para AVX2/AVX512 se disponíveis no momento da compilação do `llama.cpp`).
*   Oferecer uma API HTTP inspirada no Ollama para facilidade de uso.
*   Suportar o formato de modelo GGUF.

## Status Atual
*   Core de inferência implementado usando `llama.cpp`.
*   Servidor HTTP básico com endpoint `/api/generate`.
*   Testes unitários para a estrutura inicial.

## Pré-requisitos
*   Compilador C++ com suporte a C++17 (ex: GCC, Clang, MSVC)
*   CMake (versão 3.16+)
*   Git
*   (Opcional) `ccache` para acelerar rebuilds.
*   (Opcional) Docker para build e execução em contêiner.

## Como Construir (Linux)

1.  **Clonar o repositório:**
    ```bash
    git clone <url_do_repositorio>
    cd cpu-llm-project
    ```

2.  **Configurar com CMake e Compilar:**
    ```bash
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release # Usar Release para performance
    make -j$(nproc) # Compilar usando todos os cores disponíveis
    ```
    O executável principal será `build/bin/cpu_llm_project`.

## Como Executar o Servidor API

Após a compilação, você pode iniciar o servidor API:

```bash
./build/bin/cpu_llm_project /caminho/para/seu/modelo.gguf [host] [porta] [n_ctx]
```
*   `/caminho/para/seu/modelo.gguf`: **Obrigatório.** Caminho para o arquivo do modelo GGUF.
*   `[host]`: Opcional. Host para o servidor (padrão: `localhost`).
*   `[porta]`: Opcional. Porta para o servidor (padrão: `8080`).
*   `[n_ctx]`: Opcional. Tamanho do contexto para o modelo (padrão: `2048`).

**Exemplo:**
```bash
./build/bin/cpu_llm_project ./models/phi-2.Q4_K_M.gguf localhost 8080
```

O servidor começará a escutar em `http://localhost:8080` (ou no host/porta especificados).

## Como Usar a API

### Endpoint `/api/generate` (POST)

Envie uma requisição POST para gerar texto.

**Request Body (JSON):**
```json
{
  "prompt": "Qual é a capital da França?",
  "max_tokens": 64,
  "temperature": 0.7,
  "top_k": 40,
  "top_p": 0.9,
  "repeat_penalty": 1.1
}
```
*   `prompt` (string, obrigatório): O prompt para o modelo.
*   `max_tokens` (int, opcional, padrão: 128): Número máximo de tokens a serem gerados.
*   `temperature` (float, opcional, padrão: 0.8): Controla a aleatoriedade. Valores mais baixos são mais determinísticos.
*   `top_k` (int, opcional, padrão: 40): Amostragem Top-K.
*   `top_p` (float, opcional, padrão: 0.9): Amostragem Nucleus (Top-P).
*   `repeat_penalty` (float, opcional, padrão: 1.1): Penalidade para repetição de tokens.

**Exemplo com `curl`:**
```bash
curl -X POST http://localhost:8080/api/generate -d '{
  "prompt": "Traduza para o francês: Olá, mundo!",
  "max_tokens": 50
}'
```

**Response Body (JSON):**
```json
{
  "model": "/caminho/para/seu/modelo.gguf",
  "created_at": "timestamp_iso8601",
  "response": "Bonjour le monde!",
  "done": true
}
```

### Endpoint `/health` (GET)
Verifica a saúde do servidor.
```bash
curl http://localhost:8080/health
```
**Response Body (JSON):**
```json
{
  "status": "ok"
}
```

## Docker
Consulte o arquivo `Dockerfile` para construir e executar em um contêiner Docker.

## Contribuições
Contribuições são bem-vindas. Por favor, abra uma issue para discutir mudanças propostas.

## Licença
Este projeto é licenciado sob a Licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
