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

## Configuração (Arquivos YAML de Persona/Modelo)

Este projeto utiliza arquivos de configuração no formato YAML para definir "Personas" ou configurações específicas de modelos. Isso permite gerenciar diferentes modelos, system prompts, e parâmetros de amostragem de forma organizada.

**Estrutura do Arquivo YAML:**

Um arquivo de configuração de persona (ex: `minha_persona.yaml`) deve seguir a estrutura abaixo. Veja `persona_example.yaml` para um exemplo completo.

```yaml
name: "Nome Descritivo da Persona"
model_gguf_path: "/caminho/para/seu/modelo.gguf" # Obrigatório
n_ctx: 2048
num_threads: 0 # 0 para automático
system_prompt: "Este é o prompt de sistema para esta persona."
max_tokens: 256
temperature: 0.7
top_k: 40
top_p: 0.9
repeat_penalty: 1.1
# api_host: "localhost" # Opcional, se esta persona tiver uma config de API específica
# api_port: 8080      # Opcional
```

**Diretório Padrão de Personas:**

Por padrão, ao usar a flag `--run <nome_da_persona>`, o programa procurará por `<nome_da_persona>.yaml` dentro de um diretório chamado `personas/` na raiz do projeto. Crie este diretório se ele não existir e coloque seus arquivos YAML de persona lá.

**Prioridade das Configurações:**
1.  Argumentos de linha de comando (têm a maior prioridade).
2.  Valores definidos no arquivo YAML da persona/modelo carregado.
3.  Valores padrão definidos no código.

(A funcionalidade de listar e criar personas via CLI (`--list`, `--create`) será adicionada em uma fase futura.)

## Como Executar

O `cpu_llm_project` pode ser executado de várias maneiras, dependendo de como você deseja carregar o modelo e suas configurações.

**Argumentos de Linha de Comando Principais:**

*   **Especificador do Modelo/Configuração (um dos seguintes é obrigatório):**
    *   `<caminho_para_config.yaml>`: Caminho direto para um arquivo YAML de configuração de persona/modelo.
    *   `--run <nome_da_persona>`: Carrega a persona `<nome_da_persona>.yaml` do diretório padrão `personas/`.
    *   `<caminho_para_modelo.gguf>`: (Comportamento legado) Caminho direto para um arquivo GGUF. Neste caso, são usados parâmetros padrão para system prompt, amostragem, etc., que podem ser sobrescritos por outras flags.

*   **Flags Opcionais (sobrescrevem valores do YAML ou defaults):**
    *   `--host <hostname>`: Define o host para o servidor API.
    *   `--port <numero_porta>`: Define a porta para o servidor API.
    *   `--n_ctx <numero>`: Define o tamanho do contexto.
    *   `--threads <numero>`: Define o número de threads (0 para automático).
    *   `--interactive`: Força o modo interativo CLI. Tem prioridade sobre as flags de servidor.

### Modo Servidor API

Este modo inicia um servidor HTTP que expõe endpoints para interagir com o modelo LLM. Ele é ativado se argumentos como `--host` ou `--port` são fornecidos, ou se argumentos posicionais para host/porta são detectados e a flag `--interactive` não está presente.

**Comando (Exemplos):**
```bash
# Usando valores padrão para host/porta/n_ctx/threads
./build/bin/cpu_llm_project /caminho/para/seu/modelo.gguf --host 0.0.0.0 --port 8008

# Especificando todos os parâmetros
./build/bin/cpu_llm_project /caminho/para/seu/modelo.gguf --host localhost --port 8080 --n_ctx 4096 --threads 4

# Argumentos posicionais para host e porta (menos recomendado se misturar com muitas flags)
./build/bin/cpu_llm_project /caminho/para/seu/modelo.gguf localhost 8081
```
*   `/caminho/para/seu/modelo.gguf`: **Obrigatório.** Caminho para o arquivo do modelo GGUF.

O servidor começará a escutar no host e porta especificados. Consulte a seção "Como Usar a API" para detalhes sobre os endpoints.

### Modo Interativo (CLI)

Este modo permite que você converse diretamente com o modelo através da linha de comando.

**Como Iniciar:**

O modo interativo é ativado se:
1.  A flag `--interactive` é fornecida.
2.  Nenhum argumento de modo servidor (como `--host` ou `--port`) é fornecido, e apenas o caminho do modelo é passado.

**Exemplos de Comando:**
```bash
# Modo interativo padrão (apenas modelo)
./build/bin/cpu_llm_project /caminho/para/seu/modelo.gguf

# Modo interativo explícito
./build/bin/cpu_llm_project /caminho/para/seu/modelo.gguf --interactive

# Modo interativo com número de threads customizado
./build/bin/cpu_llm_project /caminho/para/seu/modelo.gguf --threads 8 --interactive
# ou
./build/bin/cpu_llm_project /caminho/para/seu/modelo.gguf --interactive --threads 8
```

**Como Usar:**

Após iniciar, você verá uma mensagem de boas-vindas e um prompt `Prompt: `.
```
CPU LLM Project - Início
Info: Suporte a AVX detectado em tempo de execução.
[...]
Modelo /caminho/para/seu/modelo.gguf carregado com sucesso no LlmEngine.

Modo Interativo. Digite '//sair', '//exit' ou '//quit' para terminar.

Prompt:
```
*   Digite seu prompt e pressione Enter.
*   Para executar comandos, use o prefixo `//`. Exemplo: `//sair`.
*   Comandos de saída disponíveis: `//sair`, `//exit`, `//quit`. Você também pode usar `Ctrl+D` (EOF).

**Nota sobre a Geração de Texto:**
Atualmente, a funcionalidade de geração de texto no `LlmEngine` está simplificada para garantir a compilação do projeto (devido a desafios com a API `llama.cpp`). No modo interativo, a "resposta" do modelo será uma mensagem informativa estática: `[INFO: Text generation loop disabled for compilation. Processed prompt.]`. A restauração da capacidade completa de geração de texto e amostragem avançada é um trabalho futuro.

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
