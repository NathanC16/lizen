## Instruções para Jules (Agente de IA)

Este projeto visa criar uma alternativa ao Ollama, otimizada para inferência de LLMs em CPUs com suporte a AVX (especificamente, sem exigir AVX2). O nome provisório do projeto é "CPU LLM Project".

**Linguagens e Ferramentas:**
*   **Core:** C++ (com foco em performance e otimizações AVX).
*   **Build System:** CMake.
*   **Testes:** Framework Catch2 para testes C++.
*   **API/CLI:** Python (a ser adicionado posteriormente).
*   **Controle de Versão:** Git.

**Princípios de Desenvolvimento:**
1.  **Foco em AVX:** O código C++ deve ser otimizado para AVX. Se possível, deve haver verificações ou caminhos de código que garantam que as instruções AVX sejam utilizadas, mas que o projeto não compile ou falhe em tempo de execução se apenas SSE estiver disponível (a menos que explicitamente decidido de outra forma). O requisito mínimo é AVX. AVX2 não deve ser um requisito.
2.  **Performance:** A performance é crucial. Benchmarks serão importantes.
3.  **Modularidade:** Tente manter o código modular (ex: core de inferência, manipulação de modelos, API).
4.  **Testes:** Escreva testes unitários e de integração para garantir a corretude.
5.  **Documentação:** Comente o código e mantenha o README.md atualizado.
6.  **Inspiração em Ollama:** A interface do usuário (CLI) e a API devem ser inspiradas no Ollama para familiaridade.
7.  **Sem dependências desnecessárias:** Mantenha o número de dependências externas baixo, especialmente para o core de C++.

**Configuração de Build (CMake):**
*   Garanta que as flags de compilação para AVX estejam ativadas (ex: `-mavx`).
*   O projeto deve ser fácil de compilar.
*   Inclua o Catch2 para testes de forma apropriada (ex: via `FetchContent`).

**Próximas Etapas (após configuração inicial):**
1.  Implementar o core de inferência, começando com funcionalidades básicas.
2.  Integrar uma biblioteca como GGML (possivelmente um fork customizado) ou começar a desenvolver os componentes de álgebra linear necessários, otimizados para AVX.
3.  Desenvolver a capacidade de carregar modelos (provavelmente no formato GGUF).

**Workflow de CI (GitHub Actions):**
*   O workflow deve compilar o projeto em um ambiente Linux.
*   Deve rodar os testes.
*   Idealmente, verificar se as otimizações AVX estão sendo usadas (isso pode ser complexo de automatizar no CI inicialmente).

Lembre-se de seguir o plano estabelecido e pedir feedback se encontrar ambiguidades ou bloqueios significativos.
