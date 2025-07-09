#include <catch2/catch_test_macros.hpp>
#include "cpu_llm_project/llm_engine.hpp"
#include <fstream> // Para criar um arquivo dummy GGUF temporário

// Helper para criar um arquivo dummy temporário que se parece com um GGUF (minimamente)
// para testar o carregamento (falha esperada de forma graciosa).
// Um GGUF real é complexo; isto é apenas para simular a existência de um arquivo.
std::string create_dummy_gguf_file(const std::string& filename, bool valid_magic = true) {
    std::ofstream outfile(filename, std::ios::binary);
    if (outfile) {
        if (valid_magic) {
            // Magic number for GGUF v3
            outfile.write("GGUF", 4);
            // Version (e.g., 3)
            uint32_t version = 3;
            outfile.write(reinterpret_cast<const char*>(&version), sizeof(version));
            // Tensor count (e.g., 0 for a very basic dummy)
            uint64_t tensor_count = 0;
            outfile.write(reinterpret_cast<const char*>(&tensor_count), sizeof(tensor_count));
            // Metadata KV count (e.g., 0)
            uint64_t metadata_kv_count = 0;
            outfile.write(reinterpret_cast<const char*>(&metadata_kv_count), sizeof(metadata_kv_count));
        } else {
            outfile.write("NOTGGUF", 7);
        }
        outfile.close();
        return filename;
    }
    return "";
}

TEST_CASE("LlmEngine Initialization and Model Loading", "[llm_engine]") {
    cpu_llm_project::LlmEngine engine;

    SECTION("Engine initializes correctly") {
        REQUIRE_FALSE(engine.is_model_loaded());
        REQUIRE(engine.get_model_path().empty());
    }

    SECTION("Load non-existent model file") {
        REQUIRE_FALSE(engine.load_model("non_existent_model.gguf"));
        REQUIRE_FALSE(engine.is_model_loaded());
    }

    SECTION("Load invalid GGUF (bad magic number)") {
        std::string dummy_file_path = "dummy_invalid.gguf";
        create_dummy_gguf_file(dummy_file_path, false);

        REQUIRE_FALSE(engine.load_model(dummy_file_path));
        REQUIRE_FALSE(engine.is_model_loaded());
        std::remove(dummy_file_path.c_str());
    }

    // Este teste tentará carregar um GGUF "minimamente válido" (cabeçalho)
    // mas o llama.cpp provavelmente falhará ao tentar ler tensores/metadados inexistentes.
    // Esperamos que ele falhe de forma controlada.
    SECTION("Load dummy GGUF (valid magic, but empty)") {
        std::string dummy_file_path = "dummy_valid_empty.gguf";
        create_dummy_gguf_file(dummy_file_path, true);

        // Esta carga deve falhar porque o arquivo, embora tenha o magic number,
        // não é um GGUF completo e válido.
        REQUIRE_FALSE(engine.load_model(dummy_file_path));
        REQUIRE_FALSE(engine.is_model_loaded());
        std::remove(dummy_file_path.c_str());
    }

    // Nota: Testar o carregamento bem-sucedido de um modelo real é mais um teste de integração
    // e requer um arquivo de modelo GGUF real, o que pode ser lento e grande.
    // Esses testes podem ser separados ou marcados para execução condicional.
    // Exemplo (requer um modelo real em "test_model.gguf"):
    /*
    SECTION("Load and unload a real GGUF model (integration test - requires model)") {
        std::string real_model_path = "test_model.gguf"; // Coloque um modelo pequeno aqui para teste
        std::ifstream model_check(real_model_path);
        if (model_check.good()) {
            REQUIRE(engine.load_model(real_model_path));
            REQUIRE(engine.is_model_loaded());
            REQUIRE(engine.get_model_path() == real_model_path);
            engine.unload_model();
            REQUIRE_FALSE(engine.is_model_loaded());
            REQUIRE(engine.get_model_path().empty());
        } else {
            WARN("Skipping real model load test: " << real_model_path << " not found.");
        }
    }
    */
}

TEST_CASE("LlmEngine Prediction Logic (without full model load)", "[llm_engine]") {
    cpu_llm_project::LlmEngine engine;

    SECTION("Predict without a loaded model") {
        std::string result = engine.predict("Hello", 10);
        INFO("Result: " << result); // Para debug, se o teste falhar
        REQUIRE(result.rfind("[Error: Model not loaded]", 0) == 0); // Verifica se começa com a msg de erro
    }

    // Testar predict com um modelo carregado (mesmo que dummy e falhe na geração)
    // seria mais um teste de integração.
    // Aqui, focamos no comportamento da API da classe LlmEngine.
}
