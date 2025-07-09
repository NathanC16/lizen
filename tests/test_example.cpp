#define CATCH_CONFIG_MAIN // Isso diz ao Catch para fornecer seu próprio main() - só faça isso em UM arquivo .cpp
#include <catch2/catch_test_macros.hpp>

// Inclua headers do projeto principal que você quer testar
#include "cpu_llm_project/dummy.hpp" // Incluindo o header da nossa biblioteca

// Testes de exemplo do Catch2 (fatorial)
unsigned int Factorial( unsigned int number ) {
    return number <= 1 ? number : Factorial(number-1)*number;
}

TEST_CASE( "Factorials are computed", "[factorial]" ) {
    REQUIRE( Factorial(1) == 1 );
    REQUIRE( Factorial(2) == 2 );
    REQUIRE( Factorial(3) == 6 );
    REQUIRE( Factorial(10) == 3628800 );
}

// Teste para a função get_greeting da nossa biblioteca
TEST_CASE( "Greeting function works", "[greeting]" ) {
    REQUIRE( cpu_llm_project::get_greeting("World") == "Hello, World!" );
    REQUIRE( cpu_llm_project::get_greeting("Catch2") == "Hello, Catch2!" );
    REQUIRE( cpu_llm_project::get_greeting("") == "Hello, there!" ); // Teste com string vazia
}

TEST_CASE( "Basic true test (sanity check)", "[sanity]" ) {
    REQUIRE( true == true );
}
