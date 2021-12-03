#pragma once
#include <SFML/Graphics.hpp>

// shader's preprocessor
// supports:
// - #include<file_name.frag>
// - VAR(), VAR3(), VARI(), VAR3() macro to show and edit variables via UI
class ShaderLoader
{
public:
	static bool loadFromFile(const char *file_name, sf::Shader::Type type, sf::Shader &out_shader);

private:
	static bool preprocess(const char *file_name, std::string &out_string);
};