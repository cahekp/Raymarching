#include "shader_loader.h"
#include <string>
#include <iostream>
#include <fstream>

using namespace std;

bool ShaderLoader::loadFromFile(const char *file_name, sf::Shader::Type type, sf::Shader &out_shader)
{
	// load file and replace "#include", "VAR" etc.
	string shader_code;
	if (!preprocess(file_name, shader_code))
		return false;

	// debug
	// cout << shader_code << endl;

	// convert processed file into shader
	return out_shader.loadFromMemory(shader_code, type);
}

bool ShaderLoader::preprocess(const char *file_name, string &out_string)
{
	// try to open shader's file
	fstream file(file_name);
	if (!file.is_open())
	{
		cerr << "ShaderLoader: can't load file \"" << file_name << "\"" << endl;
		return false;
	}

	// read the file
	string line;
	while (getline(file, line))
	{
		// find #include<file_name.frag>
		std::size_t found = line.find("#include");
		if (found != std::string::npos)
		{
			// is this just a comment?
			if (found == 0 || line.rfind("//", found) == std::string::npos)
			{
				// okay, this is normal line
				// let's try to read file name!
				string file_str;
				bool reading_str = false;
				for (int i = found + 8; i < line.size(); i++)
				{
					char c = line[i];
					if (c == '"' || c == '<')
						reading_str = true;
					else if (reading_str)
					{
						if (c == '"' || c == '>')
							reading_str = false;
						else
							file_str += c;
					}
				}

				// debug
				// cout << "INCLUDE FILE: " << file_str.c_str() << endl;

				// load this include!
				if (!preprocess(file_str.c_str(), out_string))
				{
					file.close();
					return false;
				}
				else
					continue;
			}

		}

		// this is a code. load "as is"
		out_string += line + '\n';
	}
	file.close();
	return true;
}