#pragma once
#include "shader_loader.h"

class PostBloom
{
public:
	void init(const sf::Vector2f &resolution);
	void apply(const sf::Texture &src_texture, sf::RenderTexture &dst_render, const sf::Sprite &dst_sprite);

private:
	sf::Shader shader;
};