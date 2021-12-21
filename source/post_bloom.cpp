#include "post_bloom.h"

void PostBloom::init(const sf::Vector2f &resolution)
{
	ShaderLoader::loadFromFile("shaders/post/bloom.frag", sf::Shader::Fragment, shader);
	shader.setUniform("u_resolution", resolution);
}

void PostBloom::apply(const sf::Texture &src_texture, sf::RenderTexture &dst_render, const sf::Sprite &dst_sprite)
{
	shader.setUniform("u_main_tex", src_texture);
	dst_render.draw(dst_sprite, &shader);
}