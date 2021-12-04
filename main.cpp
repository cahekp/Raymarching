#include <random>
#include <imgui/imgui.h>
#include <imgui/imgui-SFML.h>
#include <SFML/Graphics.hpp>
#include "source/shader_loader.h"

// math
float clamp01(float v) { return (v < 0.0f) ? 0.0f : ((v > 1.0f) ? 1.0f : v); }
float lerp(float v0, float v1, float k) { return v0 + (v1 - v0) * clamp01(k); }

int main()
{
	int w = 1600;
	int h = 900;
	float wf = static_cast<float>(w);
	float hf = static_cast<float>(h);
	int mouseX = w / 2;
	int mouseY = h / 2;
	float mouseSensitivity = 3.0f;
	float speed = 0.1f;
	bool mouseHidden = true;
	bool wasdUD[6] = { false, false, false, false, false, false };
	sf::Vector3f pos = sf::Vector3f(2.0f, 3.0f, 3.0f);
	sf::Clock clock;
	int framesStill = 1;
	bool firstFrame = true;

	sf::RenderWindow window(sf::VideoMode(w, h), "Ray Marching", sf::Style::Titlebar | sf::Style::Close);
	window.setFramerateLimit(60);
	window.setMouseCursorVisible(false);
	ImGui::SFML::Init(window);

	sf::RenderTexture firstTexture;
	firstTexture.create(w, h);
	sf::Sprite firstTextureSprite = sf::Sprite(firstTexture.getTexture());
	sf::Sprite firstTextureSpriteFlipped = sf::Sprite(firstTexture.getTexture());
	firstTextureSpriteFlipped.setScale(1, -1);
	firstTextureSpriteFlipped.setPosition(0.0f, hf);

	sf::RenderTexture outputTexture;
	outputTexture.create(w, h);
	sf::Sprite outputTextureSprite = sf::Sprite(outputTexture.getTexture());
	sf::Sprite outputTextureSpriteFlipped = sf::Sprite(firstTexture.getTexture());
	outputTextureSpriteFlipped.setScale(1, -1);
	outputTextureSpriteFlipped.setPosition(0.0f, hf);

	sf::RenderTexture postTexture;
	postTexture.create(w, h);
	sf::Sprite postTextureSprite(postTexture.getTexture());

	sf::Shader shader;
	ShaderLoader::loadFromFile("output_shader.frag", sf::Shader::Fragment, shader);
	shader.setUniform("u_resolution", sf::Vector2f(wf, hf));

	sf::Shader post_shader;
	ShaderLoader::loadFromFile("post.frag", sf::Shader::Fragment, post_shader);
	post_shader.setUniform("u_resolution", sf::Vector2f(wf, hf));

	std::random_device rd;
	std::mt19937 e2(rd());
	std::uniform_real_distribution<> dist(0.0f, 1.0f);

	sf::Clock delta_clock;
	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			ImGui::SFML::ProcessEvent(window, event);
			ImGuiIO &io = ImGui::GetIO();
			if (mouseHidden)
			{
				ImGui::SetMouseCursor(ImGuiMouseCursor_None);
			}

			if (event.type == sf::Event::Closed)
			{
				window.close();
			}
			else if (!io.WantCaptureMouse && event.type == sf::Event::MouseMoved)
			{
				if (mouseHidden)
				{
					int mx = event.mouseMove.x - w / 2;
					int my = event.mouseMove.y - h / 2;
					mouseX += mx;
					mouseY += my;
					sf::Mouse::setPosition(sf::Vector2i(w / 2, h / 2), window);
					if (mx != 0 || my != 0) framesStill = 1;
				}
			}
			else if (!io.WantCaptureMouse && event.type == sf::Event::MouseButtonPressed)
			{
				if (!mouseHidden) framesStill = 1;
				window.setMouseCursorVisible(false);
				mouseHidden = true;
			}
			else if (event.type == sf::Event::KeyPressed)
			{
				if (event.key.code == sf::Keyboard::Escape)
				{
					window.setMouseCursorVisible(true);
					mouseHidden = false;
				}
				else if (event.key.code == sf::Keyboard::W) wasdUD[0] = true;
				else if (event.key.code == sf::Keyboard::A) wasdUD[1] = true;
				else if (event.key.code == sf::Keyboard::S) wasdUD[2] = true;
				else if (event.key.code == sf::Keyboard::D) wasdUD[3] = true;
				else if (event.key.code == sf::Keyboard::Space) wasdUD[4] = true;
				else if (event.key.code == sf::Keyboard::LShift) wasdUD[5] = true;
			}
			else if (event.type == sf::Event::KeyReleased)
			{
				if (event.key.code == sf::Keyboard::W) wasdUD[0] = false;
				else if (event.key.code == sf::Keyboard::A) wasdUD[1] = false;
				else if (event.key.code == sf::Keyboard::S) wasdUD[2] = false;
				else if (event.key.code == sf::Keyboard::D) wasdUD[3] = false;
				else if (event.key.code == sf::Keyboard::Space) wasdUD[4] = false;
				else if (event.key.code == sf::Keyboard::LShift) wasdUD[5] = false;
			}
		}

		sf::Time delta_time = delta_clock.restart();
		ImGui::SFML::Update(window, delta_time);
		static float smooth_fps = 60.0f;
		smooth_fps = lerp(smooth_fps, 1 / delta_time.asSeconds(), 5.0f * delta_time.asSeconds());
		ImGui::Text("FPS: %.1f", smooth_fps);
		if (ImGui::Button("Reload shader"))
		{
			ShaderLoader::loadFromFile("output_shader.frag", sf::Shader::Fragment, shader);
			shader.setUniform("u_resolution", sf::Vector2f(wf, hf));
			firstFrame = true;
		}
		if (ImGui::Button("Reload post shader"))
		{
			ShaderLoader::loadFromFile("post.frag", sf::Shader::Fragment, post_shader);
			post_shader.setUniform("u_resolution", sf::Vector2f(wf, hf));
			firstFrame = true;
		}

		if (mouseHidden || firstFrame)
		{
			firstFrame = false;

			float mx = ((float)mouseX / w - 0.5f) * mouseSensitivity;
			float my = ((float)mouseY / h - 0.5f) * mouseSensitivity;
			sf::Vector3f dir = sf::Vector3f(0.0f, 0.0f, 0.0f);
			sf::Vector3f dirTemp;
			dir += sf::Vector3f(0.0f, 0.0f, static_cast<float>(wasdUD[2] - wasdUD[0])); // W/S - Forward/Backward
			dir += sf::Vector3f(static_cast<float>(wasdUD[3] - wasdUD[1]), 0.0f, 0.0f); // A/D - Left/Right
			dir += sf::Vector3f(0.0f, static_cast<float>(wasdUD[4] - wasdUD[5]), 0.0f); // Space/Shift - Up/Down

			// rotate YZ plane
			dirTemp.y = dir.y * cos(-my) - dir.z * sin(-my);
			dirTemp.z = dir.y * sin(-my) + dir.z * cos(-my);
			dirTemp.x = dir.x;

			// rotate XZ plane
			dir.x = dirTemp.x * cos(mx) - dirTemp.z * sin(mx);
			dir.z = dirTemp.x * sin(mx) + dirTemp.z * cos(mx);
			dir.y = dirTemp.y;

			pos += dir * speed;

			// right handed coordinate system
			// x - right
			// y - up
			// z - back
			// rotation order: XZ -> YZ -> XY

			for (int i = 0; i < 6; i++)
			{
				if (wasdUD[i])
				{
					framesStill = 1;
					break;
				}
			}
			shader.setUniform("u_pos", pos);
			shader.setUniform("u_mouse", sf::Vector2f(mx, my));
		}
		shader.setUniform("u_time", clock.getElapsedTime().asSeconds());
		shader.setUniform("u_sample_part", 1.0f / framesStill);
		shader.setUniform("u_seed1", sf::Vector2f((float)dist(e2), (float)dist(e2)) * 999.0f);
		shader.setUniform("u_seed2", sf::Vector2f((float)dist(e2), (float)dist(e2)) * 999.0f);

		if (framesStill % 2 == 1)
		{
			shader.setUniform("u_sample", firstTexture.getTexture());
			outputTexture.draw(firstTextureSpriteFlipped, &shader);
			//window.draw(outputTextureSprite);
		}
		else
		{
			shader.setUniform("u_sample", outputTexture.getTexture());
			firstTexture.draw(outputTextureSpriteFlipped, &shader);
			//window.draw(firstTextureSprite);
		}

		post_shader.setUniform("u_main_tex", framesStill % 2 == 1 ? outputTexture.getTexture() : firstTexture.getTexture());
		postTexture.draw(postTextureSprite, &post_shader);
		window.draw(postTextureSprite);

		ImGui::SFML::Render(window);

		window.display();
		framesStill++;
	}

	ImGui::SFML::Shutdown();
	return 0;
}