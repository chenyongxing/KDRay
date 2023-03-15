#pragma once

#include <string>

namespace kdray
{
	class Scene;
	class SceneLoader
	{
	public:
		SceneLoader(Scene* scene);
		bool LoadFromFile(const std::string& filepath);

	private:
		Scene* _scene = nullptr;
	};
}
