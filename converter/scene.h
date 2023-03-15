#pragma once

#include <string>
#include <vector>

namespace kdray
{
	struct Camera
	{
		std::string type;
		std::string width;
		std::string height;
		std::string fovY;
		std::string transformMatrix;
	};

	struct Light
	{
		std::string type;
		std::string radiance;
		std::string transformMatrix;
		std::string doubleSided;
		//type = rect
		std::string width;
		std::string height;
		//type = point
		std::string radius;
		//type = envmap
		std::string texture;
	};

	struct Material
	{
		std::string name;
		std::string type;
		std::string color;
		std::string roughness;
		std::string ior;
		// metal
		std::string eta;
		std::string k;
		// texture
		std::string colorTexture;
		std::string normalTexture;
		std::string bumpTexture;
	};

	struct Mesh
	{
		std::string filepath;
		std::string material;
		std::string transformMatrix;
	};

	struct Scene
	{
		Camera camera;
		std::vector<Light> lights;
		std::vector<Material> materials;
		std::vector<Mesh> meshes;
	};
}