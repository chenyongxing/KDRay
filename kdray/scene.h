#pragma once

#include <optix.h>
#include "texture.h"
#include "mesh.h"
#include "scene_loader.h"

namespace kdray
{
	enum struct LightType
	{
		None = 0,
		Directional = 1,
		Envmap = 2,
		Rect = 3,
		Point = 4,
		Disk = 5,
		Spot = 6
	};
	struct Light
	{
		LightType type = LightType::None;
		float4x4 transform
		{
			1.f, 0.f, 0.f, 0.f,
			0.f, 1.f, 0.f, 0.f,
			0.f, 0.f, 1.f, 0.f,
			0.f, 0.f, 0.f, 1.f
		};;
		float3 radiance{ 1.0f, 1.0f, 1.0f };
		//定向光角直径
		float angularDiameter = 0.53f;
		//矩形光宽高
		float width = 1.0f;
		float height = 1.0f;
		//球/圆盘光半径
		float radius = 1.0f;
		//聚光灯外圆锥角度
		float outerConeAngle = 45.0f;
		float innerConeAngle = 0.0f;
		bool visible = false;
		bool doubleSided = false;
		Texture* texture = nullptr;
	};

	enum struct MaterialType
	{
		Diffuse = 0,
		Metal = 1,
		Plastic = 2,
		Glass = 3,
		Principled = 4
	};
	struct Material
	{
		MaterialType type = MaterialType::Diffuse;
		std::string name;
		// Params
		float3 emission{ 0.0f, 0.0f, 0.0f }; // radiance
		float opacity = 1.0f;
		float3 color{1.0f, 1.0f, 1.0f};
		float roughness = 0.4f;
		float anisotropic = 0.0f;
		float ior = 1.5f;
		// 默认铝AL
		// https://github.com/tunabrain/tungsten/blob/master/src/core/bsdfs/ComplexIorData.hpp
		float3 eta{ 1.6574599595f, 0.8803689579f, 0.5212287346f };
		float3 k{ 9.2238691996f, 6.2695232477f, 4.8370012281f };
		// disney
		float metallic = 0.0f;
		float specular = 0.5f;
		float specularTint = 0.0f;
		float subsurface = 0.0f;
		float sheen = 0.0f;
		float sheenTint = 0.5f;
		float clearcoat = 0.0f;
		float clearcoatGloss = 1.0f;
		float transmission = 0.0f;

		Texture* colorTexture = nullptr;
		Texture* aoRoughMetalTexture = nullptr;
		Texture* normalTexture = nullptr;
		Texture* bumpTexture = nullptr;
	};

	struct Instance
	{
		float transform[16] = 
		{
			1.f, 0.f, 0.f, 0.f,
			0.f, 1.f, 0.f, 0.f,
			0.f, 0.f, 1.f, 0.f,
			0.f, 0.f, 0.f, 1.f
		};
		Mesh* mesh = nullptr;
		Material* material = nullptr;
	};

	class Scene
	{
	public:
		Scene();
		~Scene();

		void AddLight(Light* light);

		void SetLightTransform(Light* light, float transform[16]);

		void AddTexture(Texture* texture);

		void AddMaterial(Material* material);

		Material* GetMaterialByName(const std::string& name);

		MaterialDevice* GetMaterialDeviceAddress(Material* material);

		void AddMesh(Mesh* mesh);

		void AddInstance(Instance* instance);

		void SetInstanceTransform(Instance* instance, float transform[16]);

		void Update();

		void Clear();

		inline bool LoadFromFile(const std::string& filepath)
		{
			return loader->LoadFromFile(filepath);
		}

		inline OptixTraversableHandle GetTraverHandle()
		{
			return _traverHandle;
		}

		inline LightDevice* GetLightsBuffer()
		{
			return reinterpret_cast<LightDevice*>(_lightsBuffer);
		}

		inline uint32_t GetLightsNum()
		{
			return static_cast<uint32_t>(_lights.size());
		}

	private:
		std::vector<Light*> _lights;
		bool _updateLight = false;
		std::vector<Texture*> _textures;
		bool _updateTexture = false;
		std::vector<Material*> _materials;
		bool _updateMaterial = false;
		std::vector<Mesh*> _meshes;
		std::vector<Instance*> _instances;
		bool _rebuildInstanceAccel = false;

		void* _lightsBuffer = nullptr;
		void* _materialsBuffer = nullptr;
		void* _instancesBuffer = nullptr;
		void* _accelBuffer = nullptr;
		OptixTraversableHandle _traverHandle = 0;

		std::unique_ptr<SceneLoader> loader;

		void _BuildInstance();
		void _UpdateLightDevice();
		void _UpdateMaterialDevice();
	};
}
