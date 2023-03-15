#include "scene.h"
#include "optix_context.h"
#include "logger.h"
#include "renderer.h"
#include <algorithm>

namespace kdray
{
	Scene::Scene()
	{
		loader = std::make_unique<SceneLoader>(this);
	}

	Scene::~Scene()
	{
		Clear();
	}

	void Scene::AddLight(Light* light)
	{
		if (!light) return;

		_lights.emplace_back(light);
		_updateLight = true;
	}

	void Scene::SetLightTransform(Light* light, float transform[16])
	{
		light->transform = make_float4x4
		(
			transform[0], transform[1], transform[2], transform[3],
			transform[4], transform[5], transform[6], transform[7],
			transform[8], transform[9], transform[10], transform[11],
			transform[12], transform[13], transform[14], transform[15]
		);
		_updateLight = true;
	}

	void Scene::AddTexture(Texture* texture)
	{
		if (!texture) return;

		_textures.emplace_back(texture);
	}
	
	void Scene::AddMaterial(Material* material)
	{
		if (!material) return;

		_materials.emplace_back(material);
		_updateMaterial = true;
	}

	Material* Scene::GetMaterialByName(const std::string& name)
	{
		auto iter = std::find_if(_materials.begin(), _materials.end(), 
			[&](const Material* mat)
			{
				return mat->name == name;
			});
		if (iter != _materials.end())	return *iter;

		return nullptr;
	}

	MaterialDevice* Scene::GetMaterialDeviceAddress(Material* material)
	{
		auto iter = std::find(_materials.begin(), _materials.end(), material);
		size_t index = std::distance(_materials.begin(), iter);
		MaterialDevice* headPtr = reinterpret_cast<MaterialDevice*>(_materialsBuffer);
		return headPtr + index;
	}

	void Scene::AddMesh(Mesh* mesh)
	{
		if (!mesh) return;

		_meshes.emplace_back(mesh);
	}

	void Scene::AddInstance(Instance* instance)
	{
		if (!instance) return;

		_instances.emplace_back(instance);
		_rebuildInstanceAccel = true;
	}

	void Scene::SetInstanceTransform(Instance* instance, float _transform[16])
	{
		if (!instance) return;
		memcpy(instance->transform, _transform, sizeof(float) * 16);

		if (!_instancesBuffer) return;
		auto iter = std::find(_instances.begin(), _instances.end(), instance);
		if (iter != _instances.cend())
		{
			size_t index = std::distance(_instances.begin(), iter);
			size_t offsetInstance = sizeof(OptixInstance) * index;
			size_t offsetTransform = offsetof(OptixInstance, transform);
			size_t offset = offsetInstance + offsetTransform;
			unsigned char* devicePtr = reinterpret_cast<unsigned char*>(_instancesBuffer);

			float transform[16];
			memcpy(transform, _transform, sizeof(float) * 16);
			std::swap(transform[1], transform[4]);
			std::swap(transform[2], transform[8]);
			std::swap(transform[3], transform[12]);
			std::swap(transform[6], transform[9]);
			std::swap(transform[7], transform[13]);
			std::swap(transform[11], transform[14]);
			cudaMemcpy(devicePtr + offset, transform, sizeof(float) * 12, cudaMemcpyHostToDevice);
		}
	}

	void Scene::Update()
	{
		if (_updateLight)
		{
			_UpdateLightDevice();
			_updateLight = false;
		}

		if (_updateMaterial)
		{
			_UpdateMaterialDevice();
			_updateMaterial = false;
		}

		if (_rebuildInstanceAccel)
		{
			_BuildInstance();
			_rebuildInstanceAccel = false;
		} 
	}

	void Scene::Clear()
	{
		for (auto& light : _lights)
		{
			delete light;
		}
		_lights.clear();
		for (auto& texture : _textures)
		{
			delete texture;
		}
		_textures.clear();
		for (auto& material : _materials)
		{
			delete material;
		}
		_materials.clear();
		for (auto& mesh : _meshes)
		{
			delete mesh;
		}
		_meshes.clear();
		for (auto& instance : _instances)
		{
			delete instance;
		}
		_instances.clear();

		CudaFree(_accelBuffer);
		_accelBuffer = nullptr;
		CudaFree(_instancesBuffer);
		_instancesBuffer = nullptr;
		CudaFree(_materialsBuffer);
		_materialsBuffer = nullptr;
		CudaFree(_lightsBuffer);
		_lightsBuffer = nullptr;
		_traverHandle = 0;
	}

	void Scene::_BuildInstance()
	{
		CudaFree(_accelBuffer);
		_accelBuffer = nullptr;
		CudaFree(_instancesBuffer);
		_instancesBuffer = nullptr;

		std::vector<OptixInstance> optixInstances(_instances.size());
		for (size_t i = 0; i < _instances.size(); i++)
		{
			Instance* instance = _instances[i];
			OptixInstance& optixInstance = optixInstances[i];
			memset(&optixInstance, 0, sizeof(OptixInstance));
			optixInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
			optixInstance.instanceId = (unsigned int)i;
			optixInstance.sbtOffset = (unsigned int)i;
			optixInstance.visibilityMask = 0xFF;
			optixInstance.traversableHandle = instance->mesh->GetTraverHandle();

			float transform[16];
			memcpy(transform, instance->transform, sizeof(float) * 16);
			std::swap(transform[1], transform[4]);
			std::swap(transform[2], transform[8]);
			std::swap(transform[3], transform[12]);
			std::swap(transform[6], transform[9]);
			std::swap(transform[7], transform[13]);
			std::swap(transform[11], transform[14]);
			memcpy(optixInstance.transform, transform, sizeof(float) * 12);
		}
		_instancesBuffer = CudaMalloc(optixInstances);

		// instance input info
		OptixBuildInput instanceInput = {};
		instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
		instanceInput.instanceArray.instances = reinterpret_cast<CUdeviceptr>(_instancesBuffer);
		instanceInput.instanceArray.numInstances = static_cast<unsigned int>(optixInstances.size());
		auto optixCtx = Renderer::Instance()->GetOptixContext();
		optixCtx->BuildAccelInstance(instanceInput, &_accelBuffer, &_traverHandle);

		std::vector<HitGroupData> hitGroupDatas(_instances.size());
		for (size_t i = 0; i < _instances.size(); i++)
		{
			Instance* instance = _instances[i];
			HitGroupData& hitGroupData = hitGroupDatas[i];
			hitGroupData.vertexBuffer = instance->mesh->GetVertexBuffer();
			hitGroupData.indexBuffer = instance->mesh->GetIndexBuffer();
			hitGroupData.material = this->GetMaterialDeviceAddress(instance->material);
		}
		optixCtx->BuildHitGroupSBT(hitGroupDatas);
	}

	void Scene::_UpdateLightDevice()
	{
		std::vector<LightDevice> lightDevices;
		lightDevices.reserve(_lights.size());
		for (auto iter = _lights.begin(); iter != _lights.end(); iter++)
		{
			Light* light = *iter;
			LightDevice lightDevice;
			lightDevice.type = (uint32_t)light->type;
			lightDevice.localToWorldMatrix = light->transform;
			lightDevice.radiance = light->radiance;
			lightDevice.visible = light->visible;
			lightDevice.doubleSided = light->doubleSided;
			constexpr float minRadius = 0.05f;
			if (light->type == LightType::Directional)
			{
				light->angularDiameter = clamp(light->angularDiameter, 0.f, 180.f);
				lightDevice.cosAngle = cosf(radian(0.5f * light->angularDiameter));
			}
			else if (light->type == LightType::Envmap)
			{
				lightDevice.textureWidth = light->texture->GetWidth();
				lightDevice.textureHeight = light->texture->GetHeight();
				lightDevice.texture = light->texture->cuTexture;
				lightDevice.distributionMarginal = light->texture->distributionMarginal;
				lightDevice.distributionConditional = light->texture->distributionConditional;
			}
			else if (light->type == LightType::Rect)
			{
				lightDevice.corner = make_float3(light->width * -0.5f, light->height * -0.5f, 0.0f);
				lightDevice.corner = transform_point(lightDevice.localToWorldMatrix, lightDevice.corner);
				lightDevice.u = make_float3(light->width, 0.0f, 0.0f);
				lightDevice.u = transform_direction(lightDevice.localToWorldMatrix, lightDevice.u);
				lightDevice.v = make_float3(0.0f, light->height, 0.0f);
				lightDevice.v = transform_direction(lightDevice.localToWorldMatrix, lightDevice.v);
				float area = length(cross(lightDevice.u, lightDevice.v));
				lightDevice.invArea = 1.0f / area;
			}
			else if (light->type == LightType::Point)
			{
				light->radius = std::max(light->radius, minRadius);
				lightDevice.radius = light->radius;
				float area = 4.0f * M_PIf * light->radius * light->radius;
				lightDevice.invArea = 1.0f / area;
			}
			else if (light->type == LightType::Disk)
			{
				light->radius = std::max(light->radius, minRadius);
				lightDevice.radius = light->radius;
				float area = M_PIf * light->radius * light->radius;
				lightDevice.invArea = 1.0f / area;
			}
			else if (light->type == LightType::Spot)
			{
				light->radius = std::max(light->radius, minRadius);
				light->outerConeAngle = std::min(light->outerConeAngle, 179.9f);
				light->innerConeAngle = std::min(light->innerConeAngle, light->outerConeAngle);
				lightDevice.radius = light->radius;
				lightDevice.cosAngle = cosf(radian(0.5f * light->outerConeAngle));
				lightDevice.cosAngleScale = 1.0f / (cosf(radian(0.5f * light->innerConeAngle)) - lightDevice.cosAngle);
				float area = 4.0f * M_PIf * light->radius * light->radius;
				lightDevice.invArea = 1.0f / area;
			}
			lightDevices.emplace_back(lightDevice);
		}
		CudaFree(_lightsBuffer);
		_lightsBuffer = CudaMalloc(lightDevices);
	}

	void Scene::_UpdateMaterialDevice()
	{
		std::vector<MaterialDevice> materialDevices;
		materialDevices.reserve(_materials.size());
		for (auto iter = _materials.begin(); iter != _materials.end(); iter++)
		{
			Material* material = *iter;
			MaterialDevice materialDevice;
			materialDevice.state.radiance = material->emission;
			materialDevice.state.opacity = material->opacity;
			materialDevice.state.type = (uint32_t)material->type;
			materialDevice.state.color = material->color;
			materialDevice.state.roughness = clamp(0.001f, material->roughness, 1.0f);
			materialDevice.state.anisotropic = clamp(0.0f, material->anisotropic, 1.0f);
			materialDevice.state.condEta = material->eta;
			materialDevice.state.condK = material->k;
			materialDevice.state.metallic = clamp(0.0f, material->metallic, 1.0f);
			materialDevice.state.specular = clamp(0.0f, material->specular, 1.0f);
			materialDevice.state.specularTint = clamp(0.0f, material->specularTint, 1.0f);
			materialDevice.state.subsurface = clamp(0.0f, material->subsurface, 1.0f);
			materialDevice.state.sheen = clamp(0.0f, material->sheen, 1.0f);
			materialDevice.state.sheenTint = clamp(0.0f, material->sheenTint, 1.0f);
			materialDevice.state.clearcoat = clamp(0.0f, material->clearcoat, 1.0f);
			materialDevice.state.clearcoatRoughness = clamp(0.001f, 1.0f - material->clearcoatGloss, 1.0f);
			materialDevice.state.transmission = clamp(0.0f, material->transmission, 1.0f);
			materialDevice.ior = material->ior;
			if (material->colorTexture)
			{
				materialDevice.colorTexture.texture = material->colorTexture->cuTexture;
				materialDevice.colorTexture.offset = material->colorTexture->offset;
				materialDevice.colorTexture.scale = material->colorTexture->scale;
			}
			if (material->aoRoughMetalTexture)
			{
				materialDevice.aoRoughMetalTexture.texture = material->aoRoughMetalTexture->cuTexture;
				materialDevice.aoRoughMetalTexture.offset = material->aoRoughMetalTexture->offset;
				materialDevice.aoRoughMetalTexture.scale = material->aoRoughMetalTexture->scale;
			}
			if (material->normalTexture)
			{
				materialDevice.normalTexture.texture = material->normalTexture->cuTexture;
				materialDevice.normalTexture.offset = material->normalTexture->offset;
				materialDevice.normalTexture.scale = material->normalTexture->scale;
			}
			if (material->bumpTexture)
			{
				materialDevice.bumpTexture.texture = material->bumpTexture->cuTexture;
				materialDevice.bumpTexture.offset = material->bumpTexture->offset;
				materialDevice.bumpTexture.scale = material->bumpTexture->scale;
			}
			materialDevices.emplace_back(materialDevice);
		}
		CudaFree(_materialsBuffer);
		_materialsBuffer = CudaMalloc(materialDevices);
	}

}
