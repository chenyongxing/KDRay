#pragma once

#include "math.h"
#include <optix.h>

namespace kdray
{
	struct MeshVertex
	{
		float3 position{ 0.0f };
		float3 normal{ 0.0f };
		float2 texCoord{ 0.0f };
	};

	struct LightDevice
	{
		uint32_t type = 0;
		float4x4 localToWorldMatrix;
		float3 radiance;
		bool visible = false;
		bool doubleSided = false;
		//cos半角直径，圆锥采样输入
		float cosAngle = 1.0f; //cos(deg2rad(0.5f * angularDiameter))
		float cosAngleScale = 0.5f; //1/(cos(innerConeAngle) - cos(outerConeAngle));
		float radius = 1.0f;
		float invArea = 1.0f; //pdf=1/area
		//rect
		float3 corner;
		float3 u;
		float3 v;
		// envmap
		int textureWidth = 0;
		int textureHeight = 0;
		cudaTextureObject_t texture = 0;
		float* distributionMarginal = nullptr;
		float* distributionConditional = nullptr;
	};
	
	struct TextureDevice
	{
		cudaTextureObject_t texture = 0;
		float2 scale{ 1.0f, 1.0f };
		float2 offset{ 0.0f, 0.0f };
	};

	struct MaterialDeviceState
	{
		// emisson
		float3 radiance{ 0.0f, 0.0f, 0.0f };
		float opacity = 1.0f;
		uint32_t type = 0;
		float3 color{ 1.0f, 1.0f, 1.0f };
		float roughness = 0.4f;
		float anisotropic = 0.0f;
		float alphax = 0.01f;
		float alphay = 0.01f;
		float eta = 1.5f;
		float3 condEta{ 1.6574599595f, 0.8803689579f, 0.5212287346f };
		float3 condK{ 9.2238691996f, 6.2695232477f, 4.8370012281f };
		float metallic = 0.0f;
		float specular = 0.5f;
		float specularTint = 0.0f;
		float subsurface = 0.0f;
		float sheen = 0.0f;
		float sheenTint = 0.5f;
		float clearcoat = 0.0f;
		float clearcoatRoughness = 0.0f;
		float transmission = 0.0f;
	};

	struct MaterialDevice
	{
		MaterialDeviceState state;
		float ior = 1.5f;
		TextureDevice colorTexture;
		TextureDevice aoRoughMetalTexture;
		TextureDevice normalTexture;
		TextureDevice bumpTexture;
	};

	struct HitGroupData
	{
		MeshVertex* vertexBuffer = nullptr;
		uint32_t* indexBuffer = nullptr;
		MaterialDevice* material = nullptr;
	};

	struct RayTraceLaunchParams
	{
		uint32_t width = 0;
		uint32_t height = 0;
		uint32_t frameCount = 0;
		float4* colorSrgbImage = nullptr;
		float4* albedoImage = nullptr;
		float4* normalImage = nullptr;
		OptixTraversableHandle traverHandle = 0;

		float4x4 rasterToWorldMatrix;
		float4x4 cameraToWorldMatrix;
		float focalDistance = 1.0f;
		float lensRadius = 0.0f;

		uint32_t lightsNum = 0;
		LightDevice* lights = nullptr;

		// setting
		uint32_t integratorCallback = 0;
		int maxBounce = 5;
		int russianRouletteBounce = 3;
		float pathRadianceClamp = 1.0f;
	};

	struct ToneMapLaunchParams
	{
		enum Type
		{
			ACES,
		};
		Type type = ACES;
		float gamma = 2.2f;
	};
}
