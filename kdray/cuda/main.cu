#include "shared_types.h"
#include "sampler.h"
#include "filter.h"
#include "monte_carlo.h"
#include "interaction.h"
#include "integrator.h"

extern "C" __constant__ kdray::RayTraceLaunchParams Params;

__forceinline__ __device__ Ray generate_ray(uint3 idx, uint3 dim, const float2& jitter)
{
    float2 screen_pos = 2.0f * make_float2(
        (static_cast<float>(idx.x) + jitter.x) / static_cast<float>(dim.x),
        (static_cast<float>(idx.y) + jitter.y) / static_cast<float>(dim.y)) - make_float2(1.0f, 1.0f);
    
    float3 world_pos = transform_point(Params.rasterToWorldMatrix, make_float3(screen_pos, 0.0f));

    Ray ray;
    ray.origin = Params.cameraToWorldMatrix.origin();
    ray.direction = normalize(world_pos - ray.origin);
    return ray;
}

extern "C" __global__ void __raygen__main()
{
    const uint3 launch_idx = optixGetLaunchIndex();
    const uint3 launch_dim = optixGetLaunchDimensions();
    unsigned int image_index = launch_idx.x + launch_idx.y * launch_dim.x;

    //Gbuffer output
    if (Params.integratorCallback == 2)
    {
        Ray ray = generate_ray(launch_idx, launch_dim, make_float2(0.5f, 0.5f));
        Interaction isect;
        if (scene_intersect(ray, Params.traverHandle, &isect))
        {
            Params.albedoImage[image_index] = make_float4(isect.material.color, 1.0f);
            Params.normalImage[image_index] = make_float4(isect.normal, 1.0f);
        }
        else
        {
            Params.albedoImage[image_index] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
            Params.normalImage[image_index] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
        return;
    }

    Sampler sampler;
    sampler_init(sampler, image_index, Params.frameCount);

    float2 subpixel_jitter = filter_jitter(sampler_get2d(sampler));
    Ray ray = generate_ray(launch_idx, launch_dim, subpixel_jitter);
    float3 accum_current = integrator_li(ray, Params.traverHandle, sampler);
    if (isnan(accum_current))   return;
    if (Params.frameCount > 0)
    {
        float3 accum_prev = make_float3(Params.colorSrgbImage[image_index]);
        float t = 1.0f / (float)(Params.frameCount + 1);
        accum_current = lerp(accum_prev, accum_current, t);
    }
    Params.colorSrgbImage[image_index] = make_float4(accum_current.x, accum_current.y, accum_current.z, 1.0f);
}

extern "C" __global__ void __miss__main()
{
    uint32_t isectPtr0 = optixGetPayload_0();
    uint32_t isectPtr1 = optixGetPayload_1();
    Interaction* interaction = reinterpret_cast<Interaction*>(unpack_pointer(isectPtr0, isectPtr1));
    interaction->wo = -optixGetWorldRayDirection();
    interaction->distance = FLT_MAX;
    interaction->position = optixGetWorldRayOrigin();
}

extern "C" __global__ void __miss__occlusion()
{
    optixSetPayload_0(false);
}

extern "C" __global__ void __closesthit__main()
{
    using namespace kdray;

    uint32_t isectPtr0 = optixGetPayload_0();
    uint32_t isectPtr1 = optixGetPayload_1();
    Interaction* interaction = reinterpret_cast<Interaction*>(unpack_pointer(isectPtr0, isectPtr1));
    interaction->wo = -optixGetWorldRayDirection();
    interaction->distance = optixGetRayTmax();

    const HitGroupData& hitGroupData = *reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const uint32_t triangleIndex = optixGetPrimitiveIndex() * 3;
    const MeshVertex& v0 = hitGroupData.vertexBuffer[hitGroupData.indexBuffer[triangleIndex + 0]];
    const MeshVertex& v1 = hitGroupData.vertexBuffer[hitGroupData.indexBuffer[triangleIndex + 1]];
    const MeshVertex& v2 = hitGroupData.vertexBuffer[hitGroupData.indexBuffer[triangleIndex + 2]];
    
    const float2 bary = optixGetTriangleBarycentrics();
    float2 texCoord = v0.texCoord * (1.0f - bary.x - bary.y) + v1.texCoord * bary.x + v2.texCoord * bary.y;

    interaction->position = v0.position * (1.0f - bary.x - bary.y) + v1.position * bary.x + v2.position * bary.y;
    interaction->position = optixTransformPointFromObjectToWorldSpace(interaction->position);
    interaction->geomNormal = cross(v1.position - v0.position, v2.position - v0.position);
    interaction->geomNormal = optixTransformNormalFromObjectToWorldSpace(interaction->geomNormal);
    interaction->geomNormal = normalize(interaction->geomNormal);

    interaction->normal = v0.normal * (1.0f - bary.x - bary.y) + v1.normal * bary.x + v2.normal * bary.y;
    interaction->normal = optixTransformNormalFromObjectToWorldSpace(interaction->normal);
    interaction->normal = normalize(interaction->normal);
    interaction->normal = faceforward(interaction->normal, interaction->wo);
    if (hitGroupData.material->normalTexture.texture > 0)
    {
        float3 normal = sample_texture(hitGroupData.material->normalTexture, texCoord);
        normal = normalize(normal * 2.0f - make_float3(1.0f));
        normal = tangent_to_world(normal, interaction->normal);
        interaction->normal = faceforward(normal, interaction->wo);
    }
    else if (hitGroupData.material->bumpTexture.texture > 0)
    {
        float n = sample_texture(hitGroupData.material->bumpTexture, texCoord).x;
        float u = sample_texture(hitGroupData.material->bumpTexture, make_float2(texCoord.x + 1.0f, texCoord.y)).x;
        float v = sample_texture(hitGroupData.material->bumpTexture, make_float2(texCoord.x, texCoord.y + 1.0f)).x;
        float3 normal = normalize(make_float3(u - n, v - n, 1.0f));
        normal = tangent_to_world(normal, interaction->normal);
        interaction->normal = faceforward(normal, interaction->wo);
    }
    gen_tangent_basis(interaction->normal, &interaction->tangent, &interaction->binormal);

    interaction->material = hitGroupData.material->state;
    if (hitGroupData.material->colorTexture.texture > 0)
    {
        interaction->material.color = sample_texture(hitGroupData.material->colorTexture, texCoord);
    }
    if (hitGroupData.material->aoRoughMetalTexture.texture > 0)
    {
        float3 aoRoughMetal = sample_texture(hitGroupData.material->aoRoughMetalTexture, texCoord);
        interaction->material.roughness = aoRoughMetal.y;
        interaction->material.metallic = aoRoughMetal.z;
    }
    interaction->material.roughness = sqr(interaction->material.roughness);
    float aspect = sqrtf(1.0f - interaction->material.anisotropic * 0.9f);
    interaction->material.alphax = max(0.001f, interaction->material.roughness / aspect);
    interaction->material.alphay = max(0.001f, interaction->material.roughness * aspect);
    interaction->material.eta = dot(interaction->wo, interaction->geomNormal) > 0.0f ? 1.0f / hitGroupData.material->ior : hitGroupData.material->ior;
    interaction->material.clearcoatRoughness = sqr(interaction->material.clearcoatRoughness);
}
