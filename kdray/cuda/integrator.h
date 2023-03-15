#pragma once

#include "interaction.h"
#include "light.h"
#include "bsdf.h"

#define MIS_BSDF 0
#define MIS_LIGHT 1
#define MIS_BSDF_LIGHT 2
#define MIS_MODE MIS_BSDF_LIGHT

extern "C" __constant__ kdray::RayTraceLaunchParams Params;

__forceinline__ __device__ bool lights_intersect(bool camera_ray, const Ray& ray, float distance, float bsdf_pdf, float3* radiance)
{
    // 找出最近灯光碰撞
    float closest_distance = INFINITY;
    for (int light_index = 0; light_index < Params.lightsNum; ++light_index)
    {
        kdray::LightDevice& light = Params.lights[light_index];
        if (camera_ray && !light.visible)  continue;

        float light_pdf = 0.0f;
        float light_distance = INFINITY;
        float3 li = light_evaluate(light, ray.origin, ray.direction, distance, &light_pdf, &light_distance);
        if (light_pdf == 0.0f)  continue; // 无碰撞
#if MIS_MODE == MIS_BSDF_LIGHT
        li *= power_heuristic(bsdf_pdf, light_pdf);
#endif
		*radiance = li;
		closest_distance = light_distance;
    }
    return closest_distance != INFINITY;
}

__forceinline__ __device__ float3 uniform_sample_one_light(OptixTraversableHandle scene, 
    const Interaction& isect, Sampler& sampler)
{
    if (Params.lightsNum == 0) return make_float3(0.0f);

    using namespace kdray;
    uint32_t light_index = clamp(uint32_t(Params.lightsNum * sampler_get1d(sampler)), 0u, Params.lightsNum - 1u);
    kdray::LightDevice& light = Params.lights[light_index];

    float3 l = make_float3(0.0f);
    float3 wi = make_float3(0.0f);
    float distance = FLT_MAX;
    float light_pdf = 0.0f;
    float3 li = light_sample(light, isect.position, sampler_get2d(sampler), &wi, &distance, &light_pdf);
    if (light_pdf == 0.0f)    return make_float3(0.0f);
    Ray occlus_ray = isect.spawn_occlus_ray(wi, distance);
    if (scene_intersect_test(occlus_ray, scene))
    {
        float bsdf_pdf = 0.0f;
        float3 f = bsdf_material_evaluate(isect.material, isect, isect.wo, wi, &bsdf_pdf) *
            fabsf(dot(wi, isect.normal));
#if MIS_MODE == MIS_BSDF_LIGHT
        if (bsdf_pdf == 0.0f)    return make_float3(0.0f);
        l += power_heuristic(light_pdf, bsdf_pdf) * f * li / light_pdf;
#else
        l += f * li / light_pdf;
#endif
    }

    return l * Params.lightsNum; // l / pdf:(1.0f / Params.lightsNum)
}

extern "C" __device__ float3 __direct_callable__integrator_pt(const Ray& _ray, OptixTraversableHandle scene, Sampler& sampler)
{
    Ray ray = _ray;
	//薄镜模型产生景深
	float3 focal_point = ray.origin + ray.direction * Params.focalDistance;
	float2 local_lens_pos = uniform_sample_disk(sampler_get2d(sampler), Params.lensRadius);
    float3 lens_pos = transform_point(Params.cameraToWorldMatrix, make_float3(local_lens_pos.x, 0.0f, local_lens_pos.y));
	ray.origin = lens_pos;
	ray.direction = normalize(focal_point - lens_pos);

    float3 radiance = make_float3(0.0f);
    float3 throughput = make_float3(1.0f);
    BsdfType bsdf_type = Bsdf_None;
    float bsdf_pdf = INFINITY;
    for (int bounces = 0;; ++bounces)
    {
        bool first_bounces = (bounces == 0);
        Interaction isect;
        bool intersected = scene_intersect(ray, scene, &isect);

#if MIS_MODE != MIS_LIGHT
        // 灯源发光（Material sampling）
		float3 le = make_float3(0.0f);
        bool light_intersected = lights_intersect(first_bounces, ray, isect.distance, bsdf_pdf, &le);
		radiance += throughput * le;
        // 射线撞到灯源退出
        if (light_intersected)
        {
            break;
        }
#endif

        if (!intersected || bounces >= Params.maxBounce)
        {
            break;
        }

		// 材质发光
        radiance += throughput * isect.material.radiance;

#if MIS_MODE != MIS_BSDF
        // 选取一个灯进行直接光照计算（Light sampling）
        radiance += throughput * uniform_sample_one_light(scene, isect, sampler);
#endif
        
        // 下一次反弹方向，通过bsdf采样得到
        float3 wi;
        bsdf_pdf = 0.0f;
        float3 f = bsdf_material_sample(isect.material, isect, sampler, isect.wo, &wi, &bsdf_pdf, &bsdf_type);
        if (bsdf_pdf == 0.0f || all_zero(f))    break;
        throughput *= f * fabsf(dot(wi, isect.normal)) / bsdf_pdf;

		// 限制路径过亮，防止firefly
		if (!first_bounces && Params.pathRadianceClamp > 1.01f)
		{
			float max_radiance = max(radiance.x, max(radiance.y, radiance.z));
			if (max_radiance > Params.pathRadianceClamp)
			{
                radiance *= Params.pathRadianceClamp / max_radiance;
			}
		}

        // 路径亮度过小，直接跳过（Russian roulette）
        if (bounces >= Params.russianRouletteBounce)
        {
            float probability = max(throughput.x, max(throughput.y, throughput.z));
            if (probability < sampler_get1d(sampler))
                break;
            throughput /= probability;
        }

        ray = isect.spawn_ray(wi);
    }

    return radiance;
}

extern "C" __device__ float3 __direct_callable__integrator_ao(const Ray& ray, OptixTraversableHandle scene, Sampler& sampler)
{
    const int sampler_count = 16;

    float3 li = make_float3(0.0f);
    Interaction isect;
    if (scene_intersect(ray, scene, &isect))
    {
        float ao_value = 0.0f;
        for (int i = 0; i < sampler_count; i++)
        {
            float2 u = sampler_get2d(sampler);
            float3 wi = cosine_sample_hemisphere(u);
            float cos_theta = wi.z;
            float pdf = cosine_pdf_hemisphere(cos_theta);

            wi = isect.local_to_world(wi);
            Ray ao_ray = isect.spawn_ray(wi);
            if (scene_intersect_test(ao_ray, scene))
            {
                ao_value += dot(wi, isect.normal) / pdf;
            }
        }
        li = make_float3(ao_value / (float)sampler_count);
    }
    return li;
}

// DirectCall速度太慢
#define integrator_li_call(ray, scene, sampler) \
    optixDirectCall<float3, const Ray&, OptixTraversableHandle, Sampler&> \
    (Params.integratorCallback, ray, scene, sampler)

__forceinline__ __device__ float3 integrator_li(const Ray& ray, OptixTraversableHandle scene, Sampler& sampler)
{
	if (Params.integratorCallback == 1)
	{
		return __direct_callable__integrator_ao(ray, scene, sampler);
	}
	else
	{
		return __direct_callable__integrator_pt(ray, scene, sampler);
	}
}
