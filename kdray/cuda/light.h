#pragma once

#include "distribution.h"

// ===================================directional=================================
extern "C" __device__ float3 light_evaluate_directional(
	const kdray::LightDevice& light, const float3 & ray_origin, const float3& ray_dir, 
	float distance, float* pdf, float* light_distance)
{
	float3 normal = normalize(light.localToWorldMatrix.basisz());
	*light_distance = FLT_MAX;
	if (*light_distance <= distance && dot(normal, ray_dir) > light.cosAngle)
	{
		*pdf = uniform_pdf_cone(light.cosAngle);
		return light.radiance;
	}

	*pdf = 0.0f;
	return make_float3(0.0f);
}

extern "C" __device__ float3 light_sample_directional(
	const kdray::LightDevice& light, const float3& hitpos, const float2& r2,
	float3* wi, float* distance, float* pdf)
{
	*wi = uniform_sample_cone(r2, light.cosAngle);
	*wi = transform_direction(light.localToWorldMatrix, *wi);
	*pdf = uniform_pdf_cone(light.cosAngle);
	*distance = FLT_MAX;
	return light.radiance;
}

// ===================================envmap=================================
extern "C" __device__ float3 light_evaluate_envmap(
	const kdray::LightDevice& light, const float3& ray_origin, const float3& ray_dir, 
	float distance, float* pdf, float* light_distance)
{
	*light_distance = FLT_MAX;
	if (*light_distance <= distance)
	{
		float phi = atan2f(ray_dir.y, ray_dir.x);
		if (phi < 0.0f)	phi += M_2PIf;
		float theta = acosf(clamp(-ray_dir.z, -1.0f, 1.0f));
		float u = phi / M_2PIf;
		float v = theta / M_PIf;
		float3 color = make_float3(tex2D<float4>(light.texture, u, v));
		*pdf = distribution2d_pdf(light.distributionMarginal, light.distributionConditional,
			light.textureWidth, light.textureHeight, u, v);
		*pdf /= M_2PIf * M_PIf * sin(theta);
		return color;
	}

	*pdf = 0.0f;
	return make_float3(0.0f);
}

extern "C" __device__ float3 light_sample_envmap(
	const kdray::LightDevice& light, const float3& hitpos, const float2& r2,
	float3* wi, float* distance, float* pdf)
{
	int2 xy = distribution2d_sample(light.distributionMarginal, light.distributionConditional, 
		light.textureWidth, light.textureHeight, r2, pdf);
	float u = (float)xy.x / (float)light.textureWidth;
	float v = (float)xy.y / (float)light.textureHeight;
	float phi = u * M_2PIf;
	float theta = v * M_PIf;
	float sin_theta = sin(theta);
	*wi = make_float3(sin_theta * cos(phi), sin_theta * sin(phi), -cos(theta));
	*pdf /= M_2PIf * M_PIf * sin_theta;
	if (sin_theta == 0.0f)	*pdf = 0.0f;

	*distance = FLT_MAX;
	return make_float3(tex2D<float4>(light.texture, u, v));
}

// ===================================point=================================
extern "C" __device__ float3 light_evaluate_point(
	const kdray::LightDevice& light, const float3& ray_origin, const float3& ray_dir, 
	float distance, float* pdf, float* light_distance)
{
	const float3& light_pos = light.localToWorldMatrix.origin();
	*light_distance = intersect_sphere(ray_origin, ray_dir, light_pos, light.radius);
	if (*light_distance < distance)
	{
		float3 dir = light_pos - ray_origin;
		float dist_sq = dot(dir, dir);
		float sin_theta_sq = light.radius * light.radius / dist_sq;
		float cos_theta = sqrt(1.0f - sin_theta_sq);
		*pdf = uniform_pdf_cone(cos_theta);
		return light.radiance;
	}

	*pdf = 0.0f;
	return make_float3(0.0f);
}

extern "C" __device__ float3 light_sample_point(
	const kdray::LightDevice& light, const float3& hitpos, const float2& r2,
	float3* wi, float* distance, float* pdf)
{
	// 相比均匀采样球体，球中心和相交点连线构成的圆锥更快
	float3 dir = light.localToWorldMatrix.origin() - hitpos;
	float dist_sq = dot(dir, dir);
	float inv_dist = 1.0f / sqrt(dist_sq);
	dir *= inv_dist;
	*distance = dist_sq * inv_dist;

	float sin_theta = light.radius * inv_dist;
	if (sin_theta < 1.0f)
	{
		float cos_theta = sqrt(1.0f - sin_theta * sin_theta);
		*wi = uniform_sample_cone(r2, cos_theta);
		float cos_i = wi->z;
		*wi = tangent_to_world(*wi, dir);
		*pdf = uniform_pdf_cone(cos_theta);
		*distance = cos_i * (*distance) - sqrt(max(0.0f,
			light.radius * light.radius - (1.0f - cos_i * cos_i) * dist_sq));
		return light.radiance;
	}

	*pdf = 0.0f;
	return make_float3(0.0f);
}

// ===================================rect=================================
extern "C" __device__ float3 light_evaluate_rect(
	const kdray::LightDevice& light, const float3& ray_origin, const float3& ray_dir, 
	float distance, float* pdf, float* light_distance)
{
	float3 normal = normalize(light.localToWorldMatrix.basisz());
	float cos_theta = dot(ray_dir, normal);
	if (!light.doubleSided && cos_theta < 0.0f)  //backfacing?
	{
		*pdf = 0.0f;
		return make_float3(0.0f);
	}

	*light_distance = intersect_rect(ray_origin, ray_dir, light.corner, light.u, light.v);
	if (*light_distance < distance)
	{
		*pdf = light.invArea; //surface pdf
		//solid angel pdf
		*pdf *= sqr(*light_distance) / fabsf(cos_theta);
		return light.radiance;
	}

	*pdf = 0.0f;
	return make_float3(0.0f);
}

extern "C" __device__ float3 light_sample_rect(
	const kdray::LightDevice& light, const float3& hitpos, const float2& r2,
	float3* wi, float* distance, float* pdf)
{
	float3 normal = normalize(light.localToWorldMatrix.basisz());
	float3 pos = light.corner + light.u * r2.x + light.v * r2.y;
	*wi = pos - hitpos;
	float dist_sq = dot(*wi, *wi);
	*distance = sqrtf(dist_sq);
	*wi /= *distance;

	float cos_theta = dot(*wi, normal);
	if (!light.doubleSided && cos_theta < 0.0f)
	{
		*pdf = 0.0f;
		return make_float3(0.0f);
	}

	*pdf = light.invArea;
	*pdf *= dist_sq / fabsf(cos_theta);
	return light.radiance;
}

// ===================================disk=================================
extern "C" __device__ float3 light_evaluate_disk(
	const kdray::LightDevice& light, const float3& ray_origin, const float3& ray_dir, 
	float distance, float* pdf, float* light_distance)
{
	float3 center = light.localToWorldMatrix.origin();
	float3 normal = normalize(light.localToWorldMatrix.basisz());

	float cos_theta = dot(ray_dir, normal);
	if (!light.doubleSided && cos_theta < 0.0f)  //backfacing?
	{
		*pdf = 0.0f;
		return make_float3(0.0f);
	}

	*light_distance = intersect_disk(ray_origin, ray_dir, center, normal, light.radius, light.doubleSided);
	if (*light_distance < distance)
	{
		*pdf = light.invArea;
		*pdf *= sqr(*light_distance) / fabsf(cos_theta);
		return light.radiance;
	}

	*pdf = 0.0f;
	return make_float3(0.0f);
}

extern "C" __device__ float3 light_sample_disk(
	const kdray::LightDevice& light, const float3& hitpos, const float2& r2,
	float3* wi, float* distance, float* pdf)
{
	float3 normal = normalize(light.localToWorldMatrix.basisz());
	float2 local_pos = uniform_sample_disk(r2, light.radius);
	float3 pos = transform_point(light.localToWorldMatrix, make_float3(local_pos, 0.0f));
	*wi = pos - hitpos;
	float dist_sq = dot(*wi, *wi);
	*distance = sqrtf(dist_sq);
	*wi /= *distance;

	float cos_theta = dot(*wi, normal);
	if (!light.doubleSided && cos_theta < 0.0f)
	{
		*pdf = 0.0f;
		return make_float3(0.0f);
	}

	*pdf = light.invArea;
	*pdf *= dist_sq / fabsf(cos_theta);
	return light.radiance;
}

// ===================================spot=================================
extern "C" __device__ float3 light_evaluate_spot(
	const kdray::LightDevice& light, const float3& ray_origin, const float3& ray_dir, 
	float distance, float* pdf, float* light_distance)
{
	float3 radiance = light_evaluate_point(light, ray_origin, ray_dir, distance, pdf, light_distance);
	if (*pdf != 0.0f)
	{
		float3 normal = normalize(light.localToWorldMatrix.basisz());
		float cos_angle = fabsf(dot(normal, ray_dir));
		float attenuation = clamp((cos_angle - light.cosAngle) * light.cosAngleScale, 0.0f, 1.0f);
		return radiance * attenuation;
	}
	return radiance;
}

extern "C" __device__ float3 light_sample_spot(
	const kdray::LightDevice& light, const float3& hitpos, const float2& r2,
	float3* wi, float* distance, float* pdf)
{
	float3 radiance = light_sample_point(light, hitpos, r2, wi, distance, pdf);
	if (*pdf != 0.0f)
	{
		float3 normal = normalize(light.localToWorldMatrix.basisz());
		float cos_angle = fabsf(dot(normal, *wi));
		float attenuation = clamp((cos_angle - light.cosAngle) * light.cosAngleScale, 0.0f, 1.0f);
		return radiance * attenuation;
	}
	return radiance;
}

// =========================api========================================
__forceinline__ __device__ float3 light_evaluate(
	const kdray::LightDevice& light, const float3& ray_origin, const float3& ray_dir, 
	float distance, float* pdf, float* light_distance)
{
	if (light.type == 1)
	{
		return light_evaluate_directional(light, ray_origin, ray_dir, distance, pdf, light_distance);
	}
	else if (light.type == 2)
	{
		return light_evaluate_envmap(light, ray_origin, ray_dir, distance, pdf, light_distance);
	}
	else if (light.type == 3)
	{
		return light_evaluate_rect(light, ray_origin, ray_dir, distance, pdf, light_distance);
	}
	else if (light.type == 4)
	{
		return light_evaluate_point(light, ray_origin, ray_dir, distance, pdf, light_distance);
	}
	else if (light.type == 5)
	{
		return light_evaluate_disk(light, ray_origin, ray_dir, distance, pdf, light_distance);
	}
	else if (light.type == 6)
	{
		return light_evaluate_spot(light, ray_origin, ray_dir, distance, pdf, light_distance);
	}
	return make_float3(0.0f);
}

__forceinline__ __device__ float3 light_sample(
	const kdray::LightDevice& light, const float3& hitpos, const float2& r2,
	float3* wi, float* distance, float* pdf)
{
	if (light.type == 1)
	{
		return light_sample_directional(light, hitpos, r2, wi, distance, pdf);
	}
	else if (light.type == 2)
	{
		return light_sample_envmap(light, hitpos, r2, wi, distance, pdf);
	}
	else if (light.type == 3)
	{
		return light_sample_rect(light, hitpos, r2, wi, distance, pdf);
	}
	else if (light.type == 4)
	{
		return light_sample_point(light, hitpos, r2, wi, distance, pdf);
	}
	else if (light.type == 5)
	{
		return light_sample_disk(light, hitpos, r2, wi, distance, pdf);
	}
	else if (light.type == 6)
	{
		return light_sample_spot(light, hitpos, r2, wi, distance, pdf);
	}
	*pdf = 0.0f;
	return make_float3(0.0f);
}
