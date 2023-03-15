#pragma once

#include "fresnel.h"
#include "microfacet_distribution.h"

enum BsdfType
{
	Bsdf_None = 0,
	Bsdf_DiffuseReflection = 1 << 0,
	Bsdf_SpecularReflection = 1 << 1,
	Bsdf_SpecularTransmission = 1 << 2,
};

// =========================diffuse========================================
extern "C" __device__ float3 bsdf_evaluate_diffuse(
	const kdray::MaterialDeviceState& material, const Interaction& isect,
	const float3& world_wo, const float3& world_wi, float* pdf)
{
	float3 wi = isect.world_to_local(world_wi);
	float cos_theta = wi.z;
	*pdf = cosine_pdf_hemisphere(cos_theta);
	return material.color * M_1_PIf;
}

extern "C" __device__ float3 bsdf_sample_diffuse(
	const kdray::MaterialDeviceState& material, const Interaction& isect, Sampler& sampler,
	const float3& world_wo, float3* world_wi, float* pdf, BsdfType* type)
{
	float3 wi = cosine_sample_hemisphere(sampler_get2d(sampler));
	*world_wi = isect.local_to_world(wi);
	float cos_theta = wi.z;
	*pdf = cosine_pdf_hemisphere(cos_theta);
	*type = Bsdf_DiffuseReflection;

	return material.color * M_1_PIf;
}

// =========================metal========================================
extern "C" __device__ float3 bsdf_evaluate_metal(
	const kdray::MaterialDeviceState& material, const Interaction& isect,
	const float3& world_wo, const float3& world_wi, float* pdf)
{
	float3 wo = isect.world_to_local(world_wo);
	float3 wi = isect.world_to_local(world_wi);
	float3 wh = normalize(wo + wi);

	*pdf = microfacet_distribution::ggx_pdf(material.alphax, material.alphay, wo, wh) / (4.0f * dot(wo, wh));

	float3 fresnel = fresnel::conductor(dot(wo, wh), material.condEta, material.condK);
	float cos_i = fabsf(wi.z);
	float cos_o = fabsf(wo.z);
	return material.color * fresnel *
		microfacet_distribution::ggx_d(material.alphax, material.alphay, wh) *
		microfacet_distribution::ggx_g2(material.alphax, material.alphay, wo, wi) /
		(4.0f * cos_i * cos_o);
}

extern "C" __device__ float3 bsdf_sample_metal(
	const kdray::MaterialDeviceState& material, const Interaction& isect, Sampler& sampler,
	const float3& world_wo, float3* world_wi, float* pdf, BsdfType* type)
{
	float3 wo = isect.world_to_local(world_wo);
	float3 wh = microfacet_distribution::ggx_samplewh(material.alphax, material.alphay, sampler_get2d(sampler), wo);
	float3 wi = reflect(wo, wh);

	*world_wi = isect.local_to_world(wi);
	*pdf = microfacet_distribution::ggx_pdf(material.alphax, material.alphay, wo, wh) / (4.0f * dot(wo, wh));
	*type = Bsdf_SpecularReflection;

	float3 fresnel = fresnel::conductor(dot(wo, wh), material.condEta, material.condK);
	float cos_i = fabsf(wi.z);
	float cos_o = fabsf(wo.z);
	return material.color * fresnel *
		microfacet_distribution::ggx_d(material.alphax, material.alphay, wh) *
		microfacet_distribution::ggx_g2(material.alphax, material.alphay, wo, wi) /
		(4.0f * cos_i * cos_o);
}

// =========================plastic========================================
extern "C" __device__ float3 bsdf_evaluate_plastic(
	const kdray::MaterialDeviceState& material, const Interaction& isect,
	const float3& world_wo, const float3& world_wi, float* pdf)
{
	float3 wo = isect.world_to_local(world_wo);
	float3 wi = isect.world_to_local(world_wi);
	float cos_i = fabsf(wi.z);
	float cos_o = fabsf(wo.z);
	float3 wh = normalize(wo + wi);
	float fresnel = fresnel::dielectric(fabsf(dot(wo, wh)), material.eta);

	// 光泽反射
	float3 f_glossy = material.color * fresnel *
		microfacet_distribution::ggx_d(material.alphax, material.alphay, wh) *
		microfacet_distribution::ggx_g2(material.alphax, material.alphay, wo, wi) /
		(4.0f * cos_i * cos_o);
	float pdf_glossy = microfacet_distribution::ggx_pdf(material.alphax, material.alphay, wo, wh) /
		(4.0f * dot(wo, wh));
	pdf_glossy *= fresnel;

	// 漫反射
	float3 f_diffuse = material.color * (1.0f - fresnel) * M_1_PIf;
	float pdf_diffuse = cosine_pdf_hemisphere(cos_i);
	pdf_diffuse *= 1.0f - fresnel;

	*pdf = pdf_glossy + pdf_diffuse;
	return f_glossy + f_diffuse;
}

extern "C" __device__ float3 bsdf_sample_plastic(
	const kdray::MaterialDeviceState& material, const Interaction& isect, Sampler& sampler,
	const float3& world_wo, float3* world_wi, float* pdf, BsdfType* type)
{
	float3 wo = isect.world_to_local(world_wo);
	float fresnel = fresnel::dielectric(wo.z, material.eta);
	if (sampler_get1d(sampler) < fresnel)
	{
		// 光泽反射
		float3 wh = microfacet_distribution::ggx_samplewh(material.alphax, material.alphay, sampler_get2d(sampler), wo);
		if (wh.z < 0.0f)	wh = -wh;
		float3 wi = reflect(wo, wh);
		*world_wi = isect.local_to_world(wi);
		*pdf = microfacet_distribution::ggx_pdf(material.alphax, material.alphay, wo, wh) / (4.0f * dot(wo, wh));
		*pdf *= fresnel;
		*type = Bsdf_SpecularReflection;

		float cos_i = fabsf(wi.z);
		float cos_o = fabsf(wo.z);
		return material.color * fresnel *
			microfacet_distribution::ggx_d(material.alphax, material.alphay, wh) *
			microfacet_distribution::ggx_g2(material.alphax, material.alphay, wo, wi) /
			(4.0f * cos_i * cos_o);
	}
	else
	{
		// 漫反射
		float3 wi = cosine_sample_hemisphere(sampler_get2d(sampler));
		*world_wi = isect.local_to_world(wi);
		float cos_theta = wi.z;
		*pdf = cosine_pdf_hemisphere(cos_theta);
		*pdf *= 1.0f - fresnel;
		*type = Bsdf_DiffuseReflection;

		return material.color * (1.0f - fresnel) * M_1_PIf;
	}
}

// =========================glass========================================
extern "C" __device__ float3 bsdf_evaluate_glass(
	const kdray::MaterialDeviceState& material, const Interaction& isect,
	const float3& world_wo, const float3& world_wi, float* pdf)
{
	float3 wo = isect.world_to_local(world_wo);
	float3 wi = isect.world_to_local(world_wi);
	float cos_i = fabsf(wi.z);
	float cos_o = fabsf(wo.z);
	if (wi.z > 0.0f)
	{
		// 反射
		float3 wh = normalize(wi + wo);
		float fresnel = fresnel::dielectric(fabsf(dot(wo, wh)), material.eta);
		float ggx_pdf = microfacet_distribution::ggx_pdf(material.alphax, material.alphay, wo, wh);

		return material.color * fresnel *
			microfacet_distribution::ggx_d(material.alphax, material.alphay, wh) *
			microfacet_distribution::ggx_g2(material.alphax, material.alphay, wo, wi) /
			(4.0f * cos_i * cos_o);
		*pdf = fresnel * ggx_pdf / (4.0f * dot(wo, wh));
	}
	else
	{
		// 透射
		float3 wh = normalize(wi + wo * material.eta);
		if (wh.z < 0.0f)	wh = -wh;
		float fresnel = fresnel::dielectric(fabsf(dot(wo, wh)), material.eta);
		float ggx_pdf = microfacet_distribution::ggx_pdf(material.alphax, material.alphay, wo, wh);

		float sqrt_denom = dot(wo, wh) + material.eta * dot(wi, wh);
		return material.color * (1.0f - fresnel) *
			microfacet_distribution::ggx_d(material.alphax, material.alphay, wh) *
			microfacet_distribution::ggx_g2(material.alphax, material.alphay, wo, wi) *
			fabsf(dot(wo, wh)) * fabsf(dot(wi, wh)) * material.eta * material.eta /
			(cos_i * cos_o * sqrt_denom * sqrt_denom);
		*pdf = (1.0f - fresnel) * ggx_pdf * fabsf(dot(wi, wh)) * material.eta * material.eta / (sqrt_denom * sqrt_denom);
	}
}

extern "C" __device__ float3 bsdf_sample_glass(
	const kdray::MaterialDeviceState& material, const Interaction& isect, Sampler& sampler,
	const float3& world_wo, float3* world_wi, float* pdf, BsdfType* type)
{
	float3 wo = isect.world_to_local(world_wo);
	float3 wh = microfacet_distribution::ggx_samplewh(material.alphax, material.alphay, sampler_get2d(sampler), wo);
	if (wh.z < 0.0f)	wh = -wh;
	
	float fresnel = fresnel::dielectric(fabsf(dot(wo, wh)), material.eta);
	if (sampler_get1d(sampler) < fresnel)
	{
		// 反射
		float3 wi = reflect(wo, wh);

		*world_wi = isect.local_to_world(wi);
		*pdf = microfacet_distribution::ggx_pdf(material.alphax, material.alphay, wo, wh) /
			(4.0f * dot(wo, wh));
		*pdf *= fresnel;
		*type = Bsdf_SpecularReflection;

		float cos_i = fabsf(wi.z);
		float cos_o = fabsf(wo.z);
		return material.color * fresnel *
			microfacet_distribution::ggx_d(material.alphax, material.alphay, wh) *
			microfacet_distribution::ggx_g2(material.alphax, material.alphay, wo, wi) /
			(4.0f * cos_i * cos_o);
	}
	else
	{
		// 透射
		float3 wi = make_float3(0.0f);
		if (!refract(wo, wh, material.eta, &wi))
		{
			// 全内反射
			*pdf = 0.0f;
			*type = Bsdf_None;
			return make_float3(0.0f);
		}

		float sqrt_denom = dot(wo, wh) + material.eta * dot(wi, wh);
		*world_wi = isect.local_to_world(wi);
		*pdf = microfacet_distribution::ggx_pdf(material.alphax, material.alphay, wo, wh) *
			fabsf(dot(wi, wh)) * material.eta * material.eta / (sqrt_denom * sqrt_denom);
		*pdf *= 1.0f - fresnel;
		*type = Bsdf_SpecularTransmission;

		float cos_i = fabsf(wi.z);
		float cos_o = fabsf(wo.z);
		return material.color * (1.0f - fresnel) *
			microfacet_distribution::ggx_d(material.alphax, material.alphay, wh) *
			microfacet_distribution::ggx_g2(material.alphax, material.alphay, wo, wi) *
			fabsf(dot(wo, wh)) * fabsf(dot(wi, wh)) * material.eta * material.eta /
			(cos_i * cos_o * sqrt_denom * sqrt_denom);
	}
}
// =========================principled========================================
namespace disney
{
	__forceinline__ __device__ float3 evaluate_subsurface(const kdray::MaterialDeviceState& material, const float3& wo, const float3& wi)
	{
		float FL = fresnel::schlick_weight(fabsf(wi.z));
		float FV = fresnel::schlick_weight(wo.z);
		float Fd = (1.0f - 0.5f * FL) * (1.0f - 0.5f * FV);
		return make_float3(sqrtf(material.color.x), sqrtf(material.color.y), sqrtf(material.color.z)) *
			material.subsurface * M_1_PIf * Fd * (1.0f - material.metallic) * (1.0f - material.transmission);
	}

	__forceinline__ __device__ float3 evaluate_diffuse(const kdray::MaterialDeviceState& material, const float3& csheen,
		const float3& wo, const float3& wi, const float3& wh)
	{
		float cos_ih = dot(wi, wh);
		float FL = fresnel::schlick_weight(wi.z);
		float FV = fresnel::schlick_weight(wo.z);
		float FH = fresnel::schlick_weight(cos_ih);
		float Fd90 = 0.5f + 2.0f * cos_ih * cos_ih * material.roughness;
		float Fd = lerp(1.0f, Fd90, FL) * lerp(1.0f, Fd90, FV);
		float3 Fsheen = FH * material.sheen * csheen;
		return (M_1_PIf * Fd * (1.0f - material.subsurface) * material.color + Fsheen) * (1.0f - material.metallic);
	}
}

extern "C" __device__ float3 bsdf_evaluate_principled(
	const kdray::MaterialDeviceState& material, const Interaction& isect,
	const float3& world_wo, const float3& world_wi, float* pdf)
{
	float3 wo = isect.world_to_local(world_wo);
	float3 wi = isect.world_to_local(world_wi);
	float3 wh;
	if (wi.z < 0.0f)
		wh = normalize(wi + wo * material.eta);
	else
		wh = normalize(wi + wo);
	if (wh.z < 0.0f)	wh = -wh;
	
	float transmission_ratio = (1.0f - material.metallic) * material.transmission;
	float diffuse_ratio = 0.5f * (1.0f - material.metallic);
	float primary_specular_ratio = 1.0f / (1.0f + material.clearcoat);

	float3 bsdf = make_float3(0.0f);
	float pdf_bsdf = 0.0f;
	if (transmission_ratio > 0.0f)
	{
		float fresnel = fresnel::dielectric(fabsf(dot(wo, wh)), material.eta);
		if (wi.z > 0.0f)
		{
			bsdf = material.color * fresnel * microfacet_distribution::ggx_d(material.alphax, material.alphay, wh) *
				microfacet_distribution::ggx_g2(material.alphax, material.alphay, wo, wi) / (4.0f * fabsf(wi.z) * fabsf(wo.z));
			pdf_bsdf = microfacet_distribution::ggx_pdf(material.alphax, material.alphay, wo, wh) / (4.0f * dot(wo, wh));
		}
		else
		{
			float sqrt_denom = dot(wo, wh) + material.eta * dot(wi, wh);
			bsdf = material.color * (1.0f - fresnel) * microfacet_distribution::ggx_d(material.alphax, material.alphay, wh) *
				microfacet_distribution::ggx_g2(material.alphax, material.alphay, wo, wi) * fabsf(dot(wo, wh)) * fabsf(dot(wi, wh)) *
				material.eta * material.eta / (fabsf(wi.z) * fabsf(wo.z) * sqrt_denom * sqrt_denom);
			pdf_bsdf = microfacet_distribution::ggx_pdf(material.alphax, material.alphay, wo, wh) * fabsf(dot(wi, wh)) * material.eta * material.eta / (sqrt_denom * sqrt_denom);
		}
	}
	float3 brdf = make_float3(0.0f);
	float pdf_brdf = 0.0f;
	if (transmission_ratio < 1.0f)
	{
		if (wi.z < 0.0f)
		{
			brdf = disney::evaluate_subsurface(material, wo, wi);
			pdf_brdf = uniform_pdf_hemisphere();
			pdf_brdf *= material.subsurface * diffuse_ratio;
		}
		else
		{
			float clum = luminance(material.color);
			float3 ctint = clum > 0.0f ? material.color / clum : make_float3(1.0f);
			float3 ctintmix = material.specular * 0.08f * lerp(make_float3(1.0f), ctint, material.specularTint);
			float3 cspec = lerp(ctintmix, material.color, material.metallic);
			float3 csheen = lerp(make_float3(1.0f), ctint, material.sheenTint);

			// diffuse
			brdf += disney::evaluate_diffuse(material, csheen, wo, wi, normalize(wi + wo));
			float _pdf = cosine_pdf_hemisphere(wi.z);
			pdf_brdf += _pdf * (1.0f - material.subsurface) * diffuse_ratio;

			//  primary specular
			float FM = fresnel::disney(dot(wi, wh), dot(wo, wh), material.eta, material.metallic);
			float3 F = lerp(cspec, make_float3(1.0f), FM);
			brdf += F * microfacet_distribution::ggx_d(material.alphax, material.alphay, wh) *
				microfacet_distribution::ggx_g2(material.alphax, material.alphay, wo, wi) /
				(4.0f * fabsf(wi.z) * fabsf(wo.z));
			_pdf = microfacet_distribution::ggx_pdf(material.alphax, material.alphay, wo, wh) / (4.0f * dot(wo, wh));
			pdf_brdf += _pdf * primary_specular_ratio * (1.0f - diffuse_ratio);

			//  clearcoat
			float fresnel = fresnel::dielectric(fabsf(dot(wo, wh)), material.eta);
			float f = material.clearcoat * fresnel * microfacet_distribution::gtr1_d(material.clearcoatRoughness, wh) *
				microfacet_distribution::ggx_smith_g2(0.25f, wo, wi) / 4.0f;
			brdf += make_float3(f);
			_pdf = microfacet_distribution::gtr1_pdf(material.clearcoatRoughness, wh) / (4.0f * dot(wo, wh));
			pdf_brdf += _pdf * (1.0f - primary_specular_ratio) * (1.0f - diffuse_ratio);
		}
	}
	*pdf = lerp(pdf_brdf, pdf_bsdf, transmission_ratio);
	return lerp(brdf, bsdf, transmission_ratio);
}

extern "C" __device__ float3 bsdf_sample_principled(
	const kdray::MaterialDeviceState& material, const Interaction& isect, Sampler& sampler,
	const float3& world_wo, float3* world_wi, float* pdf, BsdfType* type)
{
	float3 f = make_float3(0.0f);
	*pdf = 0.0f;

	float3 wo = isect.world_to_local(world_wo);
	float diffuse_ratio = 0.5f * (1.0f - material.metallic);
	float transmission_ratio = (1.0f - material.metallic) * material.transmission;
	if (sampler_get1d(sampler) < transmission_ratio) // 玻璃比例
	{
		float3 wh = microfacet_distribution::ggx_samplewh(material.alphax, material.alphay, sampler_get2d(sampler), wo);
		float fresnel = fresnel::dielectric(fabsf(dot(wo, wh)), material.eta);
		if (sampler_get1d(sampler) < fresnel) // 反射
		{
			float3 wi = reflect(wo, wh);
			*world_wi = isect.local_to_world(wi);
			*pdf = microfacet_distribution::ggx_pdf(material.alphax, material.alphay, wo, wh) / (4.0f * dot(wo, wh));
			*pdf *= fresnel;
			*type = Bsdf_SpecularReflection;
			f = material.color * fresnel * microfacet_distribution::ggx_d(material.alphax, material.alphay, wh) *
				microfacet_distribution::ggx_g2(material.alphax, material.alphay, wo, wi) / (4.0f * fabsf(wi.z) * fabsf(wo.z));
		}
		else // 透射
		{
			float3 wi = make_float3(0.0f);
			if (!refract(wo, wh, material.eta, &wi))
			{
				*pdf = 0.0f;
				*type = Bsdf_None;
				return make_float3(0.0f);
			}
			float sqrt_denom = dot(wo, wh) + material.eta * dot(wi, wh);
			*world_wi = isect.local_to_world(wi);
			*pdf = microfacet_distribution::ggx_pdf(material.alphax, material.alphay, wo, wh) * fabsf(dot(wi, wh)) * 
				material.eta * material.eta / (sqrt_denom * sqrt_denom);
			*pdf *= 1.0f - fresnel;
			*type = Bsdf_SpecularTransmission;
			f = material.color * (1.0f - fresnel) * microfacet_distribution::ggx_d(material.alphax, material.alphay, wh) *
				microfacet_distribution::ggx_g2(material.alphax, material.alphay, wo, wi) * fabsf(dot(wo, wh)) * fabsf(dot(wi, wh)) * 
				material.eta * material.eta / (fabsf(wi.z) * fabsf(wo.z) * sqrt_denom * sqrt_denom);
		}
		f *= transmission_ratio;
		*pdf *= transmission_ratio;
	}
	else // BRDF
	{
		float clum = luminance(material.color);
		float3 ctint = clum > 0.0f ? material.color / clum : make_float3(1.0f);
		float3 ctintmix = material.specular * 0.08f * lerp(make_float3(1.0f), ctint, material.specularTint);
		float3 cspec = lerp(ctintmix, material.color, material.metallic);
		float3 csheen = lerp(make_float3(1.0f), ctint, material.sheenTint);
		if (sampler_get1d(sampler) < diffuse_ratio)  // 漫反射
		{
			*type = Bsdf_DiffuseReflection;
			if (sampler_get1d(sampler) < material.subsurface) // 次表面散射
			{
				float3 wi = uniform_sample_hemisphere(sampler_get2d(sampler));
				wi.z = -wi.z;
				*world_wi = isect.local_to_world(wi);
				*pdf = uniform_pdf_hemisphere();
				*pdf *= material.subsurface * diffuse_ratio;
				f = disney::evaluate_subsurface(material, wo, wi);
			}
			else 
			{
				float3 wi = cosine_sample_hemisphere(sampler_get2d(sampler));
				*world_wi = isect.local_to_world(wi);
				*pdf = cosine_pdf_hemisphere(wi.z);
				*pdf *= (1.0f - material.subsurface) * diffuse_ratio;
				f = disney::evaluate_diffuse(material, csheen, wo, wi, normalize(wi + wo));
			}
		}
		else
		{
			*type = Bsdf_SpecularReflection;
			float primary_specular_ratio = 1.0f / (1.0f + material.clearcoat);
			if (sampler_get1d(sampler) < primary_specular_ratio) // 主瓣波
			{
				float3 wh = microfacet_distribution::ggx_samplewh(material.alphax, material.alphay, sampler_get2d(sampler), wo);
				float3 wi = reflect(wo, wh);
				*world_wi = isect.local_to_world(wi);
				*pdf = microfacet_distribution::ggx_pdf(material.alphax, material.alphay, wo, wh) / (4.0f * dot(wo, wh));
				*pdf *= primary_specular_ratio * (1.0f - diffuse_ratio);
				float FM = fresnel::disney(dot(wi, wh), dot(wo, wh), material.eta, material.metallic);
				float3 F = lerp(cspec, make_float3(1.0f), FM);
				f = F * microfacet_distribution::ggx_d(material.alphax, material.alphay, wh) *
					microfacet_distribution::ggx_g2(material.alphax, material.alphay, wo, wi) /
					(4.0f * fabsf(wi.z) * fabsf(wo.z));
			}
			else // 清漆层
			{
				float3 wh = microfacet_distribution::gtr1_samplewh(material.clearcoatRoughness, sampler_get2d(sampler));
				float3 wi = reflect(wo, wh);
				*world_wi = isect.local_to_world(wi);
				*pdf = microfacet_distribution::gtr1_pdf(material.clearcoatRoughness, wh) / (4.0f * dot(wo, wh));
				*pdf *= (1.0f - primary_specular_ratio) * (1.0f - diffuse_ratio);
				float fresnel = fresnel::dielectric(fabsf(dot(wo, wh)), material.eta);
				float _f = material.clearcoat * fresnel * microfacet_distribution::gtr1_d(material.clearcoatRoughness, wh) *
					microfacet_distribution::ggx_smith_g2(0.25f, wo, wi) / 4.0f;
				f = make_float3(_f);
			}
		}
		f *= (1.0f - transmission_ratio);
		*pdf *= (1.0f - transmission_ratio);
	}
	return f;
}

// =========================api========================================
__forceinline__ __device__ float3 bsdf_material_evaluate(
	const kdray::MaterialDeviceState& material, const Interaction& isect,
	const float3& wo, const float3& wi, float* pdf)
{
	if (material.type == 1)
	{
		return bsdf_evaluate_metal(material, isect, wo, wi, pdf);
	}
	else if (material.type == 2)
	{
		return bsdf_evaluate_plastic(material, isect, wo, wi, pdf);
	}
	else if (material.type == 3)
	{
		return bsdf_evaluate_glass(material, isect, wo, wi, pdf);
	}
	else if (material.type == 4)
	{
		return bsdf_evaluate_principled(material, isect, wo, wi, pdf);
	}
	else
	{
		return bsdf_evaluate_diffuse(material, isect, wo, wi, pdf);
	}
}

__forceinline__ __device__ float3 bsdf_material_sample(
	const kdray::MaterialDeviceState& material, const Interaction& isect, Sampler& sampler,
	const float3& wo, float3* wi, float* pdf, BsdfType* type)
{
	if (material.type == 1)
	{
		return bsdf_sample_metal(material, isect, sampler, wo, wi, pdf, type);
	}
	else if (material.type == 2)
	{
		return bsdf_sample_plastic(material, isect, sampler, wo, wi, pdf, type);
	}
	else if (material.type == 3)
	{
		return bsdf_sample_glass(material, isect, sampler, wo, wi, pdf, type);
	}
	else if (material.type == 4)
	{
		return bsdf_sample_principled(material, isect, sampler, wo, wi, pdf, type);
	}
	else
	{
		return bsdf_sample_diffuse(material, isect, sampler, wo, wi, pdf, type);
	}
}
