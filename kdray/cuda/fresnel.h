#pragma once

namespace fresnel
{
	__forceinline__ __device__ float3 conductor(float cos_i, const float3& eta, const float3& k)
	{
		float cos_i_sq = cos_i * cos_i;
		float sin_i_sq = 1.0f - cos_i * cos_i;
		float3 eta2 = eta * eta;
		float3 etak2 = k * k;

		float3 t0 = eta2 - etak2 - make_float3(sin_i_sq);
		float3 a2plusb2 = sqrt(t0 * t0 + 4.0f * eta2 * etak2);
		float3 t1 = a2plusb2 + make_float3(cos_i_sq);
		float3 a = sqrt((a2plusb2 + t0) / 2.0f);
		float3 t2 = 2.0f * a * cos_i;
		float3 rs = (t1 - t2) / (t1 + t2);

		float3 t3 = cos_i_sq * a2plusb2 + make_float3(sin_i_sq * sin_i_sq);
		float3 t4 = t2 * sin_i_sq;
		float3 rp = rs * (t3 - t4) / (t3 + t4);

		return 0.5f * (rp + rs);
	}

	__forceinline__ __device__ float dielectric(float cos_i, float eta)
	{
		float sin_i_sq = 1.0f - cos_i * cos_i;
		float cos_t_sq = 1.0f - sin_i_sq * (eta * eta);

		// total internal reflection
		if (cos_t_sq < 0.0f)	return 1.0f;
		float cos_t = sqrt(cos_t_sq);

		float rp = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
		float rs = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
		return 0.5f * (rs * rs + rp * rp);
	}

	__forceinline__ __device__ float schlick_weight(float cos_theta)
	{
		float m = clamp(1.0f - cos_theta, 0.0f, 1.0f);
		return m * m * m * m * m;
	}

	__forceinline__ __device__ float schlick(float r0, float cos_theta)
	{
		return lerp(r0, 1.0f, schlick_weight(cos_theta));
	}

	__forceinline__ __device__ float disney(float cos_ih, float cos_oh, float eta, float metallic)
	{
		float metallic_fresnel = schlick_weight(cos_ih);
		float dielectric_fresnel = dielectric(cos_oh, eta);
		return lerp(dielectric_fresnel, metallic_fresnel, metallic);
	}
}
