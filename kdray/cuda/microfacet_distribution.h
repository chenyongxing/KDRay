#pragma once

// 立体角w都是单位向量在切线空间，w.z就代表cos_theta
namespace microfacet_distribution
{
	__forceinline__ __device__ float ggx_lambda(
		float alphax, float alphay, const float3& w)
	{
		float cos_theta_sq = w.z * w.z;
		float inv_a2 = (sqr(w.x * alphax) + sqr(w.y * alphay)) / cos_theta_sq;
		return 0.5f * (-1.0f + sqrt(1.0f + inv_a2));
	}

	__forceinline__ __device__ float ggx_g1(
		float alphax, float alphay, const float3& w)
	{
		return 1.0f / (1.0f + ggx_lambda(alphax, alphay, w));
	}

	__forceinline__ __device__ float ggx_g2(
		float alphax, float alphay, const float3& wo, const float3& wi)
	{
		return 1.0f / (1.0f + ggx_lambda(alphax, alphay, wo) +
			ggx_lambda(alphax, alphay, wi));
	}

	__forceinline__ __device__ float ggx_d(
		float alphax, float alphay, const float3& wh)
	{
		float cos_theta_sq = wh.z * wh.z;

		// 粗糙度过小容易出现亮斑
		float e = (sqr(wh.x / alphax) + sqr(wh.y / alphay)) / cos_theta_sq;
		return 1.0f / (M_PIf * alphax * alphay * sqr(cos_theta_sq * (1.0f + e)));
	}

	__forceinline__ __device__ float ggx_pdf(
		float alphax, float alphay, const float3& wo, const float3& wh)
	{
		float cos_theta_o = wo.z;
		return ggx_d(alphax, alphay, wh) * ggx_g1(alphax, alphay, wo) * fabsf(dot(wo, wh)) / fabsf(cos_theta_o);
	}

	__forceinline__ __device__ float3 ggx_samplewh(
		float alphax, float alphay, const float2& r2, const float3& wo)
	{
		// stretch wo
		float3 v = normalize(make_float3(alphax * wo.x, alphay * wo.y, wo.z));

		// orthonormal basis
		float3 t1 = (v.z < 0.9999f) ? normalize(cross(v, make_float3(0.0f, 0.0f, 1.0f))) : make_float3(1.0f, 0.0f, 0.0f);
		float3 t2 = cross(t1, v);

		//sample point with polar coordinates (r, phi)
		float a = 1.0f / (1.0f + v.z);
		float r = sqrt(r2.x);
		float phi = (r2.y < a) ? r2.y / a * M_PIf : M_PIf + (r2.y - a) / (1.0f - a) * M_PIf;
		float p1 = r * cos(phi);
		float p2 = r * sin(phi) * ((r2.y < a) ? 1.0f : v.z);

		//compute normal
		float3 wh = p1 * t1 + p2 * t2 + sqrt(max(0.0f, 1.0f - p1 * p1 - p2 * p2)) * v;

		// unstretch
		return normalize(make_float3(alphax * wh.x, alphay * wh.y, max(0.0f, wh.z)));
	}
	
	__forceinline__ __device__ float ggx_smith_g1(float alpha, const float3& w)
	{
		float cos_theta = fabsf(w.z);
		float cos_theta_sq = cos_theta * cos_theta;
		return 1.0f / (cos_theta + sqrt(alpha + cos_theta_sq - alpha * cos_theta_sq));
	}

	__forceinline__ __device__ float ggx_smith_g2(float alpha, const float3& wo, const float3& wi)
	{
		return ggx_smith_g1(alpha, wo) * ggx_smith_g1(alpha, wi);
	}

	__forceinline__ __device__ float gtr1_d(float alpha, const float3& wh)
	{
		float a2 = alpha * alpha;
		float cos_theta_sq = wh.z * wh.z;
		float t = 1.0f + (a2 - 1.0f) * cos_theta_sq;
		return (a2 - 1.0f) / (M_PIf * logf(a2) * t);
	}

	__forceinline__ __device__ float gtr1_pdf(float alpha, const float3& wh)
	{
		return gtr1_d(alpha, wh) * wh.z;
	}

	__forceinline__ __device__ float3 gtr1_samplewh(float alpha, const float2& r2)
	{
		float a2 = alpha * alpha;
		float cos_theta = sqrtf(max(0.0f, (1.0f - powf(a2, 1.0f - r2.x)) / (1.0f - a2)));
		float sin_theta = sqrtf(max(0.0f, 1.0f - cos_theta * cos_theta));
		float phi = r2.y * M_2PIf;
		// spherical direction
		return make_float3(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);
	}
}
