#pragma once

__forceinline__ __device__ float power_heuristic(float pdf1, float pdf2)
{
	if (isinf(pdf1))	return 1.0f;
	
	float temp = pdf1 * pdf1;
	return temp / (pdf2 * pdf2 + temp);
}

__forceinline__ __device__ float2 uniform_sample_disk(const float2& r2, float radius)
{
	float2 p = make_float2(0.0f, 0.0f);

	float theta = M_2PIf * r2.x;
	radius *= sqrtf(r2.y);

	p.x = radius * cosf(theta);
	p.y = radius * sinf(theta);
	return p;
}

__forceinline__ __device__ float3 uniform_sample_cone(const float2& r2, float cos_angle)
{
	float3 p = make_float3(0.0f);

	float phi = M_2PIf * r2.x;
	float cos_theta = lerp(cos_angle, 1.0f, r2.y);
	float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

	p.x = sin_theta * cosf(phi);
	p.y = sin_theta * sinf(phi);
	p.z = cos_theta;
	return p;
}

__forceinline__ __device__ float uniform_pdf_cone(float cos_angle)
{
	return 1.0f / (M_2PIf * (1.0f - cos_angle));
}

__forceinline__ __device__ float3 uniform_sample_sphere(const float2& r2)
{
	float3 p = make_float3(0.0f);

	float phi = M_2PIf * r2.x;
	float cos_theta = 1.0f - 2.0f * r2.y;
	float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

	p.x = sin_theta * cosf(phi);
	p.y = sin_theta * sinf(phi);
	p.z = cos_theta;
	return p;
}

__forceinline__ __device__ float uniform_pdf_sphere()
{
	return 1.0f / (4.0f * M_PIf);
}

__forceinline__ __device__ float3 uniform_sample_hemisphere(const float2& r2)
{
	float r = sqrtf(max(0.0f, 1.0f - r2.x * r2.x));
	float phi = M_2PIf * r2.y;
	return make_float3(r * cos(phi), r * sin(phi), r2.x);
}

__forceinline__ __device__ float uniform_pdf_hemisphere()
{
	return 1.0f / M_2PIf;
}

__forceinline__ __device__ float3 cosine_sample_hemisphere(const float2& r2)
{
	float3 p = make_float3(0.0f);

	// uniformly sample disk
	float r = sqrtf(r2.x);
	float phi = M_2PIf * r2.y;
	p.x = r * cosf(phi);
	p.y = r * sinf(phi);

	// project up to hemisphere
	p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y)); //cosTheta NoL
	return p;
}

__forceinline__ __device__ float cosine_pdf_hemisphere(float cos_theta)
{
	return cos_theta * M_1_PIf;
}
