#pragma once

#define FILTER_BOX 0
#define FILTER_TENT 1
#define FILTER_TRIANGLE 2
#define FILTER_GAUSSIAN 3
#define FILTER_MODE FILTER_TENT

__forceinline__ __device__ float2 filter_box(const float2& r2)
{
	return r2 - make_float2(0.5f, 0.5f);
}

__forceinline__ __device__ float2 filter_tent(const float2& r2)
{
	float2 j = r2;
	j = j * 2.0f;
	j.x = j.x < 1.0f ? sqrtf(j.x) - 1.0f : 1.0f - sqrtf(2.0f - j.x);
	j.y = j.y < 1.0f ? sqrtf(j.y) - 1.0f : 1.0f - sqrtf(2.0f - j.y);
	return make_float2(0.5f, 0.5f) + j;
}

__forceinline__ __device__ float2 filter_triangle(const float2& r2)
{
	float u1 = r2.x;
	float u2 = r2.y;
	if (u2 > u1) 
	{
		u1 *= 0.5f;
		u2 -= u1;
	}
	else 
	{
		u2 *= 0.5f;
		u1 -= u2;
	}
	return make_float2(0.5f, 0.5f) + make_float2(u1, u2);
}

__forceinline__ __device__ float2 filter_gaussian(const float2& rnd2)
{
	float r1 = fmaxf(FLT_MIN, rnd2.x);
	float r = sqrtf(-2.0f * logf(r1));
	float theta = M_2PIf * rnd2.y;
	float2 uv = r * make_float2(cosf(theta), sinf(theta));
	return make_float2(0.5f, 0.5f) + 0.375f * uv;
}

__forceinline__ __device__ float2 filter_jitter(const float2& r2)
{
#if FILTER_MODE == FILTER_TENT
	return filter_tent(r2);
#elif FILTER_MODE == FILTER_TRIANGLE
	return filter_triangle(r2);
#elif FILTER_MODE == FILTER_GAUSSIAN
	return filter_gaussian(r2);
#else
	return filter_box(r2);
#endif
}