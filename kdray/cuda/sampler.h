#pragma once

struct Sampler
{
	unsigned int seed = 0;
};

__forceinline__ __device__ void sampler_init(Sampler& sampler, unsigned int v0, unsigned int v1)
{
	unsigned int s0 = 0;
	for (unsigned int n = 0; n < 4; n++)
	{
		s0 += 0x9e3779b9;
		v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
		v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
	}
	sampler.seed = v0;
}

__forceinline__ __device__ float sampler_get1d(Sampler& sampler)
{
	//Generate random unsigned int in[0, 2 ^ 24)
	const unsigned int LCG_A = 1664525u;
	const unsigned int LCG_C = 1013904223u;
	sampler.seed = (LCG_A * sampler.seed + LCG_C);
	unsigned int seed2 = sampler.seed & 0x00FFFFFF;

	return ((float)seed2 / (float)0x01000000);
}

__forceinline__ __device__ float2 sampler_get2d(Sampler& sampler)
{
	return make_float2(sampler_get1d(sampler), sampler_get1d(sampler));
}
