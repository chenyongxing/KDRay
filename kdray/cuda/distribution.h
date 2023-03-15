#pragma once

__forceinline__ __device__ int binary_search(const float* data, int size, float value)
{
	int lower = 0;
	int upper = size - 1;
	while (lower < upper)
	{
		int mid = (lower + upper) >> 1;
		if (value < data[mid])
			upper = mid;
		else
			lower = mid + 1;
	}
	return lower;
}

__forceinline__ __device__ int distribution1d_sample(const float* cdf, int size, float rnd, float* pdf)
{
	int index = binary_search(cdf, size, rnd * cdf[size - 1]);
	float weight = index == 0 ? cdf[index] : cdf[index] - cdf[index - 1];
	*pdf = weight / cdf[size - 1] * size;
	return index;
}

__forceinline__ __device__ int2 distribution2d_sample(const float* marginal, const float* conditional, 
	int width, int height, const float2& r2, float* pdf) 
{
	float pdfs[2];
	int v = distribution1d_sample(marginal, height, r2.y, &pdfs[1]);
	int u = distribution1d_sample(conditional + v * width, width, r2.x, &pdfs[0]);
	*pdf = pdfs[0] * pdfs[1];
	return make_int2(u, v);
}

__forceinline__ __device__ float distribution2d_pdf(const float* marginal, const float* conditional, 
	int width, int height, float u, float v)
{
	int2 uv = make_int2(u * width, v * height);
	float weight = 0.0f;
	if (uv.x != 0)
	{
		int index = uv.x + uv.y * width;
		weight = conditional[index] - conditional[index - 1];
	}
	else
		weight = conditional[uv.y * width];
	return weight / marginal[height - 1] * width * height;
}
