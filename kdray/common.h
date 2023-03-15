#pragma once

#include <string>
#include <array>
#include <vector>
#include <memory>
#include <iostream>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

namespace kdray
{
	template<typename T>
	T* CudaMalloc(size_t size)
	{
		T* devicePtr = nullptr;
		cudaMalloc(reinterpret_cast<void**>(&devicePtr), size * sizeof(T));
		return devicePtr;
	}

	template<typename T>
	T* CudaMalloc(const std::vector<T>& vt)
	{
		T* deviceBuffer = nullptr;
		size_t size = vt.size() * sizeof(T);
		cudaMalloc((void**)&deviceBuffer, size);
		cudaMemcpy(deviceBuffer, vt.data(), size, cudaMemcpyHostToDevice);
		return deviceBuffer;
	}

	inline void CudaFree(void* devicePtr)
	{
		if (devicePtr)
		{
			cudaFree(devicePtr);
		}
	}

	inline bool FloatEqual(const float f1, const float f2, const float v = 0.01f)
	{
		return std::abs(f1 - f2) < v;
	}
}
