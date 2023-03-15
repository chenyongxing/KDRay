#pragma once

#include "common.h"

namespace kdray
{
	class Texture
	{
	public:
		bool isSRGB= false;
		float2 scale{ 1.0f, 1.0f };
		float2 offset{ 0.0f, 0.0f };
		cudaTextureObject_t cuTexture = 0;
		// GPU Address
		float* distributionMarginal = nullptr;
		float* distributionConditional = nullptr;

		Texture();
		virtual ~Texture();
		bool LoadFromFile(const std::string& filepath);

		inline int GetWidth() { return _width; }
		inline int GetHeight() { return _height; }

	private:
		cudaArray_t _cuArray = nullptr;
		int _width = 0;
		int _height = 0;
		bool _isHDR = false;

		void _CreateCuTexture(const void* image);
		void _CreateDistribution2D(const float* image);
	};
}
