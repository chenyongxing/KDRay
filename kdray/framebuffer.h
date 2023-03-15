#pragma once

#include "common.h"

namespace kdray
{
	class Framebuffer
	{
	public:
		Framebuffer();
		~Framebuffer();
		void Resize(uint32_t width, uint32_t height);
		void SaveToFile(const std::string& filepath);
		bool SetOpenGLBuffer(unsigned glBuffer);
		void InvokeDenoise();

		inline uint32_t GetWidth()
		{
			return _width;
		}

		inline uint32_t GetHeight()
		{
			return _height;
		}

		inline float4* GetColorSrgbImage()
		{
			return _colorSrgbImage;
		}

		inline float4* GetNormalImage()
		{
			return _normalImage;
		}

		inline float4* GetAlbedoImage()
		{
			return _albedoImage;
		}

		inline float4* GetDenoisedImage()
		{
			return _denoisedImage;
		}

		inline uchar4* GetColorImage()
		{
			return _colorImage;
		}

		inline void CopyColorImageToGLBuffer()
		{
			if (_glBuffer && _colorImage)
			{
				cudaMemcpy(_glBuffer, _colorImage, _width * _height * 4, cudaMemcpyDeviceToDevice);
			}
		}

	private:
		uint32_t _width = 0;
		uint32_t _height = 0;
		float4* _colorSrgbImage = nullptr; 
		float4* _albedoImage = nullptr;
		float4* _normalImage = nullptr;
		float4* _denoisedImage = nullptr;
		uchar4* _colorImage = nullptr;
		cudaGraphicsResource* _glBufferResource = nullptr;
		uchar4* _glBuffer = nullptr;

		void _ReleaseOpenGLBuffer();
	};
}