#include "framebuffer.h"
#include "logger.h"
#include "optix_context.h"
#include "renderer.h"

#include <cuda_gl_interop.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace kdray
{
	Framebuffer::Framebuffer()
	{
	}

	Framebuffer::~Framebuffer()
	{
		_ReleaseOpenGLBuffer();
		CudaFree(_colorSrgbImage);
		CudaFree(_albedoImage);
		CudaFree(_normalImage);
		CudaFree(_denoisedImage);
		CudaFree(_colorImage);
	}

	void Framebuffer::Resize(uint32_t width, uint32_t height)
	{
		if (_width == width && _height == height)
		{
			return;
		}

		_width = width;
		_height = height;
		CudaFree(_colorSrgbImage);
		CudaFree(_albedoImage);
		CudaFree(_normalImage);
		CudaFree(_denoisedImage);
		CudaFree(_colorImage);
		_colorSrgbImage = CudaMalloc<float4>(width * height);
		_albedoImage = CudaMalloc<float4>(width * height);
		_normalImage = CudaMalloc<float4>(width * height);
		_denoisedImage = CudaMalloc<float4>(width * height);
		_colorImage = CudaMalloc<uchar4>(width * height);

		Renderer::Instance()->GetOptixContext()->SetupDenoise(width, height, 
			_colorSrgbImage, _albedoImage, _normalImage, _denoisedImage);
	}

	bool Framebuffer::SetOpenGLBuffer(unsigned glBuffer)
	{
		_ReleaseOpenGLBuffer();

		cudaError_t error;
		error = cudaGraphicsGLRegisterBuffer(&_glBufferResource, glBuffer, cudaGraphicsMapFlagsWriteDiscard);
		if (error != CUDA_SUCCESS)
		{
			GetLogger()->critical("register opengl buffer failed. {}", cudaGetErrorString(error));
			return false;
		}
		error = cudaGraphicsMapResources(1, &_glBufferResource, 0);
		if (error != CUDA_SUCCESS)
		{
			cudaGraphicsUnregisterResource(_glBufferResource);
			GetLogger()->critical("map opengl buffer failed. {}", cudaGetErrorString(error));
			return false;
		}
		size_t numBytes;
		error = cudaGraphicsResourceGetMappedPointer((void**)&_glBuffer, &numBytes, _glBufferResource);
		if (error != CUDA_SUCCESS)
		{
			cudaGraphicsUnmapResources(1, &_glBufferResource, 0);
			cudaGraphicsUnregisterResource(_glBufferResource);
			_glBuffer = nullptr;
			GetLogger()->critical("get opengl buffer pointer failed. {}", cudaGetErrorString(error));
			return false;
		}
		return true;
	}

	void Framebuffer::InvokeDenoise()
	{
		Renderer::Instance()->GetOptixContext()->InvokeDenoise();
	}

	void Framebuffer::SaveToFile(const std::string& filepath)
	{
		std::size_t found = filepath.find_last_of(".");
		std::string extStr = filepath.substr(found + 1);
		if (extStr == "hdr")
		{
			size_t bufferSize = _width * _height * 4 * sizeof(float);
			float* imageData = new float[_width * _height * 4];
			memset(imageData, 0, bufferSize);
			cudaMemcpy(imageData, _colorSrgbImage, bufferSize, cudaMemcpyDeviceToHost);
			stbi_flip_vertically_on_write(true);
			stbi_write_hdr(filepath.c_str(), _width, _height, 4, imageData);
			delete[] imageData;
		}
		else
		{
			size_t bufferSize = _width * _height * 4;
			auto imageData = new unsigned char[bufferSize];
			memset(imageData, 0, bufferSize);
			cudaMemcpy(imageData, _colorImage, bufferSize, cudaMemcpyDeviceToHost);
			stbi_flip_vertically_on_write(true);
			stbi_write_png(filepath.c_str(), _width, _height, 4, imageData, 0);
			delete[] imageData;
		}
	}

	void Framebuffer::_ReleaseOpenGLBuffer()
	{
		if (_glBufferResource)
		{
			cudaGraphicsUnmapResources(1, &_glBufferResource, 0);
			cudaGraphicsUnregisterResource(_glBufferResource);
			_glBuffer = nullptr;
		}
	}
}