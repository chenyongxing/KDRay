#include "texture.h"
#include "logger.h"
#include "cuda/math.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace kdray
{
	Texture::Texture()
	{
	}

	Texture::~Texture()
	{
		if (distributionMarginal)
		{
			CudaFree(distributionMarginal);
			distributionMarginal = nullptr;
		}
		if (distributionConditional)
		{
			CudaFree(distributionConditional);
			distributionConditional = nullptr;
		}
		if (cuTexture)
		{
			cudaDestroyTextureObject(cuTexture);
			cuTexture = 0;
		}
		if (_cuArray)
		{
			cudaFreeArray(_cuArray);
			_cuArray = nullptr;
		}
	}

	bool Texture::LoadFromFile(const std::string& filepath)
	{
		std::size_t found = filepath.find_last_of(".");
		std::string extStr = filepath.substr(found + 1);
		if (extStr == "hdr")
		{
			_isHDR = true;
		}
		else
		{
			_isHDR = false;
		}

		stbi_set_flip_vertically_on_load(true);
		int comp = 4;
		int reqComp = STBI_rgb_alpha;
		if (!_isHDR)
		{
			unsigned char* image = stbi_load(filepath.c_str(), &_width, &_height, &comp, reqComp);
			if (image)
			{
				_CreateCuTexture(image);
				STBI_FREE(image);
				return true;
			}
		}
		else
		{
			float* image = stbi_loadf(filepath.c_str(), &_width, &_height, &comp, reqComp);
			if (image)
			{
				_CreateCuTexture(image);
				_CreateDistribution2D(image);
				STBI_FREE(image);
				return true;
			}
		}

		return false;
	}

	void Texture::_CreateCuTexture(const void* image)
	{
		cudaChannelFormatDesc channelDesc = {};
		int32_t pitch = 0;
		if (_isHDR)
		{
			channelDesc = cudaCreateChannelDesc<float4>();
			pitch = _width * 4 * sizeof(float);
		}
		else
		{
			channelDesc = cudaCreateChannelDesc<uchar4>();
			pitch = _width * 4 * sizeof(unsigned char);
		}
		cudaMallocArray(&_cuArray, &channelDesc, _width, _height);
		cudaMemcpy2DToArray(_cuArray, 0, 0, image, pitch, pitch, _height, cudaMemcpyHostToDevice);
		cudaResourceDesc resDesc = {};
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = _cuArray;

		//sampler
		cudaTextureDesc texDesc = {};
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = _isHDR ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
		texDesc.normalizedCoords = 1;
		texDesc.maxAnisotropy = 1;
		texDesc.maxMipmapLevelClamp = 99;
		texDesc.minMipmapLevelClamp = 0;
		texDesc.mipmapFilterMode = cudaFilterModeLinear;
		texDesc.borderColor[0] = 1.0f;
		texDesc.sRGB = isSRGB;

		cudaError_t result = cudaCreateTextureObject(&cuTexture, &resDesc, &texDesc, nullptr);
		if (result != cudaSuccess)
		{
			std::stringstream ss;
			ss << "createTexture failed" << std::endl
				<< "file=" __FILE__ << ", line=" << __LINE__ << std::endl
				<< "error: " << cudaGetErrorString(result) << std::endl;
			GetLogger()->critical("{}", ss.str());
		}
	}

	static inline void distribution1d_init(float* cdf, int size)
	{
		for (int i = 1; i < size; ++i)
		{
			cdf[i] += cdf[i - 1];
		}
	}

	void Texture::_CreateDistribution2D(const float* image)
	{
		if (!_isHDR) return;

		std::vector<float> conditionals(_width * _height);
		for (int y = 0; y < _height; ++y)
		{
			for (int x = 0; x < _width; ++x)
			{
				int index = x + y * _width;
				float3 color;
				color.x = image[index * 4];
				color.y = image[index * 4 + 1];
				color.z = image[index * 4 + 2];
				conditionals[index] = luminance(color);
				float sinTheta = sinf(M_PIf * (y + 0.5f) / _height);
				conditionals[index] *= sinTheta;
			}
		}

		for (int y = 0; y < _height; ++y) 
		{
			distribution1d_init(conditionals.data() + y * _width, _width);
		}

		std::vector<float> marginal(_height);
		for (int y = 0; y < _height; ++y)
		{
			marginal[y] = conditionals[y * _width + _width - 1];
		}
		distribution1d_init(marginal.data(), _height);
		
		distributionMarginal = CudaMalloc(marginal);
		distributionConditional = CudaMalloc(conditionals);
	}
}
