#include "math.h"

__forceinline__ __device__ unsigned char quantize_byte(float x)
{
    x = clamp(x, 0.0f, 1.0f);
    enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
    return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}

__global__ void tone_map_kernel(uchar4* output_image, float4* render_image, int2 image_size, 
    float exposure, int type)
{
    int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
    int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (pixel_x >= image_size.x) return;
    if (pixel_y >= image_size.y) return;
    int pixel = pixel_x + image_size.x * pixel_y;
    float3 color = make_float3(render_image[pixel]);

    if (type == 0) // clamp
    {
        output_image[pixel] = make_uchar4(
            quantize_byte(color.x),
            quantize_byte(color.y),
            quantize_byte(color.z),
            255u);
        return;
    }

    color *= exposure;

    if (type == 2) // aces
    {
        constexpr float a = 2.51f;
        constexpr float b = 0.03f;
        constexpr float c = 2.43f;
        constexpr float d = 0.59f;
        constexpr float e = 0.14f;
        color = color * (a * color + b) / (color * (c * color + d) + e);
    }
    else if (type == 3) // uncharted2
    {
        constexpr float A = 0.15f;
        constexpr float B = 0.50f;
        constexpr float C = 0.10f;
        constexpr float D = 0.20f;
        constexpr float E = 0.02f;
        constexpr float F = 0.30f;
        constexpr float W = 11.2f;
        color = ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - E / F;
        float white = ((W * (A * W + C * B) + D * E) / (W * (A * W + B) + D * F)) - E / F;
        color /= white;
    }

    // gamma
    constexpr float inv_gamma = 1.0f / 2.2f;
    color = pow(color, inv_gamma);

    output_image[pixel] = make_uchar4(
        quantize_byte(color.x),
        quantize_byte(color.y),
        quantize_byte(color.z),
        255u);
}

void CudaToneMap(uchar4* outputImage, float4* renderImage, int2 imageSize, float exposure, int type)
{
    const int blockSizeX = 32;
    const int blockSizeY = 32;
    int blockNumX = (imageSize.x + blockSizeX - 1) / blockSizeX;
    int blockNumY = (imageSize.y + blockSizeY - 1) / blockSizeY;

    tone_map_kernel
        <<<dim3(blockNumX, blockNumY), dim3(blockSizeX, blockSizeY)>>>
        (outputImage, renderImage, imageSize, exposure, type);
}
