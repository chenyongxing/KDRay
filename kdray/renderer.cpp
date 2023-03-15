#include "renderer.h"
#include "logger.h"
#include "optix_context.h"
#include "scene.h"

extern void CudaToneMap(uchar4* outputImage, float4* renderImage, int2 imageSize, float exposure, int type);

namespace kdray
{
	Renderer* Renderer::Instance()
	{
		static Renderer instance;
		return &instance;
	}

	bool Renderer::Init()
	{
		LoggerInit();

		_optixContext = std::make_unique<OptixContext>();
		if (!_optixContext->Init())
		{
			Destory();
			return false;
		}

		_framebuffer = std::make_unique<Framebuffer>();
		_framebuffer->Resize(800, 600);
		_scene = std::make_unique<Scene>();
		_camera = std::make_unique<Camera>();
		_camera->LookAt(make_float3(5.0f), make_float3(0.0f), make_float3(0.0f, 0.0f, 1.0f));
		_camera->Perspect(45.f, 800.f / 600.f, 0.1f, 100.f);
		return true;
	}

	void Renderer::Destory()
	{
		if (_camera)
		{
			_camera.reset();
		}

		if (_scene)
		{
			_scene.reset();
		}

		if (_framebuffer)
		{
			_framebuffer.reset();
		}

		if (_optixContext)
		{
			_optixContext->Destory();
			_optixContext.reset();
		}

		LoggerDestory();
	}

	void Renderer::RenderOneFrame()
	{
		if (!_framebuffer) return;

		if (_launchParams.frameCount >= _sampleFrameCount)
		{
			_PostrOneFrame(_setting.denoise);
			return;
		}
		
		_scene->Update();
		_camera->Update();

		// optix ray trace
		_launchParams.width = _framebuffer->GetWidth();
		_launchParams.height = _framebuffer->GetHeight();
		_launchParams.colorSrgbImage = _framebuffer->GetColorSrgbImage();
		_launchParams.albedoImage = _framebuffer->GetAlbedoImage();
		_launchParams.normalImage = _framebuffer->GetNormalImage();
		_launchParams.traverHandle = _scene->GetTraverHandle();
		_launchParams.rasterToWorldMatrix = _camera->GetRasterToWorldMatrix();
		_launchParams.cameraToWorldMatrix = _camera->GetTransformMatrix();
		_launchParams.focalDistance = _camera->focalDistance;
		if (_camera->fStop > FLT_EPSILON)
		{
			_launchParams.lensRadius = _camera->focalDistance / (2.0f * _camera->fStop);
		}
		_launchParams.lightsNum = _scene->GetLightsNum();
		_launchParams.lights = _scene->GetLightsBuffer();
		_optixContext->LaunchRayTrace(_launchParams);
		_launchParams.frameCount++;

		// 第0帧额外渲染一次Gbuffer
		if (_launchParams.frameCount == 1)
		{
			uint32_t lastIntegrator = _launchParams.integratorCallback;
			_launchParams.integratorCallback = (uint32_t)IntegratorType::GBuffer;
			_optixContext->LaunchRayTrace(_launchParams);
			_launchParams.integratorCallback = lastIntegrator;
		}

		// denoise
		bool denoised = false;
		if (_setting.denoise && _launchParams.frameCount == _sampleFrameCount)
		{
			_framebuffer->InvokeDenoise();
			denoised = true;
		}

		_PostrOneFrame(denoised);
	}

	void Renderer::_PostrOneFrame(bool denoised)
	{
		if (_setting.aov == AOVType::Beauty)
		{
			CudaToneMap(_framebuffer->GetColorImage(),
				denoised ? _framebuffer->GetDenoisedImage() : _framebuffer->GetColorSrgbImage(),
				make_int2(_framebuffer->GetWidth(), _framebuffer->GetHeight()), _setting.exposure, (int)_setting.toneMap);
		}
		else if (_setting.aov == AOVType::Albedo)
		{
			CudaToneMap(_framebuffer->GetColorImage(), _framebuffer->GetAlbedoImage(),
				make_int2(_framebuffer->GetWidth(), _framebuffer->GetHeight()), _setting.exposure, (int)ToneMapType::Clamp);
		}
		else if (_setting.aov == AOVType::Normal)
		{
			CudaToneMap(_framebuffer->GetColorImage(), _framebuffer->GetNormalImage(),
				make_int2(_framebuffer->GetWidth(), _framebuffer->GetHeight()), _setting.exposure, (int)ToneMapType::Clamp);
		}
		
		_framebuffer->CopyColorImageToGLBuffer();
	}

	void Renderer::SetRenderSetting(const RenderSetting& setting)
	{
		_setting = setting;

		bool changed = false;
		if (_launchParams.integratorCallback != (uint32_t)setting.integrator)
		{
			_launchParams.integratorCallback = (uint32_t)setting.integrator;
			changed = true;
		}
		if (_launchParams.maxBounce != std::max(0, std::min(100, setting.maxBounce)))
		{
			_launchParams.maxBounce = std::max(0, std::min(100, setting.maxBounce));
			changed = true;
		}
		if (_launchParams.pathRadianceClamp != std::max(1.f, setting.intensityClamp))
		{
			_launchParams.pathRadianceClamp = std::max(1.f, setting.intensityClamp);
			changed = true;
		}
		if (_sampleFrameCount != std::max(1u, setting.sampleCount))
		{
			_sampleFrameCount = std::max(1u, setting.sampleCount);
			changed = true;
		}

		if (changed)
			ResetAccumCount();
	}
}
