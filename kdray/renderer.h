#pragma once

#include "common.h"
#include "framebuffer.h"
#include "camera.h"
#include "cuda/shared_types.h"

namespace kdray
{
	enum struct AOVType
	{
		Beauty = 0,
		Albedo = 1,
		Normal = 2
	};
	enum struct IntegratorType
	{
		PathTracing = 0,
		AmbientOcclusion = 1,
		GBuffer = 2
	};
	enum struct ToneMapType
	{
		Clamp = 0,
		Gamma = 1,
		ACES = 2,
		Uncharted = 3
	};
	struct RenderSetting
	{
		AOVType aov = AOVType::Beauty;
		IntegratorType integrator = IntegratorType::PathTracing;
		uint32_t sampleCount = 2048;
		int maxBounce = 5;
		float intensityClamp = 30.0f;
		float exposure = 1.0f;
		ToneMapType toneMap = ToneMapType::Gamma;
		bool denoise = true;
	};

	class OptixContext;
	class Scene;
	class Renderer
	{
	public:
		static Renderer* Instance();

		bool Init();
		void Destory();
		void RenderOneFrame();
		void SetRenderSetting(const RenderSetting& setting);

		inline OptixContext* GetOptixContext()
		{
			return _optixContext.get();
		}

		inline Framebuffer* GetFramebuffer()
		{
			return _framebuffer.get();
		}

		inline Scene* GetScene()
		{
			return _scene.get();
		}

		inline Camera* GetCamera()
		{
			return _camera.get();
		}

		inline RenderSetting& GetRenderSetting()
		{
			return _setting;
		}

		inline uint32_t GetAccumCount()
		{
			return _launchParams.frameCount;
		}

		inline void ResetAccumCount()
		{
			_launchParams.frameCount = 0;
		}

	private:
		std::unique_ptr<OptixContext> _optixContext;
		std::unique_ptr<Framebuffer> _framebuffer;
		std::unique_ptr<Scene> _scene;
		std::unique_ptr<Camera> _camera;

		RenderSetting _setting;
		uint32_t _sampleFrameCount = 2048;
		RayTraceLaunchParams _launchParams;

		void _PostrOneFrame(bool denoise);
	};
}
