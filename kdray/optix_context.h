#pragma once

#include "common.h"
#include <optix.h>
#include "cuda/shared_types.h"

namespace kdray
{
	class OptixContext
	{
	public:
		bool Init();
		void Destory();
		bool BuildAccelGeometry(const OptixBuildInput& triangleInput,
			void** buffer, OptixTraversableHandle* traverHandle);
		bool BuildAccelInstance(OptixBuildInput instanceInput,
			void** buffer, OptixTraversableHandle* traverHandle);
		void BuildHitGroupSBT(const std::vector<HitGroupData>& datas);

		void LaunchRayTrace(const RayTraceLaunchParams& launchParams);
		
		void SetupDenoise(uint32_t width, uint32_t height,
			float4* colorImage, float4* albedoImage, float4* normalImage, float4* denoisedImage);
		void InvokeDenoise();

	private:
		CUstream _cuStream = 0;
		OptixDeviceContext _context = nullptr;
		OptixModuleCompileOptions _moduleCompileOptions = {};
		OptixPipelineCompileOptions _pipelineCompileOptions = {};
		OptixPipelineLinkOptions _pipelineLinkOptions = {};

		OptixModule _raytraceModule = nullptr;
		OptixProgramGroup _raygenProgramGroup = nullptr;
		OptixProgramGroup _missProgramGroup = nullptr;
		OptixProgramGroup _missOcclusionProgramGroup = nullptr;
		OptixProgramGroup _hitProgramGroup = nullptr;
		std::array<OptixProgramGroup, 2> _callbacksGroups;
		OptixPipeline _raytracePipeline = nullptr;
		OptixShaderBindingTable _sbt = {};

		CUdeviceptr _launchParamsDevice = 0;
		
		OptixDenoiser _denoiser = nullptr;
		OptixDenoiserSizes _denoiserSizes = {};
		OptixDenoiserParams _denoiserParams = {};
		OptixDenoiserGuideLayer _denoiserGuideLayer = {};
		OptixDenoiserLayer _denoiserLayer = {};
		void* _denoiserHdrIntensity = nullptr;
		void* _denoiserScratch = nullptr;
		void* _denoiserState = nullptr;

		void* _tempBuffer = nullptr;
		size_t _tempBufferSize = 0;

		static void _OptixLogCallback(unsigned int level, const char* tag, const char* message, void* cbdata);
		bool _CreateProgramGroup();
		bool _CreatePipeline();
		void _BuildSBT();
		void _CheckTempBufferSize(size_t size);
	};
}
