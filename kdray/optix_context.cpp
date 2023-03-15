#include "optix_context.h"
#include "logger.h"

#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

extern "C" char cuRayTracer[];

namespace kdray
{
	namespace {
		std::array<char, 4096> Log;
		constexpr unsigned int MaxTraceDepth = 30;

		struct SbtRecordEmpty
		{
			__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		};

		struct SbtRecordHitGroup
		{
			__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
			HitGroupData data;
		};
	}

	bool OptixContext::Init()
	{
		cudaFree(0);
		int numDevices;
		cudaGetDeviceCount(&numDevices);
		if (numDevices == 0)
		{
			GetLogger()->critical("find cuda device failed");
			return false;
		}

		OptixResult result = optixInit();
		if (result != OPTIX_SUCCESS)
		{
			GetLogger()->critical("optixInit failed. {}", optixGetErrorString(result));
			return false;
		}

		CUcontext cuContext = 0;
		OptixDeviceContextOptions options = {};
		options.logCallbackFunction = &_OptixLogCallback;
		options.logCallbackLevel = 4;
		if (optixDeviceContextCreate(cuContext, &options, &_context) != OPTIX_SUCCESS)
		{
			GetLogger()->critical("optixDeviceContextCreate failed");
			return false;
		}

		_moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#if !defined(NDEBUG)
		_moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
		_moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

		_pipelineCompileOptions.usesMotionBlur = false;
		_pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
		_pipelineCompileOptions.numPayloadValues = 2;
		_pipelineCompileOptions.numAttributeValues = 0;
#if !defined(NDEBUG)
		_pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG;
#endif
		_pipelineCompileOptions.pipelineLaunchParamsVariableName = "Params";

		_pipelineLinkOptions.maxTraceDepth = MaxTraceDepth;
		_pipelineLinkOptions.debugLevel = _moduleCompileOptions.debugLevel;

		const std::string ptxCode = cuRayTracer;
		size_t logSize = Log.size();
		if (optixModuleCreateFromPTX(
			_context,
			&_moduleCompileOptions,
			&_pipelineCompileOptions,
			ptxCode.c_str(),
			ptxCode.size(),
			Log.data(),
			&logSize,
			&_raytraceModule
		) != OPTIX_SUCCESS)
		{
			std::stringstream ss;
			ss << "optixModuleCreateFromPTX failed" << std::endl
				<< "file=" __FILE__ << ", line=" << __LINE__ << std::endl
				<< "error log: " << Log.data() << std::endl;
			GetLogger()->critical("{}", ss.str());
			return false;
		}

		if (!_CreateProgramGroup())
		{
			return false;
		}
		if (!_CreatePipeline())
		{
			return false;
		}

		_BuildSBT();

		cudaMalloc(reinterpret_cast<void**>(&_launchParamsDevice), sizeof(RayTraceLaunchParams));

		OptixDenoiserOptions denoiserOptions = {};
		denoiserOptions.guideAlbedo = 1;
		denoiserOptions.guideNormal = 1;
		if (optixDenoiserCreate(_context, OPTIX_DENOISER_MODEL_KIND_HDR, 
			&denoiserOptions, &_denoiser) != OPTIX_SUCCESS)
		{
			GetLogger()->critical("optixDenoiserCreate failed");
			return false;
		}
		_denoiserParams.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
		_denoiserParams.blendFactor = 0.0f;
		cudaMalloc(&_denoiserHdrIntensity, sizeof(float));
		_denoiserParams.hdrIntensity = (CUdeviceptr)_denoiserHdrIntensity;

		return true;
	}

	void OptixContext::Destory()
	{
		CudaFree(_denoiserHdrIntensity);
		CudaFree(_denoiserState);
		CudaFree(_denoiserScratch);
		if (_denoiser)
		{
			optixDenoiserDestroy(_denoiser);
			_denoiser = nullptr;
		}
		
		if (_context)
		{
			optixDeviceContextDestroy(_context);
			_context = nullptr;
		}
	}

	bool OptixContext::BuildAccelGeometry(const OptixBuildInput& triangleInput,
		void** buffer, OptixTraversableHandle* traverHandle)
	{
		// query accel size
		OptixAccelBuildOptions buildOptions = {};
		buildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
		OptixAccelBufferSizes gasBufferSizes;
		optixAccelComputeMemoryUsage(
			_context,
			&buildOptions,
			&triangleInput,
			1,
			&gasBufferSizes
		);
		// query compaction size info
		void* compactedSizeGpu = nullptr;
		cudaMalloc(&compactedSizeGpu, sizeof(uint64_t));
		OptixAccelEmitDesc compactedSizeEmitDesc;
		compactedSizeEmitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		compactedSizeEmitDesc.result = (CUdeviceptr)compactedSizeGpu;
		// alloc gpu mem
		void* outputBuffer = nullptr;
		cudaMalloc(&outputBuffer, gasBufferSizes.outputSizeInBytes);
		_CheckTempBufferSize(gasBufferSizes.tempSizeInBytes);
		// build accel
		optixAccelBuild(
			_context,
			_cuStream,
			&buildOptions,
			&triangleInput,
			1,
			(CUdeviceptr)_tempBuffer,
			gasBufferSizes.tempSizeInBytes,
			(CUdeviceptr)outputBuffer,
			gasBufferSizes.outputSizeInBytes,
			traverHandle,
			&compactedSizeEmitDesc,
			1
		);
		cudaDeviceSynchronize();
		// query compaction size
		uint64_t compactedSize;
		cudaMemcpy(&compactedSize, compactedSizeGpu, sizeof(uint64_t), cudaMemcpyDeviceToHost);
		cudaFree(compactedSizeGpu);
		// compaction accel
		cudaMalloc(buffer, compactedSize);
		optixAccelCompact(
			_context,
			_cuStream,
			*traverHandle,
			(CUdeviceptr)*buffer,
			compactedSize,
			traverHandle
		);
		cudaDeviceSynchronize();
		cudaFree(outputBuffer);

		return true;
	}

	bool OptixContext::BuildAccelInstance(OptixBuildInput instanceInput,
		void** buffer, OptixTraversableHandle* traverHandle)
	{
		OptixAccelBuildOptions buildOptions = {};
		buildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
		buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		// query accel size
		OptixAccelBufferSizes accelBufferSizes;
		optixAccelComputeMemoryUsage(
			_context,
			&buildOptions,
			&instanceInput,
			1,
			&accelBufferSizes
		);
		// alloc gpu mem
		cudaMalloc(buffer, accelBufferSizes.outputSizeInBytes);
		_CheckTempBufferSize(accelBufferSizes.tempSizeInBytes);
		// build accel
		optixAccelBuild(
			_context,
			_cuStream,
			&buildOptions,
			&instanceInput,
			1,
			(CUdeviceptr)_tempBuffer,
			accelBufferSizes.tempSizeInBytes,
			(CUdeviceptr)*buffer,
			accelBufferSizes.outputSizeInBytes,
			traverHandle,
			nullptr,
			0
		);
		cudaDeviceSynchronize();

		return true;
	}

	void OptixContext::LaunchRayTrace(const RayTraceLaunchParams& launchParams)
	{
		cudaMemcpy(reinterpret_cast<void*>(_launchParamsDevice), &launchParams,
			sizeof(RayTraceLaunchParams), cudaMemcpyHostToDevice);

		OptixResult result = optixLaunch(
			_raytracePipeline,
			_cuStream,
			_launchParamsDevice,
			sizeof(RayTraceLaunchParams),
			&_sbt,
			launchParams.width,
			launchParams.height,
			1);
		if (result != OPTIX_SUCCESS)
		{
			std::stringstream ss;
			ss << "optixLaunch failed" << std::endl
				<< "file=" __FILE__ << ", line=" << __LINE__ << std::endl
				<< "error: " << optixGetErrorString(result) << std::endl;
			GetLogger()->critical("{}", ss.str());
		}
		cudaDeviceSynchronize();
	}

	void OptixContext::SetupDenoise(uint32_t width, uint32_t height,
		float4* colorImage, float4* albedoImage, float4* normalImage, float4* denoisedImage)
	{
		optixDenoiserComputeMemoryResources(_denoiser, width, height, &_denoiserSizes);

		CudaFree(_denoiserState);
		CudaFree(_denoiserScratch);
		cudaMalloc(&_denoiserState, _denoiserSizes.stateSizeInBytes);
		cudaMalloc(&_denoiserScratch, _denoiserSizes.withoutOverlapScratchSizeInBytes);
		
		optixDenoiserSetup(_denoiser, _cuStream,
			width, height,
			(CUdeviceptr)_denoiserState,
			_denoiserSizes.stateSizeInBytes,
			(CUdeviceptr)_denoiserScratch,
			_denoiserSizes.withoutOverlapScratchSizeInBytes);

		_denoiserLayer.input.data = (CUdeviceptr)colorImage;
		_denoiserLayer.input.width = width;
		_denoiserLayer.input.height = height;
		_denoiserLayer.input.rowStrideInBytes = width * sizeof(float4);
		_denoiserLayer.input.pixelStrideInBytes = sizeof(float4);
		_denoiserLayer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;

		_denoiserLayer.output.data = (CUdeviceptr)denoisedImage;
		_denoiserLayer.output.width = width;
		_denoiserLayer.output.height = height;
		_denoiserLayer.output.rowStrideInBytes = width * sizeof(float4);
		_denoiserLayer.output.pixelStrideInBytes = sizeof(float4);
		_denoiserLayer.output.format = OPTIX_PIXEL_FORMAT_FLOAT4;

		_denoiserGuideLayer.albedo.data = (CUdeviceptr)albedoImage;
		_denoiserGuideLayer.albedo.width = width;
		_denoiserGuideLayer.albedo.height = height;
		_denoiserGuideLayer.albedo.rowStrideInBytes = width * sizeof(float4);
		_denoiserGuideLayer.albedo.pixelStrideInBytes = sizeof(float4);
		_denoiserGuideLayer.albedo.format = OPTIX_PIXEL_FORMAT_FLOAT4;

		_denoiserGuideLayer.normal.data = (CUdeviceptr)normalImage;
		_denoiserGuideLayer.normal.width = width;
		_denoiserGuideLayer.normal.height = height;
		_denoiserGuideLayer.normal.rowStrideInBytes = width * sizeof(float4);
		_denoiserGuideLayer.normal.pixelStrideInBytes = sizeof(float4);
		_denoiserGuideLayer.normal.format = OPTIX_PIXEL_FORMAT_FLOAT4;
	}

	void OptixContext::InvokeDenoise()
	{
		optixDenoiserComputeIntensity(_denoiser, _cuStream, &_denoiserLayer.input, _denoiserParams.hdrIntensity,
			(CUdeviceptr)_denoiserScratch, _denoiserSizes.withoutOverlapScratchSizeInBytes);

		optixDenoiserInvoke(_denoiser, _cuStream, &_denoiserParams,
			(CUdeviceptr)_denoiserState, _denoiserSizes.stateSizeInBytes,
			&_denoiserGuideLayer, &_denoiserLayer, 1, 0, 0,
			(CUdeviceptr)_denoiserScratch, _denoiserSizes.withoutOverlapScratchSizeInBytes);
	}

	void OptixContext::_OptixLogCallback(unsigned int level, const char* tag, const char* message, void* cbdata)
	{
		switch (level)
		{
		case 1:
			GetLogger()->critical("[{}]: {}", tag, message);
			break;
		case 2:
			GetLogger()->error("[{}]: {}", tag, message);
			break;
		case 3:
			GetLogger()->warn("[{}]: {}", tag, message);
			break;
		case 4:
			GetLogger()->info("[{}]: {}", tag, message);
			break;
		default:
			break;
		}
	}

	bool OptixContext::_CreateProgramGroup()
	{
		OptixProgramGroupOptions programGroupOptions = {};
		OptixProgramGroupDesc programGroupDesc = {};

		size_t logSize = Log.size();
		programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		programGroupDesc.raygen.module = _raytraceModule;
		programGroupDesc.raygen.entryFunctionName = "__raygen__main";
		if (optixProgramGroupCreate(_context,
			&programGroupDesc,
			1,
			&programGroupOptions,
			Log.data(),
			&logSize,
			&_raygenProgramGroup
		) != OPTIX_SUCCESS)
		{
			std::stringstream ss;
			ss << "optixProgramGroupCreate failed" << std::endl
				<< "file=" __FILE__ << ", line=" << __LINE__ << std::endl
				<< "error log: " << Log.data() << std::endl;
			GetLogger()->critical("{}", ss.str());
			return false;
		}

		logSize = Log.size();
		programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		programGroupDesc.miss.module = _raytraceModule;
		programGroupDesc.miss.entryFunctionName = "__miss__main";
		if (optixProgramGroupCreate(_context,
			&programGroupDesc,
			1,
			&programGroupOptions,
			Log.data(),
			&logSize,
			&_missProgramGroup
		) != OPTIX_SUCCESS)
		{
			std::stringstream ss;
			ss << "optixProgramGroupCreate failed" << std::endl
				<< "file=" __FILE__ << ", line=" << __LINE__ << std::endl
				<< "error log: " << Log.data() << std::endl;
			GetLogger()->critical("{}", ss.str());
			return false;
		}

		logSize = Log.size();
		programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		programGroupDesc.miss.module = _raytraceModule;
		programGroupDesc.miss.entryFunctionName = "__miss__occlusion";
		if (optixProgramGroupCreate(_context,
			&programGroupDesc,
			1,
			&programGroupOptions,
			Log.data(),
			&logSize,
			&_missOcclusionProgramGroup
		) != OPTIX_SUCCESS)
		{
			std::stringstream ss;
			ss << "optixProgramGroupCreate failed" << std::endl
				<< "file=" __FILE__ << ", line=" << __LINE__ << std::endl
				<< "error log: " << Log.data() << std::endl;
			GetLogger()->critical("{}", ss.str());
			return false;
		}

		logSize = Log.size();
		programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		programGroupDesc.hitgroup.moduleCH = _raytraceModule;
		programGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__main";
		if (optixProgramGroupCreate(_context,
			&programGroupDesc,
			1,
			&programGroupOptions,
			Log.data(),
			&logSize,
			&_hitProgramGroup
		) != OPTIX_SUCCESS)
		{
			std::stringstream ss;
			ss << "optixProgramGroupCreate failed" << std::endl
				<< "file=" __FILE__ << ", line=" << __LINE__ << std::endl
				<< "error log: " << Log.data() << std::endl;
			GetLogger()->critical("{}", ss.str());
			return false;
		}

		logSize = Log.size();
		std::vector<OptixProgramGroupDesc> callbacksProgramGroup;
		programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
		programGroupDesc.callables.moduleDC = _raytraceModule;
		programGroupDesc.callables.entryFunctionNameDC = "__direct_callable__integrator_pt";
		callbacksProgramGroup.emplace_back(programGroupDesc);
		programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
		programGroupDesc.callables.moduleDC = _raytraceModule;
		programGroupDesc.callables.entryFunctionNameDC = "__direct_callable__integrator_ao";
		callbacksProgramGroup.emplace_back(programGroupDesc);
		if (optixProgramGroupCreate(_context,
			callbacksProgramGroup.data(),
			(uint32_t)callbacksProgramGroup.size(),
			&programGroupOptions,
			Log.data(),
			&logSize,
			_callbacksGroups.data()
		) != OPTIX_SUCCESS)
		{
			std::stringstream ss;
			ss << "optixProgramGroupCreate failed" << std::endl
				<< "file=" __FILE__ << ", line=" << __LINE__ << std::endl
				<< "error log: " << Log.data() << std::endl;
			GetLogger()->critical("{}", ss.str());
			return false;
		}

		return true;
	}

	bool OptixContext::_CreatePipeline()
	{
		std::vector<OptixProgramGroup> programGroups;
		programGroups.push_back(_raygenProgramGroup);
		programGroups.push_back(_missProgramGroup);
		programGroups.push_back(_missOcclusionProgramGroup);
		programGroups.push_back(_hitProgramGroup);
		for (OptixProgramGroup& callbacksGroup : _callbacksGroups)
		{
			programGroups.push_back(callbacksGroup);
		}
		
		size_t logSize = Log.size();
		if (optixPipelineCreate(
			_context,
			&_pipelineCompileOptions,
			&_pipelineLinkOptions,
			programGroups.data(),
			(uint32_t)programGroups.size(),
			Log.data(),
			&logSize,
			&_raytracePipeline
		) != OPTIX_SUCCESS)
		{
			std::stringstream ss;
			ss << "optixPipelineCreate failed" << std::endl
				<< "file=" __FILE__ << ", line=" << __LINE__ << std::endl
				<< "error log: " << Log.data() << std::endl;
			GetLogger()->critical("{}", ss.str());
			return false;
		}

		OptixStackSizes stackSizes = {};
		for (OptixProgramGroup& progGroup : programGroups)
		{
			optixUtilAccumulateStackSizes(progGroup, &stackSizes);
		}

		uint32_t directCallableStackSizeFromTraversal;
		uint32_t directCallableStackSizeFromState;
		uint32_t continuationStackSize;
		optixUtilComputeStackSizes(
			&stackSizes,
			MaxTraceDepth,
			1,  // maxCCDepth
			1,  // maxDCDEpth
			&directCallableStackSizeFromTraversal,
			&directCallableStackSizeFromState,
			&continuationStackSize
		);

		if (optixPipelineSetStackSize(
			_raytracePipeline,
			directCallableStackSizeFromTraversal,
			directCallableStackSizeFromState,
			continuationStackSize,
			2  // maxTraversableDepth
		) != OPTIX_SUCCESS)
		{
			std::stringstream ss;
			ss << "optixPipelineSetStackSize failed" << std::endl
				<< "file=" __FILE__ << ", line=" << __LINE__ << std::endl
				<< "error log: " << Log.data() << std::endl;
			GetLogger()->critical("{}", ss.str());
			return false;
		}
		return true;
	}

	void OptixContext::_BuildSBT()
	{
		CudaFree((void*)_sbt.raygenRecord);
		CudaFree((void*)_sbt.missRecordBase);

		SbtRecordEmpty raygenRecordHost;
		optixSbtRecordPackHeader(_raygenProgramGroup, &raygenRecordHost);
		CUdeviceptr raygenRecordDevice = 0;
		cudaMalloc(reinterpret_cast<void**>(&raygenRecordDevice), sizeof(SbtRecordEmpty));
		cudaMemcpy(reinterpret_cast<void*>(raygenRecordDevice), &raygenRecordHost,
			sizeof(SbtRecordEmpty), cudaMemcpyHostToDevice);
		_sbt.raygenRecord = raygenRecordDevice;

		std::vector<SbtRecordEmpty> missRecords(2);
		{
			SbtRecordEmpty& missRecordHost = missRecords[0];
			optixSbtRecordPackHeader(_missProgramGroup, &missRecordHost);
		}
		{
			SbtRecordEmpty& missRecordHost = missRecords[1];
			optixSbtRecordPackHeader(_missOcclusionProgramGroup, &missRecordHost);
		}
		_sbt.missRecordBase = (CUdeviceptr)CudaMalloc(missRecords);
		_sbt.missRecordStrideInBytes = (uint32_t)sizeof(SbtRecordEmpty);
		_sbt.missRecordCount = (uint32_t)missRecords.size();

		std::vector<SbtRecordEmpty> callableRecords(_callbacksGroups.size());
		for (uint32_t i = 0; i < callableRecords.size(); i ++)
		{
			OptixProgramGroup& callbacksGroup = _callbacksGroups[i];
			SbtRecordEmpty& callableRecordHost = callableRecords[i];
			optixSbtRecordPackHeader(callbacksGroup, &callableRecordHost);
		}
		_sbt.callablesRecordBase = (CUdeviceptr)CudaMalloc(callableRecords);
		_sbt.callablesRecordStrideInBytes = (uint32_t)sizeof(SbtRecordEmpty);
		_sbt.callablesRecordCount = (uint32_t)callableRecords.size();
	}

	void OptixContext::BuildHitGroupSBT(const std::vector<HitGroupData>& datas)
	{
		CudaFree((void*)_sbt.hitgroupRecordBase);

		std::vector<SbtRecordHitGroup> hitgroupRecords(datas.size());
		for (size_t i = 0; i < datas.size(); ++i)
		{
			SbtRecordHitGroup& hitgroupRecordHost = hitgroupRecords[i];
			optixSbtRecordPackHeader(_hitProgramGroup, &hitgroupRecordHost);
			hitgroupRecordHost.data = datas[i];
		}
		_sbt.hitgroupRecordBase = (CUdeviceptr)CudaMalloc(hitgroupRecords);
		_sbt.hitgroupRecordStrideInBytes = (uint32_t)sizeof(SbtRecordHitGroup);
		_sbt.hitgroupRecordCount = (uint32_t)hitgroupRecords.size();
	}

	void OptixContext::_CheckTempBufferSize(size_t size)
	{
		if (_tempBuffer == nullptr)
		{
			_tempBufferSize = size;
			cudaMalloc(&_tempBuffer, _tempBufferSize);
		}
		else
		{
			if (size > _tempBufferSize)
			{
				_tempBufferSize = size;
				cudaFree(_tempBuffer);
				cudaMalloc(&_tempBuffer, _tempBufferSize);
			}
		}
	}
}
