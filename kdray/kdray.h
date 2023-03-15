#pragma once

#ifdef WIN32
#define KDRAY_API _declspec(dllexport)
#endif // WIN32

#ifdef __cplusplus
extern "C" {
#endif

	using kdrayLight = void*;
	using kdrayMesh = void*;
	using kdrayInstance = void*;

	enum kdrayAOV
	{
		KDRay_AOV_Beauty = 0,
		KDRay_AOV_Albedo = 1,
		KDRay_AOV_Normal = 2
	};
	enum kdraySampler
	{
		KDRay_Sampler_PCG = 0
	};
	enum kdrayFilter
	{
		KDRay_Filter_Tent = 0
	};
	enum kdrayMIS
	{
		KDRay_MIS_BSDF_LIGHT = 0
	};
	enum kdrayIntegrator
	{
		KDRay_Integrator_PathTracing = 0,
		KDRay_Integrator_AmbientOcclusion = 1
	};
	enum kdrayToneMap
	{
		KDRay_ToneMap_Clamp = 0,
		KDRay_ToneMap_Gamma = 1,
		KDRay_ToneMap_ACES = 2,
		KDRay_ToneMap_Uncharted = 3
	};
	struct kdrayRenderSetting
	{
		int aov = KDRay_AOV_Beauty;
		int sampler = KDRay_Sampler_PCG;
		int filter = KDRay_Filter_Tent;
		int mis = KDRay_MIS_BSDF_LIGHT;
		int integrator = KDRay_Integrator_PathTracing;
		int sampleCount = 2048;
		int maxBounce = 5;
		float intensityClamp = 30.0f;
		float exposure = 1.0f;
		int toneMap = KDRay_ToneMap_Gamma;
		bool denoise = true;
	};

	struct kdrayPerspectiveCamera
	{
		float fovY = 60.f;
		float nearZ = 0.1f;
		float farZ = 100.f;
		float fStop = 0.0f;
		float focalDistance = 1.0f;
	};

	KDRAY_API bool kdrayCreateRenderer();

	KDRAY_API void kdrayDestoryRenderer();

	KDRAY_API void kdraySetRenderSetting(const kdrayRenderSetting* setting);

	KDRAY_API void kdrayGetRenderSetting(kdrayRenderSetting* setting);

	KDRAY_API int kdrayGetSampleAccum();

	KDRAY_API void kdrayFramebufferSetSize(int width, int height);

	KDRAY_API void kdrayFramebufferGetSize(int* width, int* height);

	KDRAY_API bool kdrayFrameBufferSetOpenGLBuffer(unsigned glBuffer);

	KDRAY_API void kdrayFrameBufferSaveToFile(const char* filepath);

	KDRAY_API void kdrayCameraLookAt(float eyeX, float eyeY, float eyeZ, 
		float targetX, float targetY, float targetZ);

	KDRAY_API void kdrayCameraSetPerspect(const kdrayPerspectiveCamera* params);

	KDRAY_API void kdrayCameraGetPerspect(kdrayPerspectiveCamera* params);

	KDRAY_API void kdrayCameraSetTransformMartix(float martix[16]);

	KDRAY_API void kdrayCameraGetTransformMartix(float martix[16]);

	KDRAY_API void kdrayCameraSetProjectMartix(float martix[16]);

	KDRAY_API bool kdraySceneLoadFormFile(const char* filepath);

	KDRAY_API kdrayLight kdrayCreateDirectionalLight();

	KDRAY_API kdrayLight kdrayCreatePointLight(float radius);

	KDRAY_API void kdrayLightSetTransformMartix(kdrayLight light, float martix[16]);

	KDRAY_API kdrayMesh kdrayCreateMeshFromFile(const char* filepath);

	KDRAY_API kdrayInstance kdrayCreateInstance(kdrayMesh mesh);

	KDRAY_API void kdrayInstanceSetTransform(kdrayInstance instance, float transform[16]);

	//bool kdrayShapeSetMaterial()

	KDRAY_API void kdrayRenderOneFrame();

#ifdef __cplusplus
}
#endif
