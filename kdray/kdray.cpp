#include "kdRay.h"
#include "renderer.h"
#include "scene.h"

using namespace kdray;

static inline bool MartixEqual(const float m0[16], const float m1[16])
{
	return FloatEqual(m0[0], m1[0]) && FloatEqual(m0[1], m1[1]) && FloatEqual(m0[2], m1[2]) && FloatEqual(m0[3], m1[3]) &&
		FloatEqual(m0[4], m1[4]) && FloatEqual(m0[5], m1[5]) && FloatEqual(m0[6], m1[6]) && FloatEqual(m0[7], m1[7]) &&
		FloatEqual(m0[8], m1[8]) && FloatEqual(m0[9], m1[9]) && FloatEqual(m0[10], m1[10]) && FloatEqual(m0[11], m1[11]) &&
		FloatEqual(m0[12], m1[12]) && FloatEqual(m0[13], m1[13]) && FloatEqual(m0[14], m1[14]) && FloatEqual(m0[15], m1[15]);
}

bool kdrayCreateRenderer()
{
	return Renderer::Instance()->Init();
}

void kdrayDestoryRenderer()
{
	Renderer::Instance()->Destory();
}

void kdraySetRenderSetting(const kdrayRenderSetting* setting)
{
	RenderSetting rs;
	rs.aov = (AOVType)setting->aov;
	rs.integrator = (IntegratorType)setting->integrator;
	rs.sampleCount = setting->sampleCount;
	rs.maxBounce = setting->maxBounce;
	rs.intensityClamp = setting->intensityClamp;
	rs.exposure = setting->exposure;
	rs.toneMap = (ToneMapType)setting->toneMap;
	rs.denoise = setting->denoise;
	Renderer::Instance()->SetRenderSetting(rs);
}

void kdrayGetRenderSetting(kdrayRenderSetting* setting)
{
	RenderSetting& rs = Renderer::Instance()->GetRenderSetting();
	setting->aov = (int)rs.aov;
	setting->integrator = (int)rs.integrator;
	setting->sampleCount = rs.sampleCount;
	setting->maxBounce = rs.maxBounce;
	setting->intensityClamp = rs.intensityClamp;
	setting->exposure = rs.exposure;
	setting->toneMap = (int)rs.toneMap;
	setting->denoise = rs.denoise;
}

int kdrayGetSampleAccum()
{
	return Renderer::Instance()->GetAccumCount();
}

void kdrayFramebufferSetSize(int width, int height)
{
	auto framebuffer = Renderer::Instance()->GetFramebuffer();
	if (framebuffer)
	{
		framebuffer->Resize(width, height);
	}
}

void kdrayFramebufferGetSize(int* width, int* height)
{
	auto framebuffer = Renderer::Instance()->GetFramebuffer();
	if (framebuffer)
	{
		*width = framebuffer->GetWidth();
		*height = framebuffer->GetHeight();
	}
}

bool kdrayFrameBufferSetOpenGLBuffer(unsigned glBuffer)
{
	auto framebuffer = Renderer::Instance()->GetFramebuffer();
	if (framebuffer)
	{
		return framebuffer->SetOpenGLBuffer(glBuffer);
	}
	return false;
}

void kdrayFrameBufferSaveToFile(const char* filepath)
{
	auto framebuffer = Renderer::Instance()->GetFramebuffer();
	if (framebuffer)
	{
		framebuffer->SaveToFile(filepath);
	}
}

void kdrayCameraLookAt(float eyeX, float eyeY, float eyeZ,
	float targetX, float targetY, float targetZ)
{
	auto camera = Renderer::Instance()->GetCamera();
	if (camera)
	{
		camera->LookAt(make_float3(eyeX, eyeY, eyeZ), 
			make_float3(targetX, targetY, targetZ),
			make_float3(0.0f, 0.0f, 1.0f));
	}
}

void kdrayCameraSetPerspect(const kdrayPerspectiveCamera* params)
{
	auto framebuffer = Renderer::Instance()->GetFramebuffer();
	auto camera = Renderer::Instance()->GetCamera();
	if (framebuffer && camera)
	{
		bool changed = false;
		changed |= !FloatEqual(params->nearZ, camera->GetNearZ());
		changed |= !FloatEqual(params->farZ, camera->GetFarZ());
		changed |= !FloatEqual(params->fovY, camera->GetFovY());
		changed |= !FloatEqual(params->fStop, camera->fStop);
		changed |= !FloatEqual(params->focalDistance, camera->focalDistance);
		if (changed) 
		{
			camera->fStop = params->fStop;
			camera->focalDistance = params->focalDistance;
			float aspect = (float)framebuffer->GetWidth() / (float)framebuffer->GetHeight();
			camera->Perspect(params->fovY, aspect, params->nearZ, params->farZ);
			Renderer::Instance()->ResetAccumCount();
		}
	}
}

void kdrayCameraGetPerspect(kdrayPerspectiveCamera* params)
{
	auto camera = Renderer::Instance()->GetCamera();
	if (camera)
	{
		params->nearZ = camera->GetNearZ();
		params->farZ = camera->GetFarZ();
		params->fovY = camera->GetFovY();
	}
}

void kdrayCameraSetTransformMartix(float martix[16])
{
	auto camera = Renderer::Instance()->GetCamera();
	if (camera)
	{
		float preMartix[16];
		camera->GetTransformMatrix(preMartix);
		if (!MartixEqual(preMartix, martix))
		{
			camera->SetTransformMartix(martix);
			Renderer::Instance()->ResetAccumCount();
		}
	}
}

void kdrayCameraGetTransformMartix(float martix[16])
{
	auto camera = Renderer::Instance()->GetCamera();
	if (camera)
	{
		camera->GetTransformMatrix(martix);
	}
}

void kdrayCameraSetProjectMartix(float martix[16])
{
	auto camera = Renderer::Instance()->GetCamera();
	if (camera)
	{
		camera->SetProjectMartix(martix);
	}
}

bool kdraySceneLoadFormFile(const char* filepath)
{
	auto scene = Renderer::Instance()->GetScene();
	if (scene)
	{
		return scene->LoadFromFile(filepath);
	}
	return false;
}

kdrayLight kdrayCreateDirectionalLight()
{
	auto scene = Renderer::Instance()->GetScene();
	if (scene)
	{
		Light* light = new Light();
		light->type = LightType::Directional;
		scene->AddLight(light);
		return light;
	}
	return nullptr;
}

kdrayLight kdrayCreatePointLight(float radius)
{
	auto scene = Renderer::Instance()->GetScene();
	if (scene)
	{
		Light* light = new Light();
		light->type = LightType::Point;
		light->radius = radius;
		scene->AddLight(light);
		return light;
	}
	return nullptr;
}

void kdrayLightSetTransformMartix(kdrayLight light, float martix[16])
{
	auto scene = Renderer::Instance()->GetScene();
	if (scene)
	{
		scene->SetLightTransform(reinterpret_cast<Light*>(light), martix);
	}
}

kdrayMesh kdrayCreateMeshFromFile(const char* filepath)
{
	auto scene = Renderer::Instance()->GetScene();
	if (scene)
	{
		Mesh* mesh = new Mesh();
		mesh->LoadFromFile(filepath);
		scene->AddMesh(mesh);
		return mesh;
	}
	return nullptr;
}

kdrayInstance kdrayCreateInstance(kdrayMesh mesh)
{
	auto scene = Renderer::Instance()->GetScene();
	if (scene)
	{
		Instance* instance = new Instance();
		instance->mesh = reinterpret_cast<Mesh*>(mesh);
		scene->AddInstance(instance);
		return instance;
	}
	return nullptr;
}

void kdrayInstanceSetTransform(kdrayInstance instance, float transform[16])
{
	auto scene = Renderer::Instance()->GetScene();
	{
		scene->SetInstanceTransform(reinterpret_cast<Instance*>(instance), transform);
	}
}

void kdrayRenderOneFrame()
{
	Renderer::Instance()->RenderOneFrame();
}
