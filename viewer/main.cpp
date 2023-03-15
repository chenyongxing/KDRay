#include <kdray.h>
#include <memory>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include <direct.h>
#include <nfd.h>

#include "gfx_graphics.h"
#include "time.h"
#include "input.h"
#include "camera.h"

using namespace std;
using namespace gfx;

GLFWwindow* glfwWindow = nullptr;
constexpr int leftPanelWidth = 250;
constexpr int viewWidthPadding = leftPanelWidth + 25;
constexpr int viewHeightPadding = 40;

constexpr size_t MAX_PATH = 512;
char SCENE_PATH[MAX_PATH];

std::unique_ptr<RenderBuffer> renderBuffer;
std::unique_ptr<Texture2D> renderTexture;
std::unique_ptr<Camera> camera;

kdrayRenderSetting renderSetting;
kdrayPerspectiveCamera cameraSetting;

void syncScene()
{
	int width, height;
	kdrayFramebufferGetSize(&width, &height);
	glfwSetWindowSize(glfwWindow, width + viewWidthPadding, height + viewHeightPadding);

	RenderBufferDesc bufferDesc;
	bufferDesc.type = ResourceType::PixelUnpackBuffer;
	bufferDesc.size = width * height * 4;
	renderBuffer = make_unique<RenderBuffer>();
	renderBuffer->Create(bufferDesc);

	TextureDesc texDesc;
	texDesc.type = ResourceType::RenderTarget;
	texDesc.format = RenderFormat::RGBA8;
	texDesc.width = width;
	texDesc.height = height;
	renderTexture = make_unique<Texture2D>();
	renderTexture->Create(texDesc);
	kdrayFrameBufferSetOpenGLBuffer(renderTexture->GetTextureID());

	kdrayCameraGetPerspect(&cameraSetting);
	float cameraTransform[16];
	kdrayCameraGetTransformMartix(cameraTransform);
	glm::mat4 cameraTransformGLM;
	memcpy(&cameraTransformGLM, cameraTransform, sizeof(float) * 16);
	camera->SetWorldTransform(cameraTransformGLM);

	kdrayGetRenderSetting(&renderSetting);
}

void setup()
{
	Time::Setup();
	kdrayCreateRenderer();
	camera = make_unique<Camera>();
	syncScene();
}

void render()
{
	Time::Update();
	float deltaTime = Time::DeltaTime();
	Input::Update();

	kdrayCameraSetPerspect(&cameraSetting);
	camera->Update(deltaTime);
	camera->freeCtrl.Update(deltaTime);
	float cameraTransform[16];
	memcpy(cameraTransform, &camera->WorldTransform(), sizeof(float) * 16);
	kdrayCameraSetTransformMartix(cameraTransform);

	kdraySetRenderSetting(&renderSetting);
	kdrayRenderOneFrame();

	Graphics::UnPackPBO(renderTexture.get(), renderBuffer.get());

	Input::LastUpdate();
}

void drawImgui()
{
	static ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;
	const ImGuiViewport* viewport = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(viewport->WorkPos);
	ImGui::SetNextWindowSize(viewport->WorkSize);
	static bool wndOpen = true;
	if (ImGui::Begin("Fullscreen", &wndOpen, flags))
	{
		ImGui::BeginChild("property", ImVec2(leftPanelWidth, 0), true);
		if (ImGui::Button("load scene"))
		{
			nfdchar_t* path = nullptr;
			if (NFD_OpenDialog(nullptr, SCENE_PATH, &path) == NFD_OKAY) {
				if (kdraySceneLoadFormFile(path))
				{
					syncScene();
					strcpy(SCENE_PATH, path);
				}
				free(path);
			}
		}
		ImGui::Separator();
		if (ImGui::Button("save screenshot"))
		{
			nfdchar_t* path = nullptr;
			if (NFD_SaveDialog("png", SCENE_PATH, &path) == NFD_OKAY) {
				kdrayFrameBufferSaveToFile(strcat(path, ".png"));
				free(path);
			}
		}
		ImGui::Separator();
		ImGui::PushItemWidth(120);
		const char* cameraItems[] = { "Perspective" }; //"Orthographic"
		static int cameratype = 0;
		ImGui::Combo("Camera Type", &cameratype, cameraItems, IM_ARRAYSIZE(cameraItems));
		//ImGui::InputFloat("Clip Near", &cameraSetting.nearZ);
		//ImGui::InputFloat("Clip Far", &cameraSetting.farZ);
		if (cameratype == 0)
		{
			ImGui::DragFloat("Fov Vertical", &cameraSetting.fovY, 1.0f, 1.0f, 140.0f, "%.1f");
			ImGui::DragFloat("F-Stop", &cameraSetting.fStop, 1.0f, 1.0f, 100.0f, "%.1f");
			ImGui::DragFloat("Focal Distance", &cameraSetting.focalDistance, 1.0f, 1.0f, 1000.0f, "%.1f");
		}
		ImGui::Separator();
		const char* samplerItems[] = { "PCG" };
		ImGui::Combo("Sampler", &renderSetting.sampler, samplerItems, IM_ARRAYSIZE(samplerItems));
		const char* filterItems[] = { "Tent" };
		ImGui::Combo("Filter", &renderSetting.filter, filterItems, IM_ARRAYSIZE(filterItems));
		const char* misItems[] = { "BSDF+LIGHT" };
		ImGui::Combo("MIS", &renderSetting.mis, misItems, IM_ARRAYSIZE(misItems));
		const char* integratorItems[] = { "Path", "AO" };
		ImGui::Combo("Integrator", &renderSetting.integrator, integratorItems, IM_ARRAYSIZE(integratorItems));
		ImGui::DragInt("Sample Count", &renderSetting.sampleCount, 256, 0, 8192, "%d%", ImGuiSliderFlags_AlwaysClamp);
		ImGui::DragInt("Max Bounce", &renderSetting.maxBounce, 1, 0, 100, "%d%", ImGuiSliderFlags_AlwaysClamp);
		ImGui::DragFloat("Intensity Clamp", &renderSetting.intensityClamp, 1.f, 1.0f, 100.0f, "%.1f");
		ImGui::Separator();
		const char* aovItems[] = { "Beauty", "Albedo", "Normal"};
		ImGui::Combo("AOV", &renderSetting.aov, aovItems, IM_ARRAYSIZE(aovItems));
		const char* toneMapItems[] = { "Clamp", "Gamma", "ACES", "Uncharted"};
		ImGui::DragFloat("Exposure", &renderSetting.exposure, 0.1f, 0.0f, 50.0f, "%.1f");
		ImGui::Combo("ToneMap", &renderSetting.toneMap, toneMapItems, IM_ARRAYSIZE(toneMapItems));
		ImGui::Checkbox("Denoise", &renderSetting.denoise);
		ImGui::PopItemWidth();
		ImGui::EndChild();
		ImGui::SameLine();

		ImGui::BeginChild("view", ImVec2(0, 0));
		ImGui::Text("%.1f FPS (%.3f ms)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
		ImGui::SameLine();
		ImGui::Text("	Sample Count (%d / %d)", kdrayGetSampleAccum(), renderSetting.sampleCount);
		ImGui::Separator();
		int width, height;
		kdrayFramebufferGetSize(&width, &height);
		ImGui::Image((void*)(intptr_t)renderTexture->GetTextureID(), 
			ImVec2((float)width, (float)height), ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));
		ImGui::EndChild();
	}
	ImGui::End();
}

void destory()
{
	kdrayDestoryRenderer();

	renderTexture.reset();
	renderBuffer.reset();
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (action == GLFW_PRESS)
	{
		Input::keydowns[(Input::Keycode)key] = true;
	}
	else if (action == GLFW_RELEASE)
	{
		Input::keydowns[(Input::Keycode)key] = false;
	}
}

void mousePosCallback(GLFWwindow* window, double xpos, double ypos)
{
	if (xpos > leftPanelWidth)
	{
		Input::mousePos = glm::vec2(xpos, ypos);
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);
	if (xpos > leftPanelWidth)
	{
		if (action == GLFW_PRESS)
		{
			Input::mouseButtondowns[(Input::MouseButton)button] = true;
		}
		else if (action == GLFW_RELEASE)
		{
			Input::mouseButtondowns[(Input::MouseButton)button] = false;
		}
	}
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);
	if (xpos > leftPanelWidth)
	{
		Input::scrollWheelDelta = (float)yoffset;
		Input::scrollWheelUpdated = true;
	}
}

int main()
{
	_getcwd(SCENE_PATH, MAX_PATH);
	if (!glfwInit())
	{
		return 1;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	glfwWindow = glfwCreateWindow(800, 600, "kdray viewer", nullptr, nullptr);
	if (glfwWindow == nullptr)
	{
		return 1;
	}
	glfwMakeContextCurrent(glfwWindow);
	glfwSwapInterval(true);
	glfwSetKeyCallback(glfwWindow, keyCallback);
	glfwSetCursorPosCallback(glfwWindow, mousePosCallback);
	glfwSetMouseButtonCallback(glfwWindow, mouseButtonCallback);
	glfwSetScrollCallback(glfwWindow, scrollCallback);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.Fonts->AddFontDefault();
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(glfwWindow, true);
	ImGui_ImplOpenGL3_Init(nullptr);

	setup();

	while (!glfwWindowShouldClose(glfwWindow))
	{
		glfwPollEvents();

		render();

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		drawImgui();
		ImGui::Render();
		int displayWidth, displayHeight;
		glfwGetFramebufferSize(glfwWindow, &displayWidth, &displayHeight);
		glViewport(0, 0, displayWidth, displayHeight);
		glClearColor(1.f, 1.f, 1.f, 1.f);
		glClear(GL_COLOR_BUFFER_BIT);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(glfwWindow);
	}

	destory();

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(glfwWindow);
	glfwTerminate();
	return 0;
}