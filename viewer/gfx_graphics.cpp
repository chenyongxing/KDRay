#include "gfx_graphics.h"
#include <sstream>
#include <fstream>
#include <spdlog/spdlog.h>
#include <GL/glew.h>

using namespace std;

namespace gfx
{
	RenderBuffer::~RenderBuffer()
	{
		if (m_created)
		{
			glDeleteBuffers(1, &bufferID);
			printf("RenderBuffer delete %d\n", bufferID);
		}
	}

	void RenderBuffer::Create(const RenderBufferDesc& desc)
	{
		this->desc = desc;
		int type = 0;
		switch (desc.type)
		{
		case ResourceType::PixelUnpackBuffer:
			type = GL_PIXEL_UNPACK_BUFFER;
			break;
		case ResourceType::VertexBuffer:
			type = GL_ARRAY_BUFFER;
			break;
		case ResourceType::IndexBuffer:
			type = GL_ELEMENT_ARRAY_BUFFER;
			break;
		case ResourceType::ConstantBuffer:
			type = GL_UNIFORM_BUFFER;
			break;
		default:
			throw std::runtime_error("");
			break;
		}

		glGenBuffers(1, &bufferID);
		glBindBuffer(type, bufferID);
		switch (desc.usage)
		{
		case ResourceUsage::Static:
			glBufferData(type, desc.size, desc.data, GL_STATIC_DRAW);
			break;
		case ResourceUsage::Dynamic:
			glBufferData(type, desc.size, desc.data, GL_DYNAMIC_DRAW);
			break;
		case ResourceUsage::StreamDraw:
			glBufferData(type, desc.size, desc.data, GL_STREAM_DRAW);
			break;
		default:
			throw std::runtime_error("");
			break;
		}

		glBindBuffer(type, 0);

		std::string typeStr;
		switch (desc.type)
		{
		case ResourceType::PixelUnpackBuffer:
			typeStr = "pixel unpack buffer";
			break;
		case ResourceType::VertexBuffer:
			typeStr = "vertex buffer";
			break;
		case ResourceType::IndexBuffer:
			typeStr = "index buffer";
			break;
		case ResourceType::ConstantBuffer:
			typeStr = "uniform buffer";
			break;
		default:
			throw std::runtime_error("");
			break;
		}

		spdlog::info("RenderBuffer new {0} : {1}", typeStr, bufferID);

		m_created = true;
	}

	void RenderBuffer::Release()
	{
		if (!m_created)
			return;

		glDeleteBuffers(1, &bufferID);
		m_created = false;
	}

	void RenderBuffer::Update(const void* data)
	{
		int type = 0;
		switch (desc.type)
		{
		case ResourceType::PixelUnpackBuffer:
			type = GL_PIXEL_UNPACK_BUFFER;
			break;
		case ResourceType::VertexBuffer:
			type = GL_ARRAY_BUFFER;
			break;
		case ResourceType::IndexBuffer:
			type = GL_ELEMENT_ARRAY_BUFFER;
			break;
		case ResourceType::ConstantBuffer:
			type = GL_UNIFORM_BUFFER;
			break;
		default:
			throw std::runtime_error("");
			break;
		}

		glBindBuffer(type, bufferID);
		glBufferSubData(type, 0, desc.size, data);
		glBindBuffer(type, 0);
	}

	void RenderBuffer::Resize(unsigned size)
	{
		if (!m_created)
			return;

		glDeleteBuffers(1, &bufferID);

		desc.size = size;
		Create(desc);
	}

	Texture2D::~Texture2D()
	{
		if (m_created)
			glDeleteTextures(1, &textureID);
	}

	void Texture2D::Create(const TextureDesc& desc)
	{
		this->desc = desc;

		bool ms = desc.sampleCount > 1;

		glGenTextures(1, &textureID);
		ms ? glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, textureID) : glBindTexture(GL_TEXTURE_2D, textureID);
		spdlog::info("Texture2D new texture2D: {}", textureID);

		if (desc.type == ResourceType::RenderTarget)
		{
			switch (desc.format)
			{
			case RenderFormat::RGBA8:
				if (ms)
				{
					glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, desc.sampleCount, GL_RGBA8, desc.width, desc.height, GL_FALSE);
				}
				else
				{
					glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, desc.width, desc.height);
				}
				break;
			case RenderFormat::RGBA32:
				if (ms)
				{
					glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, desc.sampleCount, GL_RGBA32F, desc.width, desc.height, GL_FALSE);
				}
				else
				{
					glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, desc.width, desc.height);
				}
				break;
			default:
				spdlog::warn("Texture2D no support RenderTarget texture format: {}", textureID);
				break;
			}
		}
		else if (desc.type == ResourceType::DepthStencil)
		{
			switch (desc.format)
			{
			case RenderFormat::D24:
				if (ms)
				{
					glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, desc.sampleCount, GL_DEPTH24_STENCIL8, desc.width, desc.height, GL_FALSE);
				}
				else
				{
					glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH24_STENCIL8, desc.width, desc.height);
				}
				break;
			default:
				spdlog::warn("Texture2D no support DepthStencil texture format: {}", textureID);
				break;
			}
		}
		else
		{
			switch (desc.format)
			{
			case RenderFormat::RGBA8:
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, desc.width, desc.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, desc.data);
				break;
			case RenderFormat::RGBA32:
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, desc.width, desc.height, 0, GL_RGBA, GL_FLOAT, desc.data);
				break;
			case RenderFormat::R32:
				glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, desc.width, desc.height, 0, GL_RED, GL_FLOAT, desc.data);
				break;
			case RenderFormat::DXT1:
			{
				//压缩纹理，每一个mipmap图片数据大小是当前图片1/4宽*高。8是像素块(blockBytes)
				double imageSize = desc.width * 0.25 * desc.height * 0.25 * 8;
				glCompressedTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, desc.width, desc.height, 0, (GLsizei)imageSize, desc.data);
				break;
			}
			case RenderFormat::DXT2:
			{
				double imageSize = desc.width * 0.25 * desc.height * 0.25 * 16;
				glCompressedTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, desc.width, desc.height, 0, (GLsizei)imageSize, desc.data);
				break;
			}
			default:
				spdlog::warn("Texture2D no support texture format: {}", textureID);
				break;
			}
		}

		ms ? glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0) : glBindTexture(GL_TEXTURE_2D, 0);

		m_created = true;
	}

	void Texture2D::Release()
	{
		if (!m_created)
			return;

		glDeleteTextures(1, &textureID);
		m_created = false;
	}

	void Texture2D::Update(const void* data)
	{
	}

	void Texture2D::Resize(unsigned width, unsigned height)
	{
		if (!m_created)
			return;

		glDeleteTextures(1, &textureID);

		desc.width = width;
		desc.height = height;
		Create(desc);
	}

	SamplerState::~SamplerState()
	{
		glDeleteSamplers(1, &sampler);
	}

	void SamplerState::Create(const SamplerDesc& desc)
	{
		this->desc = desc;

		glGenSamplers(1, &sampler);

		switch (desc.addressMode)
		{
		case TextureAddressMode::Wrap:
			glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_REPEAT);
			break;
		case TextureAddressMode::Mirror:
			glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
			glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
			break;
		case TextureAddressMode::Clamp:
			glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			break;
		case TextureAddressMode::Border:
			glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
			glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
			break;
		default:
			break;
		}

		switch (desc.filter)
		{
		case TextureFilter::Point:
			glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			break;
		case TextureFilter::Linear:
			glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			break;
		case TextureFilter::LinearMip:
			glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			break;
		case TextureFilter::Anisotropic:
			//glSamplerParameterf(sampler, GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, desc.maxAnisotropy);

			spdlog::info("SamplerState no support anisotropic sampler: {}", sampler);
			break;
		default:
			break;
		}

		glSamplerParameterfv(sampler, GL_TEXTURE_BORDER_COLOR, desc.borderColor);

		spdlog::info("SamplerState new sampler: {}", sampler);
	}

	RenderTarget::~RenderTarget()
	{
		if (m_depth_stencil_texture)
		{
			delete m_depth_stencil_texture;
		}

		for (auto color_texture : m_color_textures)
		{
			if (color_texture)
			{
				delete color_texture;
			}
		}

		glDeleteFramebuffers(1, &m_fbo);
	}

	void RenderTarget::Create(std::vector<Texture2D*>& colorTextures, Texture2D* depthStencilTexture)
	{
		if (colorTextures.size() == 0)
			return;

		m_color_textures = colorTextures;
		m_depth_stencil_texture = depthStencilTexture;

		m_width = colorTextures[0]->GetDesc().width;
		m_height = colorTextures[0]->GetDesc().height;

		glGenFramebuffers(1, &m_fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
		spdlog::info("RenderTarget new framebuffer: {}", m_fbo);

		std::vector<unsigned> attachments;
		for (unsigned i = 0; i < m_color_textures.size(); i++)
		{
			auto GLTexture = colorTextures[i]->GetDesc().sampleCount > 1 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D;
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GLTexture, m_color_textures[i]->GetTextureID(), 0);
			attachments.push_back(GL_COLOR_ATTACHMENT0 + i);

			spdlog::info("RenderTarget fbo attach color texture: {}", m_color_textures[i]->GetTextureID());
		}
		//必须调用，默认shader只会渲染到第一个
		glDrawBuffers((int)m_color_textures.size(), attachments.data());

		//深度模板缓冲区
		if (m_depth_stencil_texture)
		{
			auto GLTexture = m_depth_stencil_texture->GetDesc().sampleCount > 1 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D;
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GLTexture, m_depth_stencil_texture->GetTextureID(), 0);
			spdlog::info("RenderTarget fbo attach depth24 stencil8 texture: {}", m_depth_stencil_texture->GetTextureID());
		}

		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		{
			spdlog::error("RenderTarget framebuffer is not complete: {0:x}", glCheckFramebufferStatus(GL_FRAMEBUFFER));
			//std::throw_error("framebuffer is not complete");
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		m_created = true;
	}

	void RenderTarget::Create(std::vector<Texture2D*>&& colorTextures, Texture2D* depthStencilTexture)
	{
		Create(colorTextures, depthStencilTexture);
	}

	void RenderTarget::Resize(unsigned width, unsigned height)
	{
		if (!m_created) return;

		glDeleteFramebuffers(1, &m_fbo);

		if (m_depth_stencil_texture)
		{
			m_depth_stencil_texture->Resize(width, height);
		}

		for (auto color_texture : m_color_textures)
		{
			if (color_texture)
			{
				color_texture->Resize(width, height);
			}
		}

		Create(m_color_textures, m_depth_stencil_texture);
	}

	void RenderTarget::Release(bool releaseTex)
	{
		if (!m_created) return;

		if (releaseTex)
		{
			if (m_depth_stencil_texture)
			{
				delete m_depth_stencil_texture;
				m_depth_stencil_texture = nullptr;
			}

			for (auto color_texture : m_color_textures)
			{
				if (color_texture)
				{
					delete color_texture;
				}
			}
			m_color_textures.clear();
		}

		glDeleteFramebuffers(1, &m_fbo);
		spdlog::info("RenderTarget delete framebuffer: {}", m_fbo);
	}

	RasterShader::~RasterShader()
	{
		glDeleteProgram(this->shaderProgram);
	}

	bool RasterShader::LoadFromFile(const char* vs, const char* ps)
	{
		ifstream vsFile;
		stringstream vsStream;
		vsFile.open(vs);
		if (!vsFile.good())
		{
			spdlog::error("Shader canot find vs file: {}", vs);
			return false;
		}
		vsStream << vsFile.rdbuf();
		vsFile.close();

		ifstream psFile;
		stringstream psStream;
		psFile.open(ps);
		if (!psFile.good())
		{
			spdlog::error("Shader canot find ps file: {}", ps);
			return false;
		}
		psStream << psFile.rdbuf();
		psFile.close();

		return LoadFromString(vsStream.str(), psStream.str());
	}

	bool RasterShader::LoadFromString(const std::string& vsStr, const std::string& psStr)
	{
		auto vs = glCreateShader(GL_VERTEX_SHADER);
		auto vs_str = vsStr.c_str();
		glShaderSource(vs, 1, &vs_str, nullptr);
		glCompileShader(vs);

		auto fs = glCreateShader(GL_FRAGMENT_SHADER);
		auto fs_str = psStr.c_str();
		glShaderSource(fs, 1, &fs_str, nullptr);
		glCompileShader(fs);

		this->shaderProgram = glCreateProgram();
		glAttachShader(this->shaderProgram, vs);
		glAttachShader(this->shaderProgram, fs);
		glLinkProgram(this->shaderProgram);

		//获取链接错误消息
		int success = 1;
		glGetProgramiv(this->shaderProgram, GL_LINK_STATUS, &success);
		if (success != GL_TRUE)
		{
			int length = 0;
			glGetProgramiv(this->shaderProgram, GL_INFO_LOG_LENGTH, &length);
			auto log = new char[length];
			glGetProgramInfoLog(this->shaderProgram, length, NULL, log);
			spdlog::error("Shader compile error: {}", log);
			delete[] log;

			return false;
		}

		glDeleteShader(vs);
		glDeleteShader(fs);

		spdlog::info("Shader new shader program: {}", this->shaderProgram);

		return true;
	}

	void RasterShader::GenerateInputLayout(const std::initializer_list<RasterInputElementDesc>& descs)
	{
		for (const RasterInputElementDesc& desc : descs)
		{
			GLInputLayoutElement glElement;
			glElement.format = desc.format;
			glElement.name = desc.name;
			glElement.offset = desc.offset;
			glElement.location = glGetAttribLocation(shaderProgram, desc.name);
			if (glElement.location < 0)
			{
				spdlog::error("Shader shader {0} canot get location {1}", this->shaderProgram, desc.name);
			}
			inputLayout.push_back(glElement);
		}
	}

	void RasterShader::SetTextureUnit(const char* name, unsigned unit)
	{
		glUseProgram(shaderProgram);
		GLint location = glGetUniformLocation(shaderProgram, name);
		glUniform1i(location, unit);
	}

	void RasterShader::SetUniformBufferIndex(const char* name, unsigned slot)
	{
		unsigned index = glGetUniformBlockIndex(shaderProgram, name);
		glUniformBlockBinding(shaderProgram, index, slot);
	}

	unsigned RasterShader::CalculateVertexCount(RenderBuffer* buffer)
	{
		int stride = 0;
		for (const auto& element : inputLayout)
		{
			switch (element.format)
			{
			case RenderFormat::RG32:
				stride += 2 * sizeof(float);
				break;
			case RenderFormat::RGB32:
				stride += 3 * sizeof(float);
				break;
			case RenderFormat::RGBA32:
				stride += 4 * sizeof(float);
				break;
			}
		}
		return buffer->GetDesc().size / stride;
	}

	void GLAPIENTRY MessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, 
		GLsizei length, const GLchar* message, const void* userParam)
	{
		if (type == GL_DEBUG_TYPE_ERROR)
		{
			spdlog::error("OpenGL error message: {}", message);
		}
	}

	namespace Graphics
	{
		static RasterState* cacheRasterState = nullptr;

		static int cacheFBO = -1;

		static RenderTarget* currentRenderTarget = nullptr;

		void EnableDebug()
		{
#ifndef NDEBUG
			glEnable(GL_DEBUG_OUTPUT);
			glDebugMessageCallback(MessageCallback, 0);
#endif
		}

		void GenerateMips(const Texture2D* texture)
		{
			glBindTexture(GL_TEXTURE_2D, texture->GetTextureID());
			glGenerateMipmap(GL_TEXTURE_2D);
			//glTexParameteri(textureID, GL_TEXTURE_BASE_LEVEL, 0);
			//glTexParameteri(textureID, GL_TEXTURE_MAX_LEVEL, 8);
			glBindTexture(GL_TEXTURE_2D, 0);
		}

		void ResolveMSAA(RenderTarget* dst, const RenderTarget* src)
		{
			glBindFramebuffer(GL_READ_FRAMEBUFFER, dst->GetFboID());
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, src->GetFboID());
			glBlitFramebuffer(0, 0, src->GetWidth(), src->GetHeight(), 0, 0, dst->GetWidth(), dst->GetHeight(), GL_COLOR_BUFFER_BIT, GL_NEAREST);
		}

		unsigned int GetCurrentTextureID()
		{
			GLint currentTexID = -1;
			glGetIntegerv(GL_TEXTURE_BINDING_2D, &currentTexID);
			return currentTexID;
		}

		void CopyTexture2D(Texture2D* dst, const Texture2D* src)
		{
			if (cacheFBO < 0)
			{
				GLuint fboId = 0;
				glGenFramebuffers(1, &fboId);
				cacheFBO = fboId;
			}
			glBindFramebuffer(GL_FRAMEBUFFER, cacheFBO);
			glReadBuffer(GL_COLOR_ATTACHMENT0);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, src->GetTextureID(), 0);

			int format = GL_RGBA8;
			switch (dst->GetDesc().format)
			{
			case RenderFormat::R32:
				format = GL_R32F;
				break;
			case RenderFormat::RGBA8:
				format = GL_RGBA8;
				break;
			case RenderFormat::RGBA32:
				format = GL_RGBA32F;
				break;
			default:
				throw std::runtime_error("");
				break;
			}

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, dst->GetTextureID());
			glCopyTexImage2D(GL_TEXTURE_2D, 0, format, 0, 0, dst->GetDesc().width, dst->GetDesc().height, 0);
			glBindTexture(GL_TEXTURE_2D, 0);
		}

		void UnPackPBO(const Texture2D* texture, const RenderBuffer* pbo)
		{
			glBindTexture(GL_TEXTURE_2D, texture->GetTextureID());

			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo->GetBufferID());

			auto desc = texture->GetDesc();
			switch (desc.format)
			{
			case RenderFormat::R32:
				glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
				glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, desc.width, desc.height, GL_RED, GL_FLOAT, nullptr);
				break;
			case RenderFormat::RGBA8:
				glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
				glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, desc.width, desc.height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
				break;
			case RenderFormat::RGBA32:
				glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
				glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, desc.width, desc.height, GL_RGBA, GL_FLOAT, nullptr);
				break;
			default:
				throw std::runtime_error("");
				break;
			}

			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		}

		RenderTarget* GetRenderTarget()
		{
			return currentRenderTarget;
		}

		void SetRenderTarget(RenderTarget* renderTarget)
		{
			currentRenderTarget = renderTarget;
			if (renderTarget)
			{
				glBindFramebuffer(GL_FRAMEBUFFER, renderTarget->GetFboID());
			}
			else
			{
				glBindFramebuffer(GL_FRAMEBUFFER, 0);
			}
		}

		void ClearColorBuffer(const RenderTarget* renderTarget, const float color[4])
		{
			if (currentRenderTarget != renderTarget)
			{
				glBindFramebuffer(GL_FRAMEBUFFER, renderTarget->GetFboID());
			}

			if (color == nullptr)
			{
				glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
				glClear(GL_COLOR_BUFFER_BIT);
			}
			else
			{
				glClearColor(color[0], color[1], color[2], color[3]);
				glClear(GL_COLOR_BUFFER_BIT);
			}
		}

		void ClearColorBuffer(const RenderTarget* renderTarget, std::initializer_list<float>&& color)
		{
			if (currentRenderTarget != renderTarget)
			{
				glBindFramebuffer(GL_FRAMEBUFFER, renderTarget->GetFboID());
			}

			float _color[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
			size_t count = 0;
			for (auto& c : color)
				_color[count++] = c;

			glClearColor(_color[0], _color[1], _color[2], _color[3]);
			glClear(GL_COLOR_BUFFER_BIT);
		}

		void ClearDepth(const RenderTarget* renderTarget, float depth)
		{
			if (currentRenderTarget != renderTarget)
			{
				glBindFramebuffer(GL_FRAMEBUFFER, renderTarget->GetFboID());
			}

			//防止触发early-z
			glDepthMask(GL_TRUE);

			glClearDepth(depth);
			glClear(GL_DEPTH_BUFFER_BIT);
		}

		void ClearStencil(const RenderTarget* renderTarget, unsigned char stencil)
		{
			if (currentRenderTarget != renderTarget)
			{
				glBindFramebuffer(GL_FRAMEBUFFER, renderTarget->GetFboID());
			}

			glClearStencil(stencil);
			glClear(GL_STENCIL_BUFFER_BIT);
		}

		void SetRasterState(const RasterState* state)
		{
			//if (cacheRasterState && (*cacheRasterState == *state))
			//	return;

			glViewport(state->viewport[0], state->viewport[1], state->viewport[2], state->viewport[3]);

			switch (state->cullMode)
			{
			case CullMode::ClockwiseFace:
			{
				glEnable(GL_CULL_FACE);
				glFrontFace(GL_CW);
				glCullFace(GL_BACK);
				break;
			}
			case CullMode::CounterClockwiseFace:
			{
				glEnable(GL_CULL_FACE);
				glFrontFace(GL_CCW);
				glCullFace(GL_BACK);
				break;
			}
			default:
				glDisable(GL_CULL_FACE);
				break;
			}

			switch (state->depthTest)
			{
			case CompareFunction::Always:
				glDisable(GL_DEPTH_TEST);
				break;
			case CompareFunction::Less:
			{
				glEnable(GL_DEPTH_TEST);
				glDepthFunc(GL_LESS);
				break;
			}
			case CompareFunction::Greater:
			{
				glEnable(GL_DEPTH_TEST);
				glDepthFunc(GL_GREATER);
				break;
			}
			default:
				glDisable(GL_DEPTH_TEST);
				break;
			}

			switch (state->fillMode)
			{
			case FillMode::Point:
			{
				glPolygonMode(GL_FRONT, GL_POINT);
				glPolygonMode(GL_BACK, GL_POINT);
				break;
			}
			case FillMode::WireFrame:
			{
				glPolygonMode(GL_FRONT, GL_LINE);
				glPolygonMode(GL_BACK, GL_LINE);
				break;
			}
			case FillMode::Solid:
			{
				glPolygonMode(GL_FRONT, GL_FILL);
				glPolygonMode(GL_BACK, GL_FILL);
				break;
			}
			}

			switch (state->blend)
			{
			case Blend::Add_SrcAlpha_OneMinusSrcAlpha:
			{
				glEnable(GL_BLEND);
				glBlendEquation(GL_FUNC_ADD);
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
				break;
			}
			default:
				glDisable(GL_BLEND);
				break;
			}

			if (!state->scissorTest)
			{
				glDisable(GL_SCISSOR_TEST);
			}

			cacheRasterState = const_cast<RasterState*>(state);
		}

		void SetVertexBuffer(const RenderBuffer* renderBuffer)
		{
			glBindBuffer(GL_ARRAY_BUFFER, renderBuffer->GetBufferID());
		}

		void SetIndexBuffer(const RenderBuffer* renderBuffer)
		{
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, renderBuffer->GetBufferID());
		}

		void SetRasterShader(const RasterShader* shader)
		{
			auto inputLayout = shader->GetInputLayout();

			if (inputLayout.size() == 0)
			{
				spdlog::error("Graphic.SetRasterShader inputLayout size = 0. program: {}", shader->GetShaderProgram());
				return;
			}

			glUseProgram(shader->GetShaderProgram());

			//InputLayout
			int stride = 0;
			for (const auto& element : shader->GetInputLayout())
			{
				switch (element.format)
				{
				case RenderFormat::RG32:
					stride += 2 * sizeof(float);
					break;
				case RenderFormat::RGB32:
					stride += 3 * sizeof(float);
					break;
				case RenderFormat::RGBA32:
					stride += 4 * sizeof(float);
					break;
				}
			}

			for (const auto& element : shader->GetInputLayout())
			{
				switch (element.format)
				{
				case RenderFormat::RG32:
					glEnableVertexAttribArray(element.location);
					glVertexAttribPointer(element.location, 2, GL_FLOAT, GL_FALSE, stride, (void*)(size_t)(element.offset));
					break;
				case RenderFormat::RGB32:
					glEnableVertexAttribArray(element.location);
					glVertexAttribPointer(element.location, 3, GL_FLOAT, GL_FALSE, stride, (void*)(size_t)(element.offset));
					break;
				case RenderFormat::RGBA32:
					glEnableVertexAttribArray(element.location);
					glVertexAttribPointer(element.location, 4, GL_FLOAT, GL_FALSE, stride, (void*)(size_t)(element.offset));
					break;
				}
			}
		}

		void SetVSUniformBuffer(unsigned slot, const RenderBuffer* renderBuffer)
		{
			glBindBufferBase(GL_UNIFORM_BUFFER, slot, renderBuffer->GetBufferID());
		}

		void SetPSTexture2D(unsigned slot, const Texture2D* texture)
		{
			glActiveTexture(GL_TEXTURE0 + slot);
			glBindTexture(GL_TEXTURE_2D, texture->GetTextureID());
		}

		void SetPSSampler(unsigned slot, const SamplerState* sampler)
		{
			glBindSampler(slot, sampler->GetSamplerID());
		}

		void Draw(PrimitiveType primitiveType, unsigned count, unsigned start)
		{
			switch (primitiveType)
			{
			case PrimitiveType::LineList:
				glDrawArrays(GL_LINES, start, count);
				break;
			case PrimitiveType::LineStrip:
				glDrawArrays(GL_LINE_STRIP, start, count);
				break;
			case PrimitiveType::TriangleList:
				glDrawArrays(GL_TRIANGLES, start, count);
				break;
			case PrimitiveType::TriangleStrip:
				glDrawArrays(GL_TRIANGLE_STRIP, start, count);
				break;
			}
		}

		void DrawIndexed(PrimitiveType primitiveType, unsigned count)
		{
			switch (primitiveType)
			{
			case PrimitiveType::LineList:
				glDrawElements(GL_LINES, count, GL_UNSIGNED_INT, 0);
				break;
			case PrimitiveType::LineStrip:
				glDrawElements(GL_LINE_STRIP, count, GL_UNSIGNED_INT, 0);
				break;
			case PrimitiveType::TriangleList:
				glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, 0);
				break;
			case PrimitiveType::TriangleStrip:
				glDrawElements(GL_TRIANGLE_STRIP, count, GL_UNSIGNED_INT, 0);
				break;
			}
		}
	}
}
