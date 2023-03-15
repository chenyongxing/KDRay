#pragma once

#include <vector>
#include <string>

namespace gfx
{
	enum struct RenderFormat
	{
		RGBA32 = 2,
		RGB32 = 6,
		RGBA16 = 10,
		RG32 = 16,
		RGBA8 = 28,
		R32 = 39,
		D32 = 40,
		D24 = 45,
		D16 = 55,
		BGRA8 = 87,
		DXT1,
		DXT2,
		DXT3,
		DXT4,
		DXT5
	};

	enum struct ShaderType
	{
		Vertex = 0x1,
		Hull = 0x2,
		Domain = 0x4,
		Geometry = 0x8,
		Pixel = 0x10,
		Compute = 0x20
	};

	enum struct PrimitiveType
	{
		PointList = 1,
		LineList = 2,
		LineStrip = 3,
		TriangleList = 4,
		TriangleStrip = 5,
		TriangleFan = 6
	};

	enum struct CompareFunction
	{
		Never = 1,
		Less = 2,
		Equal = 3,
		LessEqual = 4,
		Greater = 5,
		NotEqual = 6,
		GreaterEqual = 7,
		Always = 8
	};

	enum struct CullMode
	{
		None,
		ClockwiseFace,
		CounterClockwiseFace
	};

	enum struct FillMode
	{
		Point = 1,
		WireFrame = 2,
		Solid = 3,
	};

	enum struct Blend
	{
		None,
		Add_SrcAlpha_OneMinusSrcAlpha
	};

	enum struct ResourceType
	{
		PixelUnpackBuffer = 0x0,
		VertexBuffer = 0x1,
		IndexBuffer = 0x2,
		ConstantBuffer = 0x4,
		ShaderResource = 0x8,
		RenderTarget = 0x20,
		DepthStencil = 0x40,
		UnorderedAccess = 0x80,
	};

	enum ResourceFlag
	{
		Default = 0x0,
		GenerateMips = 0x1,
		TextureCube = 0x4,
		StructuredBuffer = 0x40,
	};

	enum ResourceUsage
	{
		Static = 1,
		Dynamic = 2,
		Staging = 3,
		StreamDraw = 4
	};

	struct RasterInputElementDesc
	{
		const char* name = nullptr;
		unsigned offset = 0;
		RenderFormat format = RenderFormat::RGB32;
	};

	class RenderResourceBase
	{
	public:
		RenderResourceBase() {}
		RenderResourceBase(const RenderResourceBase&) = delete;
		RenderResourceBase& operator=(const RenderResourceBase&) = delete;

		virtual void Release() = 0;

		inline bool Created()
		{
			return m_created;
		}

	protected:
		bool m_created = false;
	};

	struct RenderBufferDesc
	{
		ResourceType type = ResourceType::ConstantBuffer;
		const void* data = nullptr;
		unsigned size = 0;
		ResourceUsage usage = ResourceUsage::Static;
	};
	class RenderBuffer final : public RenderResourceBase
	{
	public:
		~RenderBuffer();

		void Create(const RenderBufferDesc& desc);

		virtual void Release() override;

		void Update(const void* data);

		void Resize(unsigned size);

		inline unsigned GetBufferID() const
		{
			return bufferID;
		}

		inline RenderBufferDesc& GetDesc()
		{
			return desc;
		}

	protected:
		RenderBufferDesc desc;

		unsigned bufferID = 0;
	};

	struct TextureDesc
	{
		ResourceType type = ResourceType::ShaderResource;
		const void* data = nullptr;
		unsigned width = 0;
		unsigned height = 0;
		unsigned sampleCount = 1;
		RenderFormat format = RenderFormat::RGBA8;
	};
	class Texture2D final : public RenderResourceBase
	{
	public:
		~Texture2D();

		void Create(const TextureDesc& desc);

		virtual void Release() override;

		void Update(const void* data);

		void Resize(unsigned width, unsigned height);

		inline unsigned GetTextureID() const
		{
			return textureID;
		}

		inline TextureDesc GetDesc() const
		{
			return desc;
		}

		inline TextureDesc& GetDesc()
		{
			return desc;
		}

	private:
		TextureDesc desc;
		unsigned textureID = 0;
	};

	enum struct TextureAddressMode
	{
		Wrap = 1,
		Mirror = 2,
		Clamp = 3,
		Border = 4
	};

	enum struct TextureFilter
	{
		Point = 0,
		Linear = 0x15,
		Anisotropic = 0x55,
		LinearMip
	};

	struct SamplerDesc
	{
		TextureFilter filter = TextureFilter::Linear;
		TextureAddressMode addressMode = TextureAddressMode::Wrap;
		unsigned maxAnisotropy = 8;
		float borderColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	};

	class SamplerState final
	{
	public:
		~SamplerState();

		void Create(const SamplerDesc& desc);

		inline unsigned GetSamplerID() const
		{
			return sampler;
		}

	private:
		SamplerDesc desc;
		unsigned sampler = 0;
	};

	class RenderTarget final
	{
	public:
		~RenderTarget();

		void Create(std::vector<Texture2D*>& colorTextures, Texture2D* depthStencilTexture = nullptr);

		void Create(std::vector<Texture2D*>&& colorTextures, Texture2D* depthStencilTexture = nullptr);

		void Release(bool releaseTex = true);

		void Resize(unsigned width, unsigned height);

		inline bool Created()
		{
			return m_created;
		}

		inline void SetFboID(unsigned fbo)
		{
			m_fbo = fbo;
		}

		inline unsigned GetFboID() const
		{
			return m_fbo;
		}

		inline unsigned GetWidth() const
		{
			return m_width;
		}

		inline unsigned GetHeight() const
		{
			return m_height;
		}

		inline Texture2D* GetColorTexture(size_t i)
		{
			return m_color_textures[i];
		}

		inline Texture2D* GetDepthStencilTexture()
		{
			return m_depth_stencil_texture;
		}

	private:
		bool m_created = false;

		std::vector<Texture2D*> m_color_textures;
		Texture2D* m_depth_stencil_texture = nullptr;

		unsigned m_fbo = 0;

		unsigned m_width = 0;
		unsigned m_height = 0;
	};

	class RasterState
	{
	public:
		int viewport[4] = { 0, 0, 0, 0 };
		FillMode fillMode = FillMode::Solid;
		CullMode cullMode = CullMode::None;
		CompareFunction depthTest = CompareFunction::Always;
		Blend blend = Blend::None;
		bool scissorTest = false;
		int scissorBox[4] = { 0, 0, 0, 0 };
	};

	inline bool operator==(const RasterState& a, const RasterState& b)
	{
		return
			(std::abs(a.viewport[0] - b.viewport[0]) < 0.01) &&
			(std::abs(a.viewport[1] - b.viewport[1]) < 0.01) &&
			(std::abs(a.viewport[2] - b.viewport[2]) < 0.01) &&
			(std::abs(a.viewport[3] - b.viewport[3]) < 0.01) &&
			(a.fillMode == b.fillMode) &&
			(a.cullMode == b.cullMode) &&
			(a.depthTest == b.depthTest) &&
			(a.blend == b.blend);
	}

	class RasterShader final
	{
	public:
		struct GLInputLayoutElement : public RasterInputElementDesc
		{
			int location = -1;
		};
		~RasterShader();

		bool LoadFromFile(const char* vs, const char* ps);

		bool LoadFromString(const std::string& vs, const std::string& ps);

		void GenerateInputLayout(const std::initializer_list<RasterInputElementDesc>& descs);

		void SetTextureUnit(const char* name, unsigned unit);

		void SetUniformBufferIndex(const char* name, unsigned slot);

		inline unsigned GetShaderProgram() const
		{
			return shaderProgram;
		}

		inline const std::vector<GLInputLayoutElement>& GetInputLayout() const
		{
			return inputLayout;
		}

		unsigned CalculateVertexCount(RenderBuffer* buffer);

	private:
		unsigned shaderProgram = 0;

		std::vector<GLInputLayoutElement> inputLayout;
	};

	namespace Graphics
	{
		void EnableDebug();

		void GenerateMips(const Texture2D* texture);

		void ResolveMSAA(RenderTarget* dst, const RenderTarget* src);

		unsigned int GetCurrentTextureID();

		void CopyTexture2D(Texture2D* dst, const Texture2D* src);

		void UnPackPBO(const Texture2D* texture, const RenderBuffer* pbo);

		RenderTarget* GetRenderTarget();

		void SetRenderTarget(RenderTarget* renderTarget);

		void ClearColorBuffer(const RenderTarget* renderTarget, const float color[4] = nullptr);

		void ClearColorBuffer(const RenderTarget* renderTarget, std::initializer_list<float>&& color);

		void ClearDepth(const RenderTarget* renderTarget, float depth = 1.0f);

		void ClearStencil(const RenderTarget* renderTarget, unsigned char stencil = 0);

		void SetRasterState(const RasterState* rasterState);

		void SetVertexBuffer(const RenderBuffer* renderBuffer);

		void SetIndexBuffer(const RenderBuffer* renderBuffer);

		void SetRasterShader(const RasterShader* shader);

		void SetVSUniformBuffer(unsigned slot, const RenderBuffer* renderBuffer);

		void SetPSTexture2D(unsigned slot, const Texture2D* texture);

		void SetPSSampler(unsigned slot, const SamplerState* sampler);

		void Draw(PrimitiveType primitiveType, unsigned count, unsigned start = 0);

		void DrawIndexed(PrimitiveType primitiveType, unsigned count);
	}

}