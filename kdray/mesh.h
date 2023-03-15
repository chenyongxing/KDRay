#pragma once

#include "common.h"
#include "cuda/shared_types.h"
#include <optix.h>

namespace kdray
{
	class Mesh
	{
	public:
		~Mesh();

		bool Create(const std::vector<MeshVertex>& vertices, const std::vector<uint32_t>& indices);
		bool LoadFromFile(const std::string& filepath);
		bool LoadFromObj(const std::string& filepath);
		bool LoadFromPly(const std::string& filepath);

		inline MeshVertex* GetVertexBuffer()
		{
			return _vertexBuffer;
		}
		
		inline uint32_t* GetIndexBuffer()
		{
			return _indexBuffer;
		}

		inline OptixTraversableHandle GetTraverHandle()
		{
			return _traverHandle;
		}

	private:
		MeshVertex* _vertexBuffer = nullptr;
		uint32_t* _indexBuffer = nullptr;
		void* _accelBuffer = nullptr;
		OptixTraversableHandle _traverHandle = 0;
	};
}
