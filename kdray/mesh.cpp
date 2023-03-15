#include "mesh.h"
#include "logger.h"
#include "optix_context.h"
#include "renderer.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#define TINYPLY_IMPLEMENTATION
#include <tinyply.h>

namespace kdray
{
	Mesh::~Mesh()
	{
		CudaFree(_accelBuffer);
		CudaFree(_indexBuffer);
		CudaFree(_vertexBuffer);
	}

	bool Mesh::Create(const std::vector<MeshVertex>& vertices, const std::vector<uint32_t>& indices)
	{
		if (_vertexBuffer || _indexBuffer)
		{
			return false;
		}

		// upload mesh data to gpu
		_vertexBuffer = CudaMalloc(vertices);
		_indexBuffer = CudaMalloc(indices);

		// triangle input info
		OptixBuildInput triangleInput = {};
		triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangleInput.triangleArray.vertexStrideInBytes = sizeof(MeshVertex);
		triangleInput.triangleArray.numVertices = static_cast<unsigned int>(vertices.size());
		CUdeviceptr vertexBufferPtr = reinterpret_cast<CUdeviceptr>(_vertexBuffer);
		triangleInput.triangleArray.vertexBuffers = &vertexBufferPtr;

		triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangleInput.triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;
		triangleInput.triangleArray.numIndexTriplets = static_cast<unsigned int>(indices.size() / 3);
		triangleInput.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(_indexBuffer);

		unsigned int geometryFlag = OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_NONE;
		triangleInput.triangleArray.flags = &geometryFlag;
		triangleInput.triangleArray.numSbtRecords = 1;
		triangleInput.triangleArray.sbtIndexOffsetBuffer = 0;
		triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
		triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

		auto optixCtx = Renderer::Instance()->GetOptixContext();
		return optixCtx->BuildAccelGeometry(triangleInput, &_accelBuffer, &_traverHandle);
	}

	bool Mesh::LoadFromFile(const std::string& filepath)
	{
		std::size_t found = filepath.find_last_of(".");
		std::string extStr = filepath.substr(found + 1);
		if (extStr == "obj")
		{
			return LoadFromObj(filepath);
		}
		else if (extStr == "ply")
		{
			return LoadFromPly(filepath);
		}
		return false;
	}

	bool Mesh::LoadFromObj(const std::string& filepath)
	{
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;

		std::string warn;
		std::string err;
		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str()) || shapes.size() == 0)
		{
			GetLogger()->error("LoadFromObj {} failed", filepath);
			return false;
		}
		
		size_t numVertices = 0;
		for (size_t i = 0; i < shapes.size(); i++)
		{
			for (size_t j = 0; j < shapes[i].mesh.num_face_vertices.size(); j++)
			{
				numVertices += shapes[i].mesh.num_face_vertices[j];
			}
		}

		MeshVertex vertex;
		std::vector<MeshVertex> vertices;
		vertices.reserve(numVertices);

		for (size_t i = 0; i < shapes.size(); i++)
		{
			size_t indexOffset = 0;
			for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++)
			{
				size_t numFace = shapes[i].mesh.num_face_vertices[f];
				if (numFace == 3)
				{
					for (size_t v = 0; v < 3; v++)
					{
						tinyobj::index_t idx = shapes[i].mesh.indices[indexOffset + v];

						vertex.position.x = attrib.vertices[3 * idx.vertex_index];
						vertex.position.y = attrib.vertices[3 * idx.vertex_index + 1];
						vertex.position.z = attrib.vertices[3 * idx.vertex_index + 2];

						vertex.normal.x = attrib.normals[3 * idx.normal_index];
						vertex.normal.y = attrib.normals[3 * idx.normal_index + 1];
						vertex.normal.z = attrib.normals[3 * idx.normal_index + 2];

						vertex.texCoord.x = attrib.texcoords[2 * idx.texcoord_index];
						vertex.texCoord.y = attrib.texcoords[2 * idx.texcoord_index + 1];
						vertices.emplace_back(vertex);
					}
				}
				indexOffset += numFace;
			}
		}

		std::vector<uint32_t> indices(vertices.size());
		for (uint32_t i = 0; i < indices.size(); i++)
		{
			indices[i] = i;
		}
		Create(vertices, indices);
		return true;
	}

	bool Mesh::LoadFromPly(const std::string& filepath)
	{
		using namespace tinyply;

		std::ifstream fileStream(filepath, std::ios::binary);
		if (fileStream.good())
		{
			PlyFile plyFile;
			plyFile.parse_header(fileStream);
			std::shared_ptr<PlyData> vertices, normals, texcoords, faces;
			try 
			{ 
				vertices = plyFile.request_properties_from_element("vertex", { "x", "y", "z" });
				normals = plyFile.request_properties_from_element("vertex", { "nx", "ny", "nz" });
				texcoords = plyFile.request_properties_from_element("vertex", { "s", "t" });
				faces = plyFile.request_properties_from_element("face", { "vertex_indices" });
				if (vertices->count == 0 || vertices->t != tinyply::Type::FLOAT32|| 
					faces->count == 0 || faces->t != tinyply::Type::UINT32)
				{
					return false;
				}
			}
			catch (const std::exception&)
			{
				return false;
			}

			plyFile.read(fileStream);

			MeshVertex vertex;
			std::vector<MeshVertex> meshVertices;
			meshVertices.reserve(vertices->count);
			size_t float3Offset = 0;
			size_t float2Offset = 0;
			for (size_t i = 0; i < vertices->count; i++)
			{
				uint8_t* positionPtr = vertices->buffer.get() + float3Offset;
				uint8_t* normalPtr = normals->buffer.get() + float3Offset;
				uint8_t* texcoordPtr = texcoords->buffer.get() + float2Offset;
				float3Offset += sizeof(float) * 3;
				float2Offset += sizeof(float) * 2;

				memcpy(&vertex.position, positionPtr, sizeof(float) * 3);
				memcpy(&vertex.normal, normalPtr, sizeof(float) * 3);
				memcpy(&vertex.texCoord, texcoordPtr, sizeof(float) * 2);
				meshVertices.emplace_back(vertex);
			}

			std::vector<uint32_t> indices(faces->count * 3);
			memcpy(indices.data(), faces->buffer.get(), faces->buffer.size_bytes());
			Create(meshVertices, indices);
			return true;
		}
		return false;
	}
}
