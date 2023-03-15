#pragma once

#include "common.h"
#include "cuda/math.h"

namespace kdray
{
	class Camera
	{
	public:
		float fStop = 0.0f;
		float focalDistance = 1.0f;

		void LookAt(float3 eye, float3 target, float3 up);
		void Perspect(float fovY);
		void Perspect(float fovY, float aspect, float nearZ, float farZ);
		void SetTransformMartix(float martix[16]);
		void SetProjectMartix(float martix[16]);
		void Update();

		inline float GetNearZ()
		{
			return _nearZ;
		}

		inline float GetFarZ()
		{
			return _farZ;
		}

		inline float GetFovY()
		{
			return _fovY;
		}

		inline float GetAspect()
		{
			return _aspect;
		}

		inline float3 GetPosition()
		{
			return transform_point(_transformMatrix, make_float3(0.0f));
		}

		inline float4x4& GetTransformMatrix()
		{
			return _transformMatrix;
		}

		inline float4x4& GetRasterToWorldMatrix()
		{
			return _rasterToWorldMatrix;
		}

		inline void GetTransformMatrix(float m[16])
		{
			m[0] = _transformMatrix.m[0][0];
			m[1] = _transformMatrix.m[0][1];
			m[2] = _transformMatrix.m[0][2];
			m[3] = _transformMatrix.m[0][3];

			m[4] = _transformMatrix.m[1][0];
			m[5] = _transformMatrix.m[1][1];
			m[6] = _transformMatrix.m[1][2];
			m[7] = _transformMatrix.m[1][3];

			m[8] = _transformMatrix.m[2][0];
			m[9] = _transformMatrix.m[2][1];
			m[10] = _transformMatrix.m[2][2];
			m[11] = _transformMatrix.m[2][3];

			m[12] = _transformMatrix.m[3][0];
			m[13] = _transformMatrix.m[3][1];
			m[14] = _transformMatrix.m[3][2];
			m[15] = _transformMatrix.m[3][3];
		}

	protected:
		float _nearZ = 0.1f;
		float _farZ = 100.f;
		float _fovY = 30.0f;
		float _aspect = 1.0f;

		bool _dirty = true;
		float4x4 _transformMatrix;
		float4x4 _viewMatrix;
		float4x4 _projectMartix;
		float4x4 _rasterToWorldMatrix;
	};
}
