#include "camera.h"

namespace kdray
{
	void Camera::LookAt(float3 eye, float3 target, float3 up)
	{
		_viewMatrix = lookat(eye, target, up);
		_transformMatrix = inverse(_viewMatrix);
		_dirty = true;
	}

	void Camera::Perspect(float fovY)
	{
		_fovY = fovY;
		_projectMartix = perspective(radian(fovY), _aspect, _nearZ, _farZ);
		_dirty = true;
	}

	void Camera::Perspect(float fovY, float aspect, float nearZ, float farZ)
	{
		_fovY = fovY;
		_aspect = aspect;
		_nearZ = nearZ;
		_farZ = farZ;
		_projectMartix = perspective(radian(fovY), aspect, nearZ, farZ);
		_dirty = true;
	}

	void Camera::SetTransformMartix(float martix[16])
	{
		_transformMatrix = make_float4x4
		(
			martix[0], martix[1], martix[2], martix[3],
			martix[4], martix[5], martix[6], martix[7],
			martix[8], martix[9], martix[10], martix[11],
			martix[12], martix[13], martix[14], martix[15]
		);
		_viewMatrix = inverse(_transformMatrix);
		_dirty = true;
	}

	void Camera::SetProjectMartix(float martix[16])
	{
		_projectMartix = make_float4x4
		(
			martix[0], martix[1], martix[2], martix[3],
			martix[4], martix[5], martix[6], martix[7],
			martix[8], martix[9], martix[10], martix[11],
			martix[12], martix[13], martix[14], martix[15]
		);
		_dirty = true;
	}

	void Camera::Update()
	{
		if (_dirty)
		{
			_rasterToWorldMatrix = inverse(_projectMartix * _viewMatrix);
			_dirty = false;
		}
	}
}
