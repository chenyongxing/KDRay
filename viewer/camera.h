#pragma once

#include "node.h"

class Camera;
class FreeCameraController
{
public:
	Camera* camera = nullptr;
	void Update(float deltaTime);
	float moveSpeed = 1.0f;
	float rotateSpeed = 1.0f;
};

class Camera : public Node
{
public:
	Camera();

	virtual void Update(float deltaTime) override;

	void LookAt(const glm::vec3& target);

	void Perspective(glm::vec2 size, float fovY = 60.0f);
	void Ortho(glm::vec2 size);

	void FitTargetAABB(const glm::vec3& tMin, const glm::vec3& tMax);

	inline float GetFovY()
	{
		return fovY;
	}

	inline glm::mat4 GetViewMatrix()
	{
		return WorldToLocalMatrix();
	}

	inline glm::mat4& GetProjectMatrix()
	{
		return projectMatrix;
	}
	
	bool isOrtho = false;
	float m_near = 0.1f;
	float m_far = 1000.0f;
	glm::vec3 target{ 0.0f };

	FreeCameraController freeCtrl;

protected:
	float fovY = 60.0f;
	glm::mat4 projectMatrix{ 1.0f };
};
