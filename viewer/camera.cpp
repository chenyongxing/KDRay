#include "camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>
#include "input.h"

namespace
{
    static glm::vec3 RIGHT = { 1.0f, 0.0f, 0.0f };
    static glm::vec3 FRONT = { 0.0f, 1.0f, 0.0f };
    static glm::vec3 UP = { 0.0f, 0.0f, 1.0f };

	template <typename T, typename U, typename V>
	constexpr T clamp(T val, U low, V high)
	{
		if (val < low)
			return low;
		else if (val > high)
			return high;
		else
			return val;
	}

	inline glm::vec3 TransformNormal(const glm::mat4& mat, const glm::vec3& normal)
	{
		return glm::vec3(mat * glm::vec4(normal, 0));
	}

	inline glm::vec3 TransformPoint(const glm::mat4& mat, const glm::vec3& point)
	{
		return glm::vec3(mat * glm::vec4(point, 1));
	}
}

Camera::Camera()
{
    freeCtrl.camera = this;
}

void Camera::Update(float deltaTime)
{
    Node::Update(deltaTime);
}

void Camera::LookAt(const glm::vec3& target)
{
    glm::vec3 position = GetPositon();
    auto viewMatrix = glm::lookAt(position, target, UP);
    this->target = target;
    SetWorldTransform(glm::inverse(viewMatrix));
}

void Camera::Perspective(glm::vec2 size, float fovY)
{
    isOrtho = false;
    this->fovY = fovY;
    projectMatrix = glm::perspective(glm::radians(fovY), size.x / size.y, m_near, m_far);
}

void Camera::Ortho(glm::vec2 size)
{
    isOrtho = true;
    projectMatrix = glm::ortho(0.0f, size.x, 0.0f, size.y, m_near, m_far);
}

void Camera::FitTargetAABB(const glm::vec3& tMin, const glm::vec3& tMax)
{
    glm::vec3 center = (tMin + tMax) * 0.5f;
    glm::vec3 extents = (tMax - tMin) * 0.5f;
    float radius = glm::length(extents);

    float targetDistance = radius / std::tanf(glm::radians(fovY * 0.5f));

    glm::vec3 unitOne{ 0.577350259f, 0.577350259f, -0.577350259f };
    auto viewMatrix = glm::lookAt(center + targetDistance * unitOne, center, UP);
    this->target = center;
    SetWorldTransform(glm::inverse(viewMatrix));
}

void FreeCameraController::Update(float deltaTime)
{
    using Keycode = Input::Keycode;
    glm::mat4 matrix = camera->WorldTransform();

    glm::vec3 translate{ 0.0f };
    if (Input::IsKeyDown(Keycode::W))
    {
        translate -= UP;
    }
    else if (Input::IsKeyDown(Keycode::S))
    {
        translate += UP;
    }
    if (Input::IsKeyDown(Keycode::D))
    {
        translate += RIGHT;
    }
    else if (Input::IsKeyDown(Keycode::A))
    {
        translate -= RIGHT;
    }
    if (Input::IsKeyDown(Keycode::Q))
    {
        translate -= FRONT;
    }
    else if (Input::IsKeyDown(Keycode::E))
    {
        translate += FRONT;
    }

    if (Input::mouseLeftDrag || Input::mouseRightDrag)
    {
        auto up = TransformNormal(camera->GetViewMatrix(), UP);

        //dragDelta已经是帧率相关的了
        float speed = rotateSpeed * 0.2f;
        matrix = glm::rotate(matrix, glm::radians(-Input::dragDelta.x * speed), up);
        matrix = glm::rotate(matrix, glm::radians(-Input::dragDelta.y * speed), RIGHT);
    }

    matrix = glm::translate(matrix, translate * moveSpeed * 10.f * deltaTime);

    camera->SetWorldTransform(matrix);
}

