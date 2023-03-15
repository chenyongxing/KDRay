#include "input.h"

glm::vec2 Input::mousePos;
glm::vec2 Input::lastMousePos;

std::map<Input::MouseButton, bool> Input::mouseButtondowns;
std::map<Input::Keycode, bool> Input::keydowns;

float Input::scrollWheelDelta = 0.0f;
bool Input::mouseLeftDrag = false;
bool Input::mouseMiddleDrag = false;
bool Input::mouseRightDrag = false;
glm::vec2 Input::dragDelta;
glm::vec2 Input::dragStartPos;

bool Input::scrollWheelUpdated = false;
bool Input::needScrollWheelReset = false;

glm::vec2& Input::MousePosition()
{
	return mousePos;
}

bool Input::MouseLeftButtonDown()
{
	auto it = mouseButtondowns.find(MouseButton::Left);
	if (it != mouseButtondowns.end())
	{
		return it->second;
	}
	else
	{
		return false;
	}
}

bool Input::MouseMiddleButtonDown()
{
	auto it = mouseButtondowns.find(MouseButton::Middle);
	if (it != mouseButtondowns.end())
	{
		return it->second;
	}
	else
	{
		return false;
	}
}

bool Input::MouseRightButtonDown()
{
	auto it = mouseButtondowns.find(MouseButton::Right);
	if (it != mouseButtondowns.end())
	{
		return it->second;
	}
	else
	{
		return false;
	}
}

bool Input::IsKeyDown(Keycode keycode)
{
	auto it = keydowns.find(keycode);
	if (it != keydowns.end())
	{
		return it->second;
	}
	else
	{
		return false;
	}
}

void Input::Update()
{
	if (MouseRightButtonDown())
	{
		if (!mouseRightDrag)
		{
			mouseRightDrag = true;
			dragStartPos = mousePos;
		}
		else
		{
			dragDelta = mousePos - lastMousePos;
		}
	}
	else
	{
		mouseRightDrag = false;
	}

	if (MouseMiddleButtonDown())
	{
		if (!mouseMiddleDrag)
		{
			mouseMiddleDrag = true;
			dragStartPos = mousePos;
		}
		else
		{
			dragDelta = mousePos - lastMousePos;
		}
	}
	else
	{
		mouseMiddleDrag = false;
	}

	if (MouseLeftButtonDown())
	{
		if (!mouseLeftDrag)
		{
			mouseLeftDrag = true;
			dragStartPos = mousePos;
		}
		else
		{
			dragDelta = mousePos - lastMousePos;
		}
	}
	else
	{
		mouseLeftDrag = false;
	}

	lastMousePos = mousePos;

	if (scrollWheelUpdated)
	{
		scrollWheelUpdated = false;
		needScrollWheelReset = true;
	}
}

void Input::LastUpdate()
{
	if (needScrollWheelReset)
	{
		scrollWheelDelta = 0.0f;
		needScrollWheelReset = false;
	}
}

void Input::Reset()
{
	mouseButtondowns.clear();
	keydowns.clear();
}
