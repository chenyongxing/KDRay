#pragma once

#include <map>
#include <glm/glm.hpp>

class Input
{
public:
	enum struct MouseButton
	{
		Left = 0,
		Middle = 2,
		Right = 1
	};

	enum struct Keycode
	{
		Q = 81,
		W = 87,
		E = 69,
		R = 82,
		T = 84,
		A = 65,
		S = 83,
		D = 68,
		F = 70,
		Shift = 340,
		Control = 341,
		ALT = 342
	};

public:
	static glm::vec2& MousePosition();

	static bool MouseLeftButtonDown();

	static bool MouseMiddleButtonDown();

	static bool MouseRightButtonDown();

	static bool IsKeyDown(Keycode keycode);

	static float scrollWheelDelta;
	static bool mouseLeftDrag;
	static bool mouseMiddleDrag;
	static bool mouseRightDrag;
	static glm::vec2 dragDelta;

public:
	static void Update();
	static void LastUpdate();
	static void Reset();

	static glm::vec2 mousePos;
	static std::map<MouseButton, bool> mouseButtondowns;
	static std::map<Keycode, bool> keydowns;
	static bool scrollWheelUpdated;

private:
	static glm::vec2 dragStartPos;
	static glm::vec2 lastMousePos;
	static bool needScrollWheelReset;
};
