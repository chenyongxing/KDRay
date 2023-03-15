#pragma once

#include <stdint.h>
#include <chrono>
#include <map>
#include <string>
#include <functional>

class Time
{
public:
	struct Timer
	{
		std::function<void()> func;
		float period;
		int count;
		float delay;
		bool delayOver = false;
		std::chrono::system_clock::time_point start;
		std::chrono::system_clock::time_point end;
	};

	static void Setup();

	static void Update();

	static float DeltaTime();

	static void AddTimer(std::string name, std::function<void()> func, float period, int count = -1, float delay = 0.0f);

	static void RemoveTimer(std::string name);

	static void StopWatchStart(std::string name);

	static float StopWatchStop(std::string name);

	inline static uint32_t FrameCount()
	{
		return m_frameCount;
	};

private:
	static uint32_t m_frameCount;

	static std::chrono::system_clock::time_point startTimePoint;
	static std::chrono::microseconds frameTime;
	static std::chrono::system_clock::time_point lastTimePoint;

	static std::map<std::string, Timer> timers;
	static std::map<std::string, std::chrono::system_clock::time_point> stopWatchs;
};
