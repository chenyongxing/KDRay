#include "time.h"

using namespace std::chrono;

uint32_t Time::m_frameCount = 0;

std::chrono::system_clock::time_point Time::startTimePoint;

std::chrono::microseconds Time::frameTime;

std::chrono::system_clock::time_point Time::lastTimePoint;

std::map<std::string, Time::Timer> Time::timers;

std::map<std::string, system_clock::time_point> Time::stopWatchs;

void Time::Setup()
{
	startTimePoint = system_clock::now();
	lastTimePoint = startTimePoint;
}

void Time::Update()
{
	m_frameCount++;

	auto now = system_clock::now();

	frameTime = duration_cast<microseconds>(now - lastTimePoint);
	lastTimePoint = now;

	for (auto it = timers.begin(); it != timers.end();)
	{
		auto duration = duration_cast<microseconds>(now - it->second.start);
		auto second = float(duration.count()) * microseconds::period::num / microseconds::period::den;

		if (it->second.delayOver)
		{
			if (second >= it->second.period)
			{
				it->second.func();
				it->second.start = now;
				if (it->second.count > 0)
				{
					it->second.count--;
				}
			}
		}
		else
		{
			if (second >= it->second.delay)
			{
				it->second.func();
				it->second.start = now;
				it->second.delay = 0.0f;
				it->second.delayOver = true;
				if (it->second.count > 0)
				{
					it->second.count--;
				}
			}
		}

		if (it->second.count == 0)
		{
			timers.erase(it++);
		}
		else
		{
			it++;
		}
	}
}

float Time::DeltaTime()
{
	return float(frameTime.count()) * microseconds::period::num / microseconds::period::den;
}

void Time::AddTimer(std::string name, std::function<void()> func, float period, int count, float delay)
{
	Timer timer = { func, period, count, delay, false, system_clock::now() };
	timers.insert(std::make_pair(name, timer));
}

void Time::RemoveTimer(std::string name)
{
	auto it = timers.find(name);
	if (it != timers.end())
	{
		timers.erase(it);
	}
}

void Time::StopWatchStart(std::string name)
{
	stopWatchs[name] = system_clock::now();
}

float Time::StopWatchStop(std::string name)
{
	auto it = stopWatchs.find(name);
	if (it != stopWatchs.end())
	{
		auto duration = duration_cast<microseconds>(system_clock::now() - it->second);
		stopWatchs.erase(it);
		return float(duration.count()) * microseconds::period::num / microseconds::period::den;
	}

	return -1.0f;
}
