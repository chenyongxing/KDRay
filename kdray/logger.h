#pragma once

#include <spdlog/spdlog.h>

namespace kdray
{
	spdlog::logger* GetLogger();
	void LoggerInit();
	void LoggerDestory();
}
