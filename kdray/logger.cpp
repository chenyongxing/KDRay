#include "logger.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace kdray
{
	static std::unique_ptr<spdlog::logger> Logger;

	spdlog::logger* GetLogger()
	{
		return Logger.get();
	}

	void LoggerInit()
	{
		if (Logger) return;

		auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
		consoleSink->set_level(spdlog::level::trace);

		auto fileSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("kdray.log", true);
		fileSink->set_level(spdlog::level::warn);

		auto spdlog = new spdlog::logger("kdRay", { consoleSink, fileSink });
		Logger = std::unique_ptr<spdlog::logger>(spdlog);
	}

	void LoggerDestory()
	{
		if (Logger)
		{
			Logger.reset();
		}
	}
}
