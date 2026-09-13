#include <chrono>
#include <format>
#include <iostream>
#include <mutex>
#include <string>

#include "common/logger.h"


namespace cuinfer
{
namespace
{

std::mutex log_mutex;

const char* level_name(LogLevel level) noexcept
{
    switch (level)
    {
    case LogLevel::kInfo:
        return "INFO";
    case LogLevel::kError:
        return "ERROR";
    case LogLevel::kDebug:
        return "DEBUG";
    }

    return "UNKNOWN";
}

std::string_view source_file_name(std::string_view path) noexcept
{
    const auto separator = path.find_last_of("/\\");
    return separator == std::string_view::npos ? path : path.substr(separator + 1);
}

}

void Logger::log(
    LogLevel level,
    std::string_view message,
    const std::source_location& location
) noexcept
{
#ifdef NDEBUG
    if (level == LogLevel::kDebug)
    {
        return;
    }
#endif

    try
    {
        const auto now = std::chrono::floor<std::chrono::seconds>(
            std::chrono::system_clock::now()
        );
        const std::chrono::zoned_time local_time{std::chrono::current_zone(), now};

        const std::string content = std::format(
            "{} {:%Y-%m-%d %H:%M:%S} [{}]: {}\n",
            level_name(level),
            local_time,
            source_file_name(location.file_name()),
            message
        );
        std::ostream& output = level == LogLevel::kError ? std::cerr : std::cout;

        const std::lock_guard lock(log_mutex);
        output.write(content.data(), static_cast<std::streamsize>(content.size()));
        output.flush();
    }
    catch (...)
    { }
}

}
