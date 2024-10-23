#pragma once

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace dicp {

enum class LogLevel { DEBUG, INFO, WARN, ERROR, FATAL };

const std::unordered_map<LogLevel, spdlog::level::level_enum> levelMap = {{LogLevel::DEBUG, spdlog::level::debug},
                                                                          {LogLevel::INFO, spdlog::level::info},
                                                                          {LogLevel::WARN, spdlog::level::warn},
                                                                          {LogLevel::ERROR, spdlog::level::err},
                                                                          {LogLevel::FATAL, spdlog::level::critical}};

const std::unordered_map<std::string, LogLevel> strToLevel = {
    {"DEBUG", LogLevel::DEBUG}, {"INFO", LogLevel::INFO}, {"WARN", LogLevel::WARN}, {"ERROR", LogLevel::ERROR}, {"FATAL", LogLevel::FATAL}};

class LoggerInitializer {
public:
    LoggerInitializer() { initLogger(); }

    static LogLevel getCachedLogLevel() {
        static LogLevel cachedLevel = initLogLevel();
        return cachedLevel;
    }

    static bool shouldLogToFile() {
        static bool cachedShouldLogToFile = initShouldLogToFile();
        return cachedShouldLogToFile;
    }

    static const std::string& getLogFilePath() {
        static std::string cachedLogFilePath = initLogFilePath();
        return cachedLogFilePath;
    }

private:
    static void initLogger() {
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] %v");
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

        std::vector<spdlog::sink_ptr> sinks{console_sink};

        if (shouldLogToFile()) {
            auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(getLogFilePath(), true);
            sinks.push_back(file_sink);
        }

        auto logger = std::make_shared<spdlog::logger>("dicp", sinks.begin(), sinks.end());
        spdlog::set_default_logger(logger);

        spdlog::set_level(levelMap.at(getCachedLogLevel()));
    }

    static LogLevel initLogLevel() {
        const char* log_level_env = std::getenv("DICP_LOG_LEVEL");
        if (log_level_env != nullptr) {
            auto it = strToLevel.find(log_level_env);
            if (it != strToLevel.end()) {
                return it->second;
            }
        }
        return LogLevel::ERROR;
    }

    static bool initShouldLogToFile() {
        const char* log_to_file_env = std::getenv("DICP_LOG_TO_FILE");
        return (log_to_file_env != nullptr && std::string(log_to_file_env) == "1");
    }

    static std::string initLogFilePath() {
        const char* log_file_path_env = std::getenv("DICP_LOG_FILE_PATH");
        return (log_file_path_env != nullptr) ? log_file_path_env : "logs/dicp.log";
    }
};

class LogMessage {
public:
    LogMessage(LogLevel level) : level_(level) {}
    ~LogMessage() noexcept(false) {
        spdlog::log(levelMap.at(level_), stream_.str());
        if (level_ == LogLevel::FATAL) {
            throw std::runtime_error("Fatal error occurred: " + stream_.str());
        }
    }

    template <typename T>
    LogMessage& operator<<(const T& value) {
        stream_ << value;
        return *this;
    }

private:
    LogLevel level_;
    std::ostringstream stream_;
};

static LoggerInitializer loggerInitializer;

}  // namespace dicp

#define DICP_LOG(LEVEL) \
    if (dicp::LoggerInitializer::getCachedLogLevel() <= dicp::LogLevel::LEVEL) dicp::LogMessage(dicp::LogLevel::LEVEL)

#define DICP_LOG_IF(condition, LEVEL) \
    if ((condition) && dicp::LoggerInitializer::getCachedLogLevel() <= dicp::LogLevel::LEVEL) dicp::LogMessage(dicp::LogLevel::LEVEL)
