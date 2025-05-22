/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#pragma once
#ifndef BIGGUYSMAIN_TEST_SETTINGS_H
#define BIGGUYSMAIN_TEST_SETTINGS_H
#include <catch2/catch.hpp>
#include <spdlog/spdlog.h>
#include <iostream>
#include "enviro.h"
#include "check.h"
#include "inspireface/middleware/system.h"
#include "inspireface/include/inspireface/spend_timer.h"

// Define the test model file
#define TEST_MODEL_FILE Enviro::getInstance().getPackName()

// Set the pack name
#define SET_PACK_NAME(name) Enviro::getInstance().setPackName(name)

// Set the test directory
#define SET_TEST_DIR(dir) Enviro::getInstance().setTestResDir(dir)
// Set the runtime full path
#define SET_RUNTIME_FULLPATH_NAME(name) Enviro::getInstance().setTestRuntimeFullPath(name)
// Get the runtime full path
#define GET_RUNTIME_FULLPATH_NAME Enviro::getInstance().getTestRuntimeFullPath()

// Define the LFW funneled index txt file
#define TEST_LFW_FUNNELED_TXT "valid_lfw_funneled.txt"
// Define the LFW funneled data directory
#define LFW_FUNNELED_DIR ""
// Define the LFW evaluation txt file
#define TEST_LFW_EVALUATION_TXT "pairs.txt"
// Define the benchmark record file
#define TEST_BENCHMARK_RECORD "benchmark.csv"
// Define the evaluation record file
#define TEST_EVALUATION_RECORD "evaluation.csv"

using namespace Catch::Detail;

// Print test message
#define TEST_PRINT(...) SPDLOG_LOGGER_CALL(spdlog::get("TEST"), spdlog::level::trace, __VA_ARGS__)
// Print test message output
#define TEST_PRINT_OUTPUT(open) TestMessageBroadcast test_msg_broadcast_##open(open)
// Set the log output level
#define LOG_OUTPUT_LEVEL(level) LogLevelBroadcast log_level_broadcast_##level(level);
// Print test error message
#define TEST_ERROR_PRINT(...) SPDLOG_LOGGER_CALL(spdlog::get("TEST"), spdlog::level::err, __VA_ARGS__)

// Get the test data directory
#define GET_DIR getTestDataDir()
// Get the test data
#define GET_DATA(filename) getTestData(filename)
// Get the test models file
#define GET_MODEL_FILE() getTestModelsFile()
// Get the LFW funneled index txt file
#define GET_LFW_FUNNELED_TXT() getTestLFWFunneledTxt()
// Get the LFW funneled evaluation txt file
#define GET_LFW_FUNNELED_EVA_TXT() getTestLFWFunneledEvaTxt()
// Get the LFW funneled directory
#define GET_LFW_FUNNELED_DIR() getLFWFunneledDir()
// Get the benchmark record file
#define GET_BENCHMARK_RECORD_FILE() getBenchmarkRecordFile()
// Get the evaluation record file
#define GET_SAVE_DIR getTestSaveDir()
// Get the test save data
#define GET_SAVE_DATA(filename) getTestSaveData(filename)

std::string getTestDataDir();

std::string getTestData(const std::string &name);

std::string getTestSaveDir();

std::string getTestSaveData(const std::string &name);

std::string getTestModelsFile();

std::string getTestLFWFunneledTxt();

std::string getTestLFWFunneledEvaTxt();

std::string getLFWFunneledDir();

std::string getBenchmarkRecordFile();

std::string getEvaluationRecordFile();

/** Logger level */
enum LOG_LEVEL {
    TRACE = 0,  ///< trace
    DEBUG = 1,  ///< debug
    INFO = 2,   ///< information
    WARN = 3,   ///< warn
    ERROR = 4,  ///< error
    FATAL = 5,  ///< fatal
    OFF = 6,    ///< off
};

class TestMessageBroadcast {
public:
    explicit TestMessageBroadcast(bool open) : m_old_level(spdlog::level::trace) {
        auto logger = spdlog::get("TEST");
        m_old_level = logger->level();
        if (open) {
            if (m_old_level != spdlog::level::trace) {
                logger->set_level(spdlog::level::trace);
            }
        } else {
            if (m_old_level == spdlog::level::trace) {
                logger->set_level(spdlog::level::info);
            }
        }
    }

    ~TestMessageBroadcast() {
        spdlog::get("TEST")->set_level(m_old_level);
    }

private:
    spdlog::level::level_enum m_old_level;
};

/**
 * @class LogLevelBroadcast
 * @brief A class for broadcasting log level changes.
 *
 * This class is used to set the log level for the logger.
 */
class LogLevelBroadcast {
public:
    /**
     * @brief Constructor for LogLevelBroadcast.
     * @param level The log level to set.
     */
    explicit LogLevelBroadcast(LOG_LEVEL level) {
        m_oldLevel = level;
        spdlog::set_level(spdlog::level::level_enum(level));
    }

    ~LogLevelBroadcast() {
        spdlog::set_level(spdlog::level::level_enum(spdlog::level::trace));
    }

private:
    LOG_LEVEL m_oldLevel;
};

struct test_case_split {
    ~test_case_split() {
        std::cout << "===============================================================================" << std::endl;
    }

#define DRAW_SPLIT_LINE test_case_split split_line_x;
};

#endif  // BIGGUYSMAIN_TEST_SETTINGS_H
