//
// Created by Tunm-Air13 on 2023/5/24.
//
#pragma once
#ifndef BIGGUYSMAIN_TEST_SETTINGS_H
#define BIGGUYSMAIN_TEST_SETTINGS_H
#include <catch2/catch.hpp>
#include <spdlog/spdlog.h>
#include <iostream>
#include "enviro.h"


#define TEST_MODEL_FILE Enviro::getInstance().getPackName()                       // Optional model file
#define SET_PACK_NAME(name) Enviro::getInstance().setPackName(name)
#define SET_TEST_DIR(dir) Enviro::getInstance().setTestResDir(dir)
#define SET_RUNTIME_FULLPATH_NAME(name) Enviro::getInstance().setTestRuntimeFullPath(name)
#define GET_RUNTIME_FULLPATH_NAME Enviro::getInstance().getTestRuntimeFullPath()

#define TEST_LFW_FUNNELED_TXT "valid_lfw_funneled.txt"     // LFW Index txt file
#define LFW_FUNNELED_DIR ""                                // LFW funneled data dir
#define TEST_LFW_EVALUATION_TXT "pairs.txt"
#define TEST_BENCHMARK_RECORD "benchmark.csv"
#define TEST_EVALUATION_RECORD "evaluation.csv"

using namespace Catch::Detail;

#define TEST_PRINT(...) SPDLOG_LOGGER_CALL(spdlog::get("TEST"), spdlog::level::trace, __VA_ARGS__)
#define TEST_PRINT_OUTPUT(open) TestMessageBroadcast test_msg_broadcast_##open(open)
#define LOG_OUTPUT_LEVEL(level) LogLevelBroadcast log_level_broadcast_##level(level);

#define GET_DIR getTestDataDir()
#define GET_DATA(filename) getTestData(filename)
#define GET_MODEL_FILE() getTestModelsFile()
#define GET_LFW_FUNNELED_TXT() getTestLFWFunneledTxt()

#define GET_SAVE_DIR getTestSaveDir()
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


class LogLevelBroadcast {
public:
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
        std::cout
                << "==============================================================================="
                << std::endl;
    }


#define DRAW_SPLIT_LINE test_case_split split_line_x;

};

#endif //BIGGUYSMAIN_TEST_SETTINGS_H
