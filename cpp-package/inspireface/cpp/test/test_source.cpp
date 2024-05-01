//
// Created by Tunm-Air13 on 2023/9/12.
//


#define CATCH_CONFIG_RUNNER
#include <iostream>
#include "settings/test_settings.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#include "spdlog/spdlog.h"
#include "Initialization_module/launch.h"


#define ENABLE_DRAW_SPLIT_LINE 1               // Whether dividers are printed during the test
#define ENABLE_TEST_MSG 1                      // TEST PRINT output

int init_test_logger() {
    std::string name("TEST");
    auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>(name, stdout_sink);
#if ENABLE_TEST_MSG
    logger->set_level(spdlog::level::trace);
#else
    logger->set_level(spdlog::level::off);
#endif
    logger->set_pattern("%Y-%m-%d %H:%M:%S.%e [test message] =====> %v");
    spdlog::register_logger(logger);
    return 0;
}

int main(int argc, char* argv[]) {
    init_test_logger();

    auto ret = INSPIRE_LAUNCH->Load("test_res/pack/Pikachu");
    if (ret != 0) {
        std::cerr << "Load error" << std::endl;
        return -1;
    }

    return Catch::Session().run(argc, argv);
}