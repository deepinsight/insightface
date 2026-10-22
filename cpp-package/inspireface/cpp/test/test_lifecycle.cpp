/**
 * Isolated runner for C API lifecycle tests.
 *
 * Unlike test.cpp, this runner deliberately does not launch InspireFace before
 * Catch2 starts. Each test owns the complete SDK lifecycle it exercises.
 */
#include <string>

#define CATCH_CONFIG_RUNNER

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"

namespace {

void InitTestLogger() {
    auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("TEST", sink);
#if ENABLE_TEST_MSG
    logger->set_level(spdlog::level::trace);
#else
    logger->set_level(spdlog::level::off);
#endif
    logger->set_pattern("%Y-%m-%d %H:%M:%S.%e [Test Message] ===> %v");
    spdlog::register_logger(logger);
}

void EnsureTerminated() {
    HInt32 status = HF_STATUS_DISABLE;
    if (HFQueryInspireFaceLaunchStatus(&status) == HSUCCEED && status == HF_STATUS_ENABLE) {
        HFTerminateInspireFace();
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    InitTestLogger();
    TEST_PRINT_OUTPUT(true);

    Catch::Session session;
    std::string pack;
    std::string test_dir;
    std::string pack_path;
    auto cli = session.cli() | Catch::clara::Opt(pack, "value")["--pack"]("Resource pack filename") |
               Catch::clara::Opt(test_dir, "value")["--test_dir"]("Test dir resource") |
               Catch::clara::Opt(pack_path, "value")["--pack_path"]("The specified path to the pack file");
    session.cli(cli);

    const int parse_result = session.applyCommandLine(argc, argv);
    if (parse_result != 0) {
        return parse_result;
    }

    if (!test_dir.empty()) {
        SET_TEST_DIR(test_dir);
    }

    if (!pack.empty()) {
        SET_PACK_NAME(pack);
        SET_RUNTIME_FULLPATH_NAME(GET_MODEL_FILE());
    } else if (!pack_path.empty()) {
        SET_RUNTIME_FULLPATH_NAME(pack_path);
    } else {
        SET_RUNTIME_FULLPATH_NAME(GET_MODEL_FILE());
    }

    EnsureTerminated();
    const int result = session.run();
    EnsureTerminated();
    return result;
}
