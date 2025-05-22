/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#include <string>
#define CATCH_CONFIG_RUNNER

#include <iostream>
#include "settings/test_settings.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#include "spdlog/spdlog.h"
#include "inspireface/c_api/inspireface.h"
#include "unit/test_helper/simple_csv_writer.h"

int init_test_logger() {
    std::string name("TEST");
    auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>(name, stdout_sink);
#if ENABLE_TEST_MSG
    logger->set_level(spdlog::level::trace);
#else
    logger->set_level(spdlog::level::off);
#endif
    logger->set_pattern("%Y-%m-%d %H:%M:%S.%e [Test Message] ===> %v");
    spdlog::register_logger(logger);
    return 0;
}

int init_test_benchmark_record() {
#ifdef ISF_ENABLE_BENCHMARK
    if (std::remove(getBenchmarkRecordFile().c_str()) != 0) {
        spdlog::trace("Error deleting file");
    }
    BenchmarkRecord record(getBenchmarkRecordFile(), TEST_MODEL_FILE);
#endif
    return 0;
}

int init_test_evaluation_record() {
#ifdef ISF_ENABLE_TEST_EVALUATION
    if (std::remove(getEvaluationRecordFile().c_str()) != 0) {
        spdlog::trace("Error deleting file");
    }
    EvaluationRecord record(getEvaluationRecordFile());
#endif
    return 0;
}

int main(int argc, char* argv[]) {
    init_test_logger();
    init_test_benchmark_record();
    init_test_evaluation_record();
    TEST_PRINT_OUTPUT(true);

    Catch::Session session;
    // Pack file name and test directory
    std::string pack;
    std::string testDir;
    std::string packPath;

    HInt32 ret;

    // Add command line options
    auto cli = session.cli() | Catch::clara::Opt(pack, "value")["--pack"]("Resource pack filename") |
               Catch::clara::Opt(testDir, "value")["--test_dir"]("Test dir resource") |
               Catch::clara::Opt(packPath, "value")["--pack_path"]("The specified path to the pack file");

    // Set combined CLI to the session
    session.cli(cli);

    // Parse command line arguments
    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0)  // Indicate an error
        return returnCode;

    if (!testDir.empty()) {
        SET_TEST_DIR(testDir);
        TEST_PRINT("Updated test dir to: {}", getTestDataDir());
    } else {
        TEST_PRINT("Using default test dir: {}", getTestDataDir());
    }

#if defined(ISF_ENABLE_TENSORRT)
    HInt32 support_cuda;
    ret = HFCheckCudaDeviceSupport(&support_cuda);
    if (ret != HSUCCEED) {
        TEST_ERROR_PRINT("An error occurred while checking CUDA device support: {}", ret);
        return ret;
    }
    if (!support_cuda) {
        TEST_ERROR_PRINT("CUDA device support is not available");
        return HERR_DEVICE_CUDA_NOT_SUPPORT;
    }

    HFPrintCudaDeviceInfo();
#endif

    std::string fullPath;
    // Check whether custom parameters are set
    if (!pack.empty()) {
        SET_PACK_NAME(pack);
        fullPath = GET_MODEL_FILE();
        TEST_PRINT("Updated global Pack to: {}", TEST_MODEL_FILE);
        SET_RUNTIME_FULLPATH_NAME(fullPath);
    } else if (!packPath.empty()) {
        fullPath = packPath;
        TEST_PRINT("Updated global Pack File to: {}", packPath);
        SET_RUNTIME_FULLPATH_NAME(packPath);
    } else {
        fullPath = GET_MODEL_FILE();
        TEST_PRINT("Using default global Pack: {}", TEST_MODEL_FILE);
        SET_RUNTIME_FULLPATH_NAME(fullPath);
    }

    TEST_PRINT("Launching InspireFace with path: {}", fullPath);
    ret = HFLaunchInspireFace(fullPath.c_str());
    if (ret != HSUCCEED) {
        TEST_ERROR_PRINT("An error occurred while starting InspireFace: {}", ret);
        return ret;
    }
    
    // Set log level
    HFSetLogLevel(HF_LOG_INFO);

    // Run the test
    ret = session.run();

    // Terminate the InspireFace instance
    HFTerminateInspireFace();

    // Show resource statistics
    HFDeBugShowResourceStatistics();

    return ret;
}
