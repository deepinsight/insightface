cmake_minimum_required(VERSION 3.20)
project(InspireFaceTest)

if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()
# If you want to test the benchmark, you need to set this to ON
option(ISF_ENABLE_BENCHMARK "Enable the benchmark test cases." OFF)
# If you want to test the LFW data set, you need to set this to ON
option(ISF_ENABLE_USE_LFW_DATA "Enable test cases for LFW data sets." OFF)
# If you want to test the evaluation function, you need to set this to ON, need LFW data set.
option(ISF_ENABLE_TEST_EVALUATION "Enable evaluation function test cases." OFF)
# If you want to test the base test cases, you need to set this to ON, if you want to hide the symbols, you need to set ISF_ENABLE_SYMBOL_HIDING to ON
option(ISF_ENABLE_TEST_INTERNAL "Enable internal test cases." OFF)
option(ISF_ENFORCE_PUBLIC_API_TEST_COVERAGE "Fail configuration when a public C API has no external test reference." ON)
option(ISF_ENFORCE_CPP_API_TEST_COVERAGE "Fail configuration when a baseline public C++ API has no external test reference." ON)

if (ISF_ENABLE_BENCHMARK)
    add_definitions("-DISF_ENABLE_BENCHMARK")
endif ()

if (ISF_ENABLE_USE_LFW_DATA)
    add_definitions("-DISF_ENABLE_USE_LFW_DATA")
endif ()

if (ISF_ENABLE_TEST_EVALUATION)
    add_definitions("-DISF_ENABLE_TEST_EVALUATION")
endif ()

if (ISF_ENABLE_RKNN)
    if (ISF_RKNPU_MAJOR STREQUAL "rknpu1")
    set(DEPEND rknn_api dl)
    set(ISF_RKNN_API_LIB ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/rknn/${ISF_RKNPU_MAJOR}/runtime/${ISF_RK_DEVICE_TYPE}/Linux/librknn_api/${CPU_ARCH}/)
        message("Enable RKNN Inference")
        link_directories(${ISF_RKNN_API_LIB})
        set(DEPEND rknn_api dl)
    elseif(ISF_RKNPU_MAJOR STREQUAL "rknpu2" AND ISF_RK_COMPILER_TYPE STREQUAL "aarch64")
        set(DEPEND rknnrt dl)
        if(ANDROID)
        set(RK_PLATFORM "Android")
        else()
            set(RK_PLATFORM "Linux")
        endif()
        set(ISF_RKNN_API_LIB ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/rknn/${ISF_RKNPU_MAJOR}/runtime/${RK_PLATFORM}/librknn_api/${ISF_RK_COMPILER_TYPE}/)
        message("ISF_RKNN_API_LIB: ${ISF_RKNN_API_LIB}")
        link_directories(${ISF_RKNN_API_LIB})
        set(DEPEND rknnrt dl)
    else()
        set(DEPEND dl)
    endif()
endif ()

include_directories(${SRC_DIR})

# =======================Common File Configuration===========================
set(TEST_COMMON_FILES ${CMAKE_CURRENT_SOURCE_DIR}/settings/test_settings.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/settings)

# =======================Internal Base Import Tests===========================
if (ISF_ENABLE_TEST_INTERNAL)
    file(GLOB_RECURSE INTERNAL_TEST_INTERNAL_FILES CONFIGURE_DEPENDS unit/base/*.cpp)
    add_executable(TestInternal ${CMAKE_CURRENT_SOURCE_DIR}/test_base.cpp ${INTERNAL_TEST_INTERNAL_FILES} ${TEST_COMMON_FILES})
    target_link_libraries(TestInternal InspireFace ${DEPEND})
    target_include_directories(TestInternal PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/
            ${ISF_THIRD_PARTY_DIR}/spdlog/include
            ${ISF_THIRD_PARTY_DIR}/Catch2/single_include/
            ${ISF_THIRD_PARTY_DIR}/indicators/include/
            ${SRC_DIR}
            ${INSPIRECV_INCLUDE_PATH}
    )

    set_target_properties(TestInternal PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/")
endif()

# =======================External API Testing===========================
file(GLOB_RECURSE TEST_INTERNAL_FILES CONFIGURE_DEPENDS unit/api/*.cpp)
file(GLOB_RECURSE TEST_API_LIFECYCLE_FILES CONFIGURE_DEPENDS unit/api_lifecycle/*.cpp)

if(ISF_ENFORCE_PUBLIC_API_TEST_COVERAGE)
    file(READ "${SRC_DIR}/inspireface/c_api/inspireface.h" ISF_C_API_HEADER_SOURCE)
    string(REGEX MATCHALL "HYPER_CAPI_EXPORT extern[^\n;]*HF[A-Za-z0-9_]+\\(" ISF_C_API_DECLARATIONS "${ISF_C_API_HEADER_SOURCE}")
    set(ISF_C_API_TEST_SOURCE "")
    foreach(ISF_C_API_TEST_FILE IN LISTS TEST_INTERNAL_FILES TEST_API_LIFECYCLE_FILES)
        file(READ "${ISF_C_API_TEST_FILE}" ISF_C_API_TEST_FILE_SOURCE)
        string(APPEND ISF_C_API_TEST_SOURCE "\n${ISF_C_API_TEST_FILE_SOURCE}")
    endforeach()

    set(ISF_UNTESTED_C_API_SYMBOLS "")
    foreach(ISF_C_API_DECLARATION IN LISTS ISF_C_API_DECLARATIONS)
        string(REGEX MATCH "HF[A-Za-z0-9_]+\\(" ISF_C_API_SYMBOL_WITH_PAREN "${ISF_C_API_DECLARATION}")
        string(REPLACE "(" "" ISF_C_API_SYMBOL "${ISF_C_API_SYMBOL_WITH_PAREN}")
        string(FIND "${ISF_C_API_TEST_SOURCE}" "${ISF_C_API_SYMBOL}(" ISF_C_API_TEST_POSITION)
        if(ISF_C_API_TEST_POSITION EQUAL -1)
            list(APPEND ISF_UNTESTED_C_API_SYMBOLS "${ISF_C_API_SYMBOL}")
        endif()
    endforeach()

    if(ISF_UNTESTED_C_API_SYMBOLS)
        list(JOIN ISF_UNTESTED_C_API_SYMBOLS ", " ISF_UNTESTED_C_API_SYMBOLS_TEXT)
        message(FATAL_ERROR "Public C APIs without an external test reference: ${ISF_UNTESTED_C_API_SYMBOLS_TEXT}")
    endif()
endif()
add_executable(Test ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp ${TEST_INTERNAL_FILES} ${TEST_COMMON_FILES})
target_link_libraries(Test InspireFace ${DEPEND})
target_include_directories(Test PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/
        ${ISF_THIRD_PARTY_DIR}/spdlog/include
        ${ISF_THIRD_PARTY_DIR}/Catch2/single_include/
        ${ISF_THIRD_PARTY_DIR}/indicators/include/
        ${SRC_DIR}
        )

set_target_properties(Test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/")

# =======================C API Lifecycle Testing=========================
# Lifecycle tests intentionally start without a preloaded resource pack so that
# launch/reload/terminate behavior can be verified without depending on test order.
add_executable(TestLifecycle ${CMAKE_CURRENT_SOURCE_DIR}/test_lifecycle.cpp ${TEST_API_LIFECYCLE_FILES} ${TEST_COMMON_FILES})
target_link_libraries(TestLifecycle InspireFace ${DEPEND})
target_include_directories(TestLifecycle PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/
        ${ISF_THIRD_PARTY_DIR}/spdlog/include
        ${ISF_THIRD_PARTY_DIR}/Catch2/single_include/
        ${ISF_THIRD_PARTY_DIR}/indicators/include/
        ${SRC_DIR}
        )

set_target_properties(TestLifecycle PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/")

# =======================CPP API Testing===========================
file(GLOB_RECURSE TEST_CPP_FILES CONFIGURE_DEPENDS unit/cpp_api/*.cpp)
file(GLOB_RECURSE TEST_CPP_LIFECYCLE_FILES CONFIGURE_DEPENDS unit/cpp_api_lifecycle/*.cpp)

if(ISF_ENFORCE_CPP_API_TEST_COVERAGE)
    # C++ overloads, inline methods, and private PIMPL helpers make declaration
    # regexes unreliable. Keep an explicit exported-surface baseline and require
    # every name to remain referenced by an external C++ API test.
    set(ISF_REQUIRED_CPP_API_SYMBOLS
            GetCudaDeviceCount CheckCudaUsability _PrintCudaDeviceInfo PrintCudaDeviceInfo
            GetInspireFaceVersionMajorStr GetInspireFaceVersionMinorStr
            GetInspireFaceVersionPatchStr GetInspireFaceExtendedInformation GetSDKInfo
            GetFullVersionInfo GetVersionMajorStr GetVersionMinorStr GetVersionPatchStr GetVersionString
            GetComponentName GetComponentVersion GetComponentVersionsString GetDiagnosticInfo
            getInstance setLogLevel getLogLevel logAndroid logStandard
            Load Reload getMArchive isMLoad Unload
            SetRockchipDmaHeapPath GetRockchipDmaHeapPath
            ConfigurationExtensionPath GetExtensionPath
            SetGlobalCoreMLInferenceMode GetGlobalCoreMLInferenceMode BuildAppleExtensionPath
            SetGlobalCPUEnginePowerMode GetGlobalCPUEnginePowerMode
            SetCudaDeviceId GetCudaDeviceId SetFaceDetectPixelList GetFaceDetectPixelList
            SetFaceDetectModelList GetFaceDetectModelList SwitchLandmarkEngine
            SwitchImageProcessingBackend GetImageProcessingBackend
            SetImageProcessAlignedWidth GetImageProcessAlignedWidth
            Create SetDataBuffer SetPreviewSize SetPreviewScale SetRotationMode SetDataFormat SetDestFormat
            ExecuteImageAffineProcessing ExecutePreviewImageProcessing GetPreviewScale
            ExecuteImageScaleProcessing GetAffineMatrix GetRotationModeAffineMatrix GetHeight GetWidth getRotationMode
            CreatePtr ClearTrackingFace SetTrackLostRecoveryMode SetLightTrackConfidenceThreshold
            SetTrackPreviewSize SetFilterMinimumFacePixelSize SetFaceDetectThreshold
            SetTrackModeSmoothRatio SetTrackModeNumSmoothCacheFrame SetTrackModeDetectInterval
            FaceDetectAndTrack GetFaceBoundingBox GetFaceDenseLandmark GetFaceFiveKeyPoints
            FaceFeatureExtract GetFaceAlignmentImage FaceFeatureExtractWithAlignmentImage
            MultipleFacePipelineProcess GetRGBLivenessConfidence GetFaceMaskConfidence
            GetFaceQualityConfidence GetFaceInteractionState GetFaceInteractionAction
            GetFaceAttributeResult GetFaceEmotionResult
            GetInstance EnableHub DisableHub GetAllIds SearchFaceFeature SearchFaceFeatureV2 SearchFaceFeatureTopKCache
            SearchFaceFeatureTopK FaceFeatureInsert FaceFeatureRemove FaceFeatureUpdate GetFaceFeature
            ViewDBTable SetRecognitionThreshold SetRecognitionSearchMode CosineSimilarity
            GetSearchFaceFeatureCache GetFaceFeaturePtrCache GetFaceFeatureCount
            GetTopKConfidence GetTopKCustomIdsCache GetExistingIds
            IsConfigValid updateConfig updateConfigAndRecommendedThreshold getState getConfig convert
            destroyInstance getRecommendedCosineThreshold setRecommendedCosineThreshold
            Configure Update Finish GetResults GetStatus
            _now Start Stop Reset Get Average Total Count Min Max name Report Disable)

    set(ISF_CPP_API_TEST_SOURCE "")
    foreach(ISF_CPP_API_TEST_FILE IN LISTS TEST_CPP_FILES TEST_CPP_LIFECYCLE_FILES)
        file(READ "${ISF_CPP_API_TEST_FILE}" ISF_CPP_API_TEST_FILE_SOURCE)
        string(APPEND ISF_CPP_API_TEST_SOURCE "\n${ISF_CPP_API_TEST_FILE_SOURCE}")
    endforeach()
    set(ISF_UNTESTED_CPP_API_SYMBOLS "")
    foreach(ISF_CPP_API_SYMBOL IN LISTS ISF_REQUIRED_CPP_API_SYMBOLS)
        string(FIND "${ISF_CPP_API_TEST_SOURCE}" "${ISF_CPP_API_SYMBOL}" ISF_CPP_API_TEST_POSITION)
        if(ISF_CPP_API_TEST_POSITION EQUAL -1)
            list(APPEND ISF_UNTESTED_CPP_API_SYMBOLS "${ISF_CPP_API_SYMBOL}")
        endif()
    endforeach()
    if(ISF_UNTESTED_CPP_API_SYMBOLS)
        list(JOIN ISF_UNTESTED_CPP_API_SYMBOLS ", " ISF_UNTESTED_CPP_API_SYMBOLS_TEXT)
        message(FATAL_ERROR "Baseline public C++ APIs without an external test reference: ${ISF_UNTESTED_CPP_API_SYMBOLS_TEXT}")
    endif()
endif()

add_executable(TestCPP ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp.cpp ${TEST_CPP_FILES} ${TEST_COMMON_FILES})
target_link_libraries(TestCPP InspireFace ${DEPEND})
target_include_directories(TestCPP PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/
        ${ISF_THIRD_PARTY_DIR}/spdlog/include
        ${ISF_THIRD_PARTY_DIR}/Catch2/single_include/
        ${ISF_THIRD_PARTY_DIR}/indicators/include/
        ${SRC_DIR}
        )

set_target_properties(TestCPP PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/")

# C++ lifecycle tests use the non-preloading runner for the same reason as the
# C lifecycle target: each case must own Load/Reload/Unload completely.
add_executable(TestCPPLifecycle ${CMAKE_CURRENT_SOURCE_DIR}/test_lifecycle.cpp ${TEST_CPP_LIFECYCLE_FILES} ${TEST_COMMON_FILES})
target_link_libraries(TestCPPLifecycle InspireFace ${DEPEND})
target_include_directories(TestCPPLifecycle PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/
        ${ISF_THIRD_PARTY_DIR}/spdlog/include
        ${ISF_THIRD_PARTY_DIR}/Catch2/single_include/
        ${ISF_THIRD_PARTY_DIR}/indicators/include/
        ${SRC_DIR}
        )

set_target_properties(TestCPPLifecycle PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/")

# Register deterministic test gates with CTest. Benchmarks, stress cases, model
# accuracy checks, and hardware-specific checks remain opt-in Catch2 tags.
include(CTest)
if(WIN32)
    set(ISF_WINDOWS_TEST_TARGETS Test TestLifecycle TestCPP TestCPPLifecycle)
    if(TARGET TestInternal)
        list(APPEND ISF_WINDOWS_TEST_TARGETS TestInternal)
    endif()
    if(ISF_BUILD_SHARED_LIBS)
        # Always refresh the runtime, even when a DLL implementation changes
        # without changing the import library or relinking the test executable.
        add_custom_target(InspireFaceTestRuntime
                COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:Test>"
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                        "$<TARGET_FILE:InspireFace>" "$<TARGET_FILE_DIR:Test>"
                DEPENDS InspireFace
                VERBATIM)
    endif()
    foreach(ISF_TEST_TARGET IN LISTS ISF_WINDOWS_TEST_TARGETS)
        target_compile_definitions(${ISF_TEST_TARGET} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
        if(MSVC)
            # Match the standard C++ value categories used by GCC/Clang.
            target_compile_options(${ISF_TEST_TARGET} PRIVATE /permissive- /utf-8)
        endif()
        target_link_libraries(${ISF_TEST_TARGET} Psapi)
        if(ISF_BUILD_SHARED_LIBS)
            add_dependencies(${ISF_TEST_TARGET} InspireFaceTestRuntime)
        endif()
    endforeach()
endif()
if(BUILD_TESTING)
    set(ISF_TEST_RESOURCE_ARGS
            --test_dir "${CMAKE_SOURCE_DIR}/test_res"
            --pack_path "${CMAKE_SOURCE_DIR}/test_res/pack/Pikachu")

    add_test(NAME InspireFace.CAPI.Contract
            COMMAND Test ${ISF_TEST_RESOURCE_ARGS} "~[benchmark]~[stress]~[model_accuracy]~[platform]")
    add_test(NAME InspireFace.CAPI.Lifecycle
            COMMAND TestLifecycle ${ISF_TEST_RESOURCE_ARGS})
    add_test(NAME InspireFace.CPPAPI.Contract
            COMMAND TestCPP ${ISF_TEST_RESOURCE_ARGS} "~[benchmark]~[stress]~[model_accuracy]~[platform]~[performance]")
    add_test(NAME InspireFace.CPPAPI.Lifecycle
            COMMAND TestCPPLifecycle ${ISF_TEST_RESOURCE_ARGS})
    add_test(NAME InspireFace.CPPAPI.Performance
            COMMAND TestCPP ${ISF_TEST_RESOURCE_ARGS} "[performance]")
    add_test(NAME InspireFace.JNI.FaceCaptureContract
            COMMAND "${CMAKE_COMMAND}"
                    "-DISF_SOURCE_ROOT=${CMAKE_SOURCE_DIR}"
                    -P "${SRC_DIR}/inspireface/platform/jni/android/verify_face_capture_jni.cmake")

    set_tests_properties(
            InspireFace.CAPI.Contract
            InspireFace.CAPI.Lifecycle
            InspireFace.CPPAPI.Contract
            InspireFace.CPPAPI.Lifecycle
            InspireFace.JNI.FaceCaptureContract
            PROPERTIES
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/test"
            LABELS "unit;contract")
    set_tests_properties(InspireFace.CPPAPI.Performance PROPERTIES
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/test"
            LABELS "performance;consistency")
endif()

# Print Message
message(STATUS ">>>>>>>>>>>>>")
message(STATUS "InspireFace Test:")
message(STATUS "\t ISF_ENABLE_BENCHMARK: ${ISF_ENABLE_BENCHMARK}")
message(STATUS "\t ISF_ENABLE_USE_LFW_DATA: ${ISF_ENABLE_USE_LFW_DATA}")
message(STATUS "\t ISF_ENABLE_TEST_EVALUATION: ${ISF_ENABLE_TEST_EVALUATION}")

# Install bin
install(TARGETS Test RUNTIME DESTINATION ${ISF_INSTALL_ROOT}/test)
if(WIN32 AND ISF_BUILD_SHARED_LIBS)
    install(FILES "$<TARGET_FILE:InspireFace>" DESTINATION ${ISF_INSTALL_ROOT}/test)
endif()
# install(TARGETS TestSource RUNTIME DESTINATION ${ISF_INSTALL_ROOT}/test)
