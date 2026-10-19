cmake_minimum_required(VERSION 3.20)
project(InspireFaceSample)

option(ISF_BUILD_SAMPLE_CLUTTERED "Whether to compile the cluttered sample program (debug code during development)" OFF)
option(ISF_BUILD_SAMPLE_INTERNAL "Whether to compile the internal sample program (debug code during development)" OFF)

include_directories(${SRC_DIR})
include_directories(${SRC_DIR}/inspireface/c_api)
include_directories(${SRC_DIR}/inspireface/include)

if (ISF_ENABLE_RKNN AND ISF_RKNPU_MAJOR STREQUAL "rknpu1")
        set(ISF_RKNN_API_LIB ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/rknn/${ISF_RKNPU_MAJOR}/runtime/${ISF_RK_DEVICE_TYPE}/Linux/librknn_api/${CPU_ARCH}/)
    link_directories(${ISF_RKNN_API_LIB})
    set(ext rknn_api dl)
endif ()

if (ISF_ENABLE_RKNN AND ISF_RKNPU_MAJOR STREQUAL "rknpu2" AND ISF_RK_COMPILER_TYPE STREQUAL "aarch64")
    if(ANDROID)
        set(RK_PLATFORM "Android")
    else()
        set(RK_PLATFORM "Linux")
    endif()
    set(ISF_RKNN_API_LIB ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/rknn/${ISF_RKNPU_MAJOR}/runtime/${RK_PLATFORM}/librknn_api/${ISF_RK_COMPILER_TYPE}/)
    message("ISF_RKNN_API_LIB: ${ISF_RKNN_API_LIB}")
    link_directories(${ISF_RKNN_API_LIB})
    set(ext rknnrt dl)
endif ()

add_executable(Leak api/leak.c)
target_link_libraries(Leak InspireFace ${ext})
set_target_properties(Leak PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/api/"
)

# # Examples of face detection and tracking
add_executable(FaceTrackSample api/sample_face_track.c)
target_link_libraries(FaceTrackSample InspireFace ${ext})
set_target_properties(FaceTrackSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/api/"
)

add_executable(FaceTrackBenchmarkSample api/sample_face_track_benchmark.c)
target_link_libraries(FaceTrackBenchmarkSample InspireFace ${ext})
set_target_properties(FaceTrackBenchmarkSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/api/"
)

# Load the shared SDK through dlopen instead of linking it into the process at
# startup. This reproduces the loading path that glibc 2.41 hardened.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND ISF_BUILD_SHARED_LIBS)
        add_executable(LinuxDlopenCompatibilityGuardSample
                benchmark/linux_dlopen_compatibility_guard.cpp)
        target_link_libraries(LinuxDlopenCompatibilityGuardSample PRIVATE ${CMAKE_DL_LIBS})
        add_dependencies(LinuxDlopenCompatibilityGuardSample InspireFace)
        set_target_properties(LinuxDlopenCompatibilityGuardSample PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
        )
        if(BUILD_TESTING)
                add_test(NAME InspireFace.Linux.DlopenCompatibility
                        COMMAND LinuxDlopenCompatibilityGuardSample "$<TARGET_FILE:InspireFace>")
                set_tests_properties(InspireFace.Linux.DlopenCompatibility PROPERTIES
                        LABELS "compatibility;security"
                        PASS_REGULAR_EXPRESSION "load_result=success"
                        TIMEOUT 15)
        endif()
endif()

# End-to-end correctness/performance guard for internal face-detector changes.
# It is a sample target so the same benchmark can run on deployment toolchains.
add_executable(FaceDetectPostprocessBenchmarkSample benchmark/face_detect_postprocess_benchmark.cpp)
target_link_libraries(FaceDetectPostprocessBenchmarkSample InspireFace ${ext})
set_target_properties(FaceDetectPostprocessBenchmarkSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
)

add_executable(FaceTrackNmsBenchmarkSample benchmark/face_track_nms_benchmark.cpp)
target_link_libraries(FaceTrackNmsBenchmarkSample InspireFace ${ext})
set_target_properties(FaceTrackNmsBenchmarkSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
)

add_executable(FaceTrackCandidateBenchmarkSample benchmark/face_track_candidate_benchmark.cpp)
target_link_libraries(FaceTrackCandidateBenchmarkSample InspireFace ${ext})
set_target_properties(FaceTrackCandidateBenchmarkSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
)

add_executable(FaceCaptureSelectorGuardSample benchmark/face_capture_selector_guard.cpp)
target_link_libraries(FaceCaptureSelectorGuardSample InspireFace ${ext})
set_target_properties(FaceCaptureSelectorGuardSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
)
if(BUILD_TESTING)
        add_test(NAME InspireFace.FaceCapture.SelectorGuard
                COMMAND FaceCaptureSelectorGuardSample
                        "${CMAKE_SOURCE_DIR}/test_res/pack/Pikachu"
                        "${CMAKE_SOURCE_DIR}/test_res" 2000)
        set_tests_properties(InspireFace.FaceCapture.SelectorGuard PROPERTIES
                LABELS "capture;accuracy;performance"
                TIMEOUT 60)
endif()

add_executable(FaceLandmarkSmoothingBenchmarkSample benchmark/face_landmark_smoothing_benchmark.cpp)
target_link_libraries(FaceLandmarkSmoothingBenchmarkSample InspireFace ${ext})
set_target_properties(FaceLandmarkSmoothingBenchmarkSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
)

add_executable(FaceActionLogicGuardSample benchmark/face_action_logic_guard.cpp)
target_link_libraries(FaceActionLogicGuardSample InspireFace ${ext})
set_target_properties(FaceActionLogicGuardSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
)

if(NOT ISF_BUILD_SHARED_LIBS)
        add_executable(ImagePreprocessGuardSample benchmark/image_preprocess_guard.cpp)
        target_link_libraries(ImagePreprocessGuardSample InspireFace ${ext})
        set_target_properties(ImagePreprocessGuardSample PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
        )
endif()

add_executable(InternalRuntimeGuardSample
        benchmark/internal_runtime_guard.cpp
        ../inspireface/track_module/tracker_optional/bytetrack/BYTETracker.cpp
        ../inspireface/track_module/tracker_optional/bytetrack/STrack.cpp
        ../inspireface/track_module/tracker_optional/bytetrack/kalmanFilter.cpp
        ../inspireface/track_module/tracker_optional/bytetrack/lapjv.cpp
        ../inspireface/track_module/tracker_optional/bytetrack/tracker_utils.cpp)
target_link_libraries(InternalRuntimeGuardSample InspireFace ${ext})
set_target_properties(InternalRuntimeGuardSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
)

add_executable(SessionInitializationGuardSample
        benchmark/session_initialization_guard.cpp
        ../inspireface/middleware/model_archive/core_archive/microtar/microtar.c)
target_link_libraries(SessionInitializationGuardSample InspireFace ${ext})
set_target_properties(SessionInitializationGuardSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
)

if(NOT ISF_BUILD_SHARED_LIBS)
        add_executable(ArchiveConsistencyGuardSample
                benchmark/archive_consistency_guard.cpp
                ../inspireface/middleware/model_archive/core_archive/microtar/microtar.c)
        target_link_libraries(ArchiveConsistencyGuardSample InspireFace ${ext})
        set_target_properties(ArchiveConsistencyGuardSample PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
        )
        if(BUILD_TESTING)
                add_test(NAME InspireFace.Archive.Consistency COMMAND ArchiveConsistencyGuardSample)
        endif()
endif()

# This guard intentionally inspects internal MNN tensors and cache counters.
# Keep it on the static-library sample build instead of exporting test-only
# implementation symbols from the shared SDK.
if(NOT ISF_BUILD_SHARED_LIBS)
        add_executable(MnnInferenceCacheGuardSample benchmark/mnn_inference_cache_guard.cpp)
        target_link_libraries(MnnInferenceCacheGuardSample InspireFace ${ext})
        set_target_properties(MnnInferenceCacheGuardSample PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
        )

        # FeatureHub's transaction guard also exercises the internal SQLite
        # wrapper, which is intentionally not part of the shared SDK ABI.
        add_executable(FeatureHubSafetyGuardSample benchmark/feature_hub_safety_guard.cpp)
        target_link_libraries(FeatureHubSafetyGuardSample InspireFace ${ext})
        set_target_properties(FeatureHubSafetyGuardSample PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
        )

        add_executable(ResourceManagerGuardSample benchmark/resource_manager_guard.cpp)
        target_link_libraries(ResourceManagerGuardSample InspireFace ${ext})
        set_target_properties(ResourceManagerGuardSample PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
        )

        add_executable(SimilarityConverterGuardSample
                benchmark/similarity_converter_guard.cpp
                ../inspireface/middleware/model_archive/core_archive/microtar/microtar.c)
        target_link_libraries(SimilarityConverterGuardSample InspireFace ${ext})
        set_target_properties(SimilarityConverterGuardSample PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
        )
endif()

add_executable(OutputTensorInfoGuardSample benchmark/output_tensor_info_guard.cpp)
target_link_libraries(OutputTensorInfoGuardSample InspireFace ${ext})
set_target_properties(OutputTensorInfoGuardSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
)

add_executable(RknnOutputLifecycleGuardSample benchmark/rknn_output_lifecycle_guard.cpp)
target_link_libraries(RknnOutputLifecycleGuardSample InspireFace ${ext})
set_target_properties(RknnOutputLifecycleGuardSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
)
if(BUILD_TESTING)
        add_test(NAME InspireFace.RKNN.OutputLifecycle COMMAND RknnOutputLifecycleGuardSample)
endif()

# Exercise the production RKNPU2 adapter against a deterministic host runtime.
# A real RKNN build already compiles this implementation and links the vendor
# runtime, so the host mock is only needed when RKNN itself is disabled.
if(NOT ISF_ENABLE_RKNN)
        add_executable(Rknn1AdapterContractGuardSample
                benchmark/rknn1_adapter_contract_guard.cpp
                ../inspireface/middleware/inference_wrapper/inference_wrapper_rknn_adapter.cpp)
        target_compile_definitions(Rknn1AdapterContractGuardSample PRIVATE INFERENCE_WRAPPER_ENABLE_RKNN=1)
        target_include_directories(Rknn1AdapterContractGuardSample BEFORE PRIVATE
                ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/rknn/rknpu1/runtime/RV1109RV1126/Linux/librknn_api/include)
        target_link_libraries(Rknn1AdapterContractGuardSample InspireFace ${ext})
        set_target_properties(Rknn1AdapterContractGuardSample PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
        )
        if(BUILD_TESTING)
                add_test(NAME InspireFace.RKNN1.AdapterContract COMMAND Rknn1AdapterContractGuardSample)
        endif()

        add_executable(Rknn2AdapterContractGuardSample
                benchmark/rknn2_adapter_contract_guard.cpp
                ../inspireface/middleware/inference_wrapper/inference_wrapper_rknn_adapter_nano.cpp)
        target_compile_definitions(Rknn2AdapterContractGuardSample PRIVATE INFERENCE_WRAPPER_ENABLE_RKNN2=1)
        target_include_directories(Rknn2AdapterContractGuardSample BEFORE PRIVATE
                ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/rknn/rknpu2/runtime/Linux/librknn_api/include)
        target_link_libraries(Rknn2AdapterContractGuardSample InspireFace ${ext})
        set_target_properties(Rknn2AdapterContractGuardSample PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
        )
        if(BUILD_TESTING)
                add_test(NAME InspireFace.RKNN2.AdapterContract COMMAND Rknn2AdapterContractGuardSample)
        endif()
endif()

# Compile the production TensorRT adapter against a deterministic CPU-only
# CUDA/TensorRT double. This keeps ownership and error-path coverage available
# on CI hosts that do not have an NVIDIA driver or TensorRT installation.
add_executable(TensorRTLifecycleGuardSample
        benchmark/tensorrt_lifecycle_guard.cpp
        ../inspireface/middleware/inference_wrapper/tensorrt/tensorrt_adapter.cpp)
target_compile_definitions(TensorRTLifecycleGuardSample PRIVATE ISF_ENABLE_TENSORRT=1)
target_include_directories(TensorRTLifecycleGuardSample BEFORE PRIVATE benchmark/mock_tensorrt)
target_link_libraries(TensorRTLifecycleGuardSample InspireFace ${ext})
set_target_properties(TensorRTLifecycleGuardSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
)
if(BUILD_TESTING)
        add_test(NAME InspireFace.TensorRT.Lifecycle COMMAND TensorRTLifecycleGuardSample)
endif()

add_executable(JniDataLifecycleGuardSample benchmark/jni_data_lifecycle_guard.cpp)
target_link_libraries(JniDataLifecycleGuardSample InspireFace ${ext})
set_target_properties(JniDataLifecycleGuardSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
)
if(BUILD_TESTING)
        add_test(NAME InspireFace.JNI.DataLifecycle COMMAND JniDataLifecycleGuardSample)
endif()

add_executable(OhosNapiContractGuardSample benchmark/ohos_napi_contract_guard.cpp)
target_link_libraries(OhosNapiContractGuardSample InspireFace ${ext})
set_target_properties(OhosNapiContractGuardSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
)
if(BUILD_TESTING)
        add_test(NAME InspireFace.OHOS.NapiContract COMMAND OhosNapiContractGuardSample)
        set_tests_properties(InspireFace.OHOS.NapiContract PROPERTIES
                LABELS "ohos;contract;performance"
                TIMEOUT 30)
endif()

add_executable(FrameProcessContractGuardSample benchmark/frame_process_contract_guard.cpp)
target_link_libraries(FrameProcessContractGuardSample InspireFace ${ext})
set_target_properties(FrameProcessContractGuardSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
)
if(BUILD_TESTING)
        add_test(NAME InspireFace.FrameProcess.Contract COMMAND FrameProcessContractGuardSample)
endif()

# This guard exercises AnyNetAdapter and archive internals that are intentionally
# not exported by the shared SDK. Keep it on the static-library sample build.
if(NOT ISF_BUILD_SHARED_LIBS)
        # LandmarkParam and LaunchLifecycle also exercise inline archive/YAML
        # implementation details that are hidden by the shared-library ABI.
        add_executable(LandmarkStateGuardSample benchmark/landmark_state_guard.cpp)
        target_link_libraries(LandmarkStateGuardSample InspireFace ${ext})
        set_target_properties(LandmarkStateGuardSample PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
        )
        if(BUILD_TESTING)
                add_test(NAME InspireFace.Landmark.State COMMAND LandmarkStateGuardSample)
        endif()

        add_executable(LaunchLifecycleGuardSample benchmark/launch_lifecycle_guard.cpp)
        target_link_libraries(LaunchLifecycleGuardSample InspireFace ${ext})
        set_target_properties(LaunchLifecycleGuardSample PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
        )

        add_executable(AnyNetInitializationGuardSample benchmark/any_net_initialization_guard.cpp)
        target_link_libraries(AnyNetInitializationGuardSample InspireFace ${ext})
        set_target_properties(AnyNetInitializationGuardSample PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
        )

        add_executable(SoftmaxGuardSample benchmark/softmax_guard.cpp)
        target_link_libraries(SoftmaxGuardSample InspireFace ${ext})
        set_target_properties(SoftmaxGuardSample PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/benchmark/"
        )
endif()

# Examples of face recognition
add_executable(FaceComparisonSample api/sample_face_comparison.c)
target_link_libraries(FaceComparisonSample InspireFace ${ext})
set_target_properties(FaceComparisonSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/api/"
)

add_executable(FaceFeatureHubSample api/sample_feature_hub.c)
target_link_libraries(FaceFeatureHubSample InspireFace ${ext})
set_target_properties(FaceFeatureHubSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/api/"
)


add_executable(FaceLoadReloadSample api/sample_load_reload.c)
target_link_libraries(FaceLoadReloadSample InspireFace ${ext})
set_target_properties(FaceLoadReloadSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/api/"
)

add_executable(CppResourcePoolSample api/sample_cpp_resource_pool.cpp)
target_link_libraries(CppResourcePoolSample InspireFace ${ext})
set_target_properties(CppResourcePoolSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/api/"
)

add_executable(FaceCrudSample api/sample_face_crud.c)
target_link_libraries(FaceCrudSample InspireFace ${ext})
set_target_properties(FaceCrudSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/api/"
)

add_executable(FeatureHubPersistenceSample api/sample_feature_hub_persistence.c)
target_link_libraries(FeatureHubPersistenceSample InspireFace ${ext})
set_target_properties(FeatureHubPersistenceSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/api/"
)

add_executable(CPUWatchingSample api/sample_cpu_watching.c)
target_link_libraries(CPUWatchingSample InspireFace ${ext})
set_target_properties(CPUWatchingSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/api/"
)

add_executable(CreateSessionSample api/sample_create_session.c)
target_link_libraries(CreateSessionSample InspireFace ${ext})
set_target_properties(CreateSessionSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/api/"
)

add_executable(GrayInput api/sample_gray_input.c)
target_link_libraries(GrayInput InspireFace ${ext})
set_target_properties(GrayInput PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/api/"
)

add_executable(SearchFaceSample api/sample_search_face.cpp)
target_link_libraries(SearchFaceSample InspireFace ${ext})
set_target_properties(SearchFaceSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/api/"
)

add_executable(CppMultithreadingSample api/sample_cpp_multithreading.cpp)
target_link_libraries(CppMultithreadingSample InspireFace ${ext})
set_target_properties(CppMultithreadingSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/api/"
)

# --- C++ API ---

add_executable(CppSessionSample cpp_api/cpp_sample_face_track.cpp)
target_link_libraries(CppSessionSample InspireFace ${ext})
set_target_properties(CppSessionSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cpp_api/"
)

add_executable(CppFaceComparisonSample cpp_api/cpp_sample_face_comparison.cpp)
target_link_libraries(CppFaceComparisonSample InspireFace ${ext})
set_target_properties(CppFaceComparisonSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cpp_api/"
)

add_executable(CppFaceCrudSample cpp_api/cpp_sample_face_crud.cpp)
target_link_libraries(CppFaceCrudSample InspireFace ${ext})
set_target_properties(CppFaceCrudSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cpp_api/"
)

add_executable(CppSampleAffine cpp_api/cpp_sample_affine.cpp)
target_link_libraries(CppSampleAffine InspireFace ${ext})
set_target_properties(CppSampleAffine PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cpp_api/"
)

add_executable(CppSampleInspireCV cpp_api/cpp_sample_inspirecv.cpp)
target_link_libraries(CppSampleInspireCV InspireFace ${ext})
set_target_properties(CppSampleInspireCV PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cpp_api/"
)


add_executable(CppSampleGrayInput cpp_api/cpp_gray_input.cpp)
target_link_libraries(CppSampleGrayInput InspireFace ${ext})
set_target_properties(CppSampleGrayInput PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cpp_api/"
)

add_executable(CppNexusProcessSample cpp_api/cpp_nexus_process.cpp)
target_link_libraries(CppNexusProcessSample InspireFace ${ext})
set_target_properties(CppNexusProcessSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cpp_api/"
)


if(ISF_BUILD_SAMPLE_INTERNAL)
        add_executable(FaceTrackerSample source/tracker_sample.cpp)
        target_link_libraries(FaceTrackerSample InspireFace ${ext})
        set_target_properties(FaceTrackerSample PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
        )

        add_executable(ExpansionLoadSample source/expansion_load.cpp)
        target_link_libraries(ExpansionLoadSample InspireFace ${ext})
        set_target_properties(ExpansionLoadSample PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
        )

        add_executable(FaceTrackPipelineSample source/tracker_pipeline.cpp)
        target_link_libraries(FaceTrackPipelineSample InspireFace ${ext})
        set_target_properties(FaceTrackPipelineSample PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
        )

        add_executable(FeatureHubSample source/feature_hub_sample.cpp)
        target_link_libraries(FeatureHubSample InspireFace ${ext})
        set_target_properties(FeatureHubSample PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
        )

        add_executable(LandmarkSample source/landmark_sample.cpp)
        target_link_libraries(LandmarkSample InspireFace ${ext})
        set_target_properties(LandmarkSample PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
        )

endif()

# Platform watershed
if (ISF_BUILD_LINUX_ARM7 OR ISF_BUILD_LINUX_AARCH64)
    # Typically this is an embedded system or development board scenario where some GUI-related functions are not supported
elseif(ANDROID)
    # The executable program on the Android platform generally refers to the running on the shell
else()
    # Usually x86 linux or macos

endif ()

if(ISF_RK_DEVICE_TYPE STREQUAL "RV1106")
    add_executable(FaceTrackSampleRV1106 rv1106/face_detect.cpp)
    target_link_libraries(FaceTrackSampleRV1106 InspireFace ${ext})
    set_target_properties(FaceTrackSampleRV1106 PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/internal/"
            )

    add_executable(FaceAttributeSampleRV1106 rv1106/face_attribute.cpp)
    target_link_libraries(FaceAttributeSampleRV1106 InspireFace ${ext})
    set_target_properties(FaceAttributeSampleRV1106 PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/internal/"
            )

add_executable(NexusImageSample rv1106/rga_image.cpp)
target_link_libraries(NexusImageSample InspireFace ${ext})
    set_target_properties(NexusImageSample PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/internal/"
            )

        # These sample programs are debugging and testing code left behind by developers during the development process.
        # They are cluttered and have not been organized, or similar functionalities have already been organized in the standard samples.
        # You can ignore them.
        if (ISF_BUILD_SAMPLE_CLUTTERED)
        if (NOT ISF_BUILD_LINUX_ARM7 AND NOT ISF_BUILD_LINUX_AARCH64)

                # =======================InspireFace Sample===========================
                add_executable(TrackerSample cluttered/standard/tracker_sample.cpp)
                target_link_libraries(TrackerSample InspireFace)

                set_target_properties(TrackerSample PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cluttered/"
                        )


                add_executable(ContextSample cluttered/standard/context_sample.cpp)
                target_link_libraries(ContextSample InspireFace)

                set_target_properties(ContextSample PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cluttered/"
                        )


                add_executable(TestSample cluttered/standard/test_sample.cpp)
                target_link_libraries(TestSample InspireFace)

                set_target_properties(TestSample PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cluttered/"
                        )

                add_executable(NetSample cluttered/standard/net_sample.cpp)
                target_link_libraries(NetSample InspireFace)

                set_target_properties(NetSample PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cluttered/"
                        )


                add_executable(RecSample cluttered/standard/rec_sample.cpp)
                target_link_libraries(RecSample InspireFace)

                set_target_properties(RecSample PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cluttered/"
                        )


                add_executable(BMSample cluttered/standard/bm_sample.cpp)
                target_link_libraries(BMSample InspireFace)

                set_target_properties(BMSample PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cluttered/"
                        )


        else()

                # =======================RK Temporary test category===========================

                if (ISF_ENABLE_RKNN)
                set(ISF_RKNN_API_LIB ${ISF_THIRD_PARTY_DIR}/${ISF_RKNPU_MAJOR}/runtime/${ISF_RK_DEVICE_TYPE}/Linux/librknn_api/${CPU_ARCH}/)
                message("Enable RKNN Inference")
                link_directories(${ISF_RKNN_API_LIB})

                # Face detection
                add_executable(RKFaceDetSample cluttered/rk_sample/rk_face_det_sample.cpp)
                target_link_libraries(RKFaceDetSample InspireFace rknn_api dl)

                set_target_properties(RKFaceDetSample PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cluttered/"
                        )

                # Simple network test
                add_executable(RKSimpleNetSample cluttered/rk_sample/rk_simple_net_sample.cpp)
                target_link_libraries(RKSimpleNetSample InspireFace rknn_api dl)

                set_target_properties(RKSimpleNetSample PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cluttered/"
                        )

                # Face recognize
                add_executable(RKFaceRecSample cluttered/rk_sample/rk_face_recognize_sample.cpp)
                target_link_libraries(RKFaceRecSample InspireFace rknn_api dl)

                set_target_properties(RKFaceRecSample PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cluttered/"
                        )

                # Tracking module
                add_executable(RKTrackerSample cluttered/rk_sample/rk_tracker_sample.cpp)
                target_link_libraries(RKTrackerSample InspireFace rknn_api dl)

                set_target_properties(RKTrackerSample PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cluttered/"
                        )

                # Debug
                add_executable(DebugRKRec cluttered/rk_sample/debug_rk_rec.cpp)
                target_link_libraries(DebugRKRec InspireFace rknn_api dl)

                set_target_properties(DebugRKRec PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cluttered/"
                        )


                add_executable(ArchTest cluttered/standard/archive_test.cpp)
                target_link_libraries(ArchTest InspireFace)

                set_target_properties(ArchTest PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cluttered/"
                        )

                endif()

        endif()


        # Tracking module
        add_executable(SQLiteTest cluttered/standard/test_sqlite_sample.cpp)
        target_link_libraries(SQLiteTest InspireFace)

        set_target_properties(SQLiteTest PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cluttered/"
                )

        if (ISF_ENABLE_RKNN)
                set(DEPEND rknn_api dl)
        endif ()

        # C_API Demo
        add_executable(CAPISample cluttered/standard/c_api_sample.cpp)
        target_link_libraries(CAPISample InspireFace ${DEPEND})

        set_target_properties(CAPISample PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cluttered/"
                )

        # C_API Demo
        add_executable(LoopTracker cluttered/standard/loop_tracker.cpp)
        target_link_libraries(LoopTracker InspireFace ${DEPEND})

        set_target_properties(LoopTracker PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cluttered/"
                )

        add_executable(ArchTracker cluttered/standard/archive_tracker.cpp)
        target_link_libraries(ArchTracker InspireFace)

        set_target_properties(ArchTracker PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cluttered/"
                )

        add_executable(ErrorTest cluttered/standard/error_test.cpp)
        target_link_libraries(ErrorTest InspireFace)

        set_target_properties(ErrorTest PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/cluttered/"
                )
        endif ()

endif()

if(WIN32)
    get_property(ISF_WINDOWS_SAMPLE_TARGETS DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    set(ISF_SAMPLE_RUNTIME_COMMANDS)
    set(ISF_SAMPLE_RUNTIME_DIRECTORIES)
    foreach(ISF_SAMPLE_TARGET IN LISTS ISF_WINDOWS_SAMPLE_TARGETS)
        target_compile_definitions(${ISF_SAMPLE_TARGET} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
        if(MSVC)
            target_compile_options(${ISF_SAMPLE_TARGET} PRIVATE /permissive- /utf-8)
        endif()
        if(ISF_BUILD_SHARED_LIBS)
            get_target_property(ISF_SAMPLE_DIRECTORY ${ISF_SAMPLE_TARGET} RUNTIME_OUTPUT_DIRECTORY)
            if(NOT ISF_SAMPLE_DIRECTORY IN_LIST ISF_SAMPLE_RUNTIME_DIRECTORIES)
                list(APPEND ISF_SAMPLE_RUNTIME_DIRECTORIES "${ISF_SAMPLE_DIRECTORY}")
                list(APPEND ISF_SAMPLE_RUNTIME_COMMANDS
                        COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:${ISF_SAMPLE_TARGET}>"
                        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                                "$<TARGET_FILE:InspireFace>" "$<TARGET_FILE_DIR:${ISF_SAMPLE_TARGET}>")
            endif()
        endif()
    endforeach()
    if(ISF_BUILD_SHARED_LIBS)
        add_custom_target(InspireFaceSampleRuntime ${ISF_SAMPLE_RUNTIME_COMMANDS}
                DEPENDS InspireFace VERBATIM)
        foreach(ISF_SAMPLE_TARGET IN LISTS ISF_WINDOWS_SAMPLE_TARGETS)
            add_dependencies(${ISF_SAMPLE_TARGET} InspireFaceSampleRuntime)
        endforeach()
        install(FILES "$<TARGET_FILE:InspireFace>" DESTINATION ${ISF_INSTALL_ROOT}/sample)
    endif()
endif()

# Print Message
message(STATUS ">>>>>>>>>>>>>")
message(STATUS "InspireFace Sample:")
message(STATUS "\t ISF_BUILD_SAMPLE_CLUTTERED: ${ISF_BUILD_SAMPLE_CLUTTERED}")

# Install c bin
install(TARGETS Leak RUNTIME DESTINATION ${ISF_INSTALL_ROOT}/sample)
install(TARGETS FaceTrackSample RUNTIME DESTINATION ${ISF_INSTALL_ROOT}/sample)
install(TARGETS FaceComparisonSample RUNTIME DESTINATION ${ISF_INSTALL_ROOT}/sample)
install(TARGETS FaceCrudSample RUNTIME DESTINATION ${ISF_INSTALL_ROOT}/sample)

# Install cpp bin
install(TARGETS CppSessionSample RUNTIME DESTINATION ${ISF_INSTALL_ROOT}/sample)
install(TARGETS CppFaceComparisonSample RUNTIME DESTINATION ${ISF_INSTALL_ROOT}/sample)
install(TARGETS CppFaceCrudSample RUNTIME DESTINATION ${ISF_INSTALL_ROOT}/sample)
