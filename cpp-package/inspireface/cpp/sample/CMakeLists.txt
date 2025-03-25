cmake_minimum_required(VERSION 3.10)
project(InspireFaceSample)

option(ISF_BUILD_SAMPLE_CLUTTERED "Whether to compile the cluttered sample program (debug code during development)" OFF)

include_directories(${SRC_DIR})
include_directories(${SRC_DIR}/inspireface/c_api)

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

add_executable(Leak api/leak.cpp)
target_link_libraries(Leak InspireFace ${ext})
set_target_properties(Leak PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
)

# # Examples of face detection and tracking
add_executable(FaceTrackSample api/sample_face_track.cpp)
target_link_libraries(FaceTrackSample InspireFace ${ext})
set_target_properties(FaceTrackSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
)

add_executable(FaceTrackBenchmarkSample api/sample_face_track_benchmark.cpp)
target_link_libraries(FaceTrackBenchmarkSample InspireFace ${ext})
set_target_properties(FaceTrackBenchmarkSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
)

# Examples of face recognition
add_executable(FaceComparisonSample api/sample_face_comparison.cpp)
target_link_libraries(FaceComparisonSample InspireFace ${ext})
set_target_properties(FaceComparisonSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
)

add_executable(FaceFeatureHubSample api/sample_feature_hub.cpp)
target_link_libraries(FaceFeatureHubSample InspireFace ${ext})
set_target_properties(FaceFeatureHubSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
)


add_executable(FaceLoadReloadSample api/sample_load_reload.cpp)
target_link_libraries(FaceLoadReloadSample InspireFace ${ext})
set_target_properties(FaceLoadReloadSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
)

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
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
            )

    add_executable(FaceAttributeSampleRV1106 rv1106/face_attribute.cpp)
    target_link_libraries(FaceAttributeSampleRV1106 InspireFace ${ext})
    set_target_properties(FaceAttributeSampleRV1106 PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
            )
endif()

add_executable(NexusImageSample rv1106/rga_image.cpp)
target_link_libraries(NexusImageSample InspireFace ${ext})
set_target_properties(NexusImageSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
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

# Print Message
message(STATUS ">>>>>>>>>>>>>")
message(STATUS "InspireFace Sample:")
message(STATUS "\t ISF_BUILD_SAMPLE_CLUTTERED: ${ISF_BUILD_SAMPLE_CLUTTERED}")

# Install bin
install(TARGETS Leak RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/sample)
install(TARGETS FaceTrackSample RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/sample)
# install(TARGETS FaceTrackSampleCost RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/sample)
# install(TARGETS MTFaceTrackSample RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/sample)
# install(TARGETS FaceRecognitionSample RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/sample)
# install(TARGETS FaceSearchSample RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/sample)
install(TARGETS FaceComparisonSample RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/sample)

