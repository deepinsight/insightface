cmake_minimum_required(VERSION 3.10)
project(InspireFaceSample)

option(ISF_BUILD_SAMPLE_CLUTTERED "Whether to compile the cluttered sample program (debug code during development)" OFF)

include_directories(${SRC_DIR})

if (ISF_ENABLE_RKNN)
        set(ISF_RKNN_API_LIB ${ISF_THIRD_PARTY_DIR}/inspireface-precompile/rknn/${ISF_RKNPU_MAJOR}/runtime/${ISF_RK_DEVICE_TYPE}/Linux/librknn_api/${CPU_ARCH}/)
    link_directories(${ISF_RKNN_API_LIB})
    set(ext rknn_api dl)
endif ()

add_executable(Leak cpp/leak.cpp)
target_link_libraries(Leak InspireFace ${ext})
set_target_properties(Leak PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
)

# Examples of face detection and tracking
add_executable(FaceTrackSample cpp/sample_face_track.cpp)
target_link_libraries(FaceTrackSample InspireFace ${ext})
set_target_properties(FaceTrackSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
)

add_executable(FaceTrackSampleCost cpp/sample_face_track_cost.cpp)
target_link_libraries(FaceTrackSampleCost InspireFace ${ext})
set_target_properties(FaceTrackSampleCost PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
)


# Examples of face detection and tracking
add_executable(MTFaceTrackSample cpp/sample_face_track_mt.cpp)
target_link_libraries(MTFaceTrackSample InspireFace ${ext})
set_target_properties(MTFaceTrackSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
)

if(NOT DISABLE_GUI)
        # Examples of face detection and tracking
        add_executable(FaceTrackVideoSample cpp/sample_face_track_video.cpp)
        target_link_libraries(FaceTrackVideoSample InspireFace ${ext})
        set_target_properties(FaceTrackVideoSample PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
        )
endif()



# Examples of face recognition
add_executable(FaceRecognitionSample cpp/sample_face_recognition.cpp)
target_link_libraries(FaceRecognitionSample InspireFace ${ext})
set_target_properties(FaceRecognitionSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
)

add_executable(FaceSearchSample cpp/sample_face_search.cpp)
target_link_libraries(FaceSearchSample InspireFace ${ext})
set_target_properties(FaceSearchSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
)


# Examples of face recognition
add_executable(FaceComparisonSample cpp/sample_face_comparison.cpp)
target_link_libraries(FaceComparisonSample InspireFace ${ext})
set_target_properties(FaceComparisonSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
)


# Examples of face recognition
add_executable(FaceDetect cpp/face_detect.cpp)
target_link_libraries(FaceDetect InspireFace ${ext})
set_target_properties(FaceDetect PROPERTIES
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
message(STATUS "InspireFace Sample:")
message(STATUS "\t ISF_BUILD_SAMPLE_CLUTTERED: ${ISF_BUILD_SAMPLE_CLUTTERED}")

# Install bin
install(TARGETS Leak RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/sample)
install(TARGETS FaceTrackSample RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/sample)
install(TARGETS FaceTrackSampleCost RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/sample)
install(TARGETS MTFaceTrackSample RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/sample)
install(TARGETS FaceRecognitionSample RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/sample)
install(TARGETS FaceSearchSample RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/sample)
install(TARGETS FaceComparisonSample RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/sample)

