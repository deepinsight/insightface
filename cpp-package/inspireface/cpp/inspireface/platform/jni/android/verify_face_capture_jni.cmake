if(NOT DEFINED ISF_SOURCE_ROOT)
    message(FATAL_ERROR "ISF_SOURCE_ROOT is required")
endif()

set(ISF_JNI_SOURCE
        "${ISF_SOURCE_ROOT}/cpp/inspireface/platform/jni/android/face_capture_jni.cpp")
set(ISF_JAVA_ROOT
        "${ISF_SOURCE_ROOT}/cpp/inspireface/platform/jni/java/com/insightface/sdk/inspireface")
file(READ "${ISF_JNI_SOURCE}" ISF_JNI_TEXT)
file(READ "${ISF_JAVA_ROOT}/FaceCapture.java" ISF_CAPTURE_JAVA_TEXT)
file(READ "${ISF_JAVA_ROOT}/FaceDetectionSnapshot.java" ISF_SNAPSHOT_JAVA_TEXT)
file(READ "${ISF_JAVA_ROOT}/base/FaceCaptureConfig.java" ISF_CONFIG_JAVA_TEXT)
file(READ "${ISF_JAVA_ROOT}/base/FaceCaptureProgress.java" ISF_PROGRESS_JAVA_TEXT)

set(ISF_CAPTURE_NATIVE_METHODS
        nativeGetDefaultConfig
        nativeCreate
        nativeUpdate
        nativeUpdateWithSnapshot
        nativeGetResults
        nativeFinish
        nativeReset
        nativeRelease)
foreach(ISF_METHOD IN LISTS ISF_CAPTURE_NATIVE_METHODS)
    string(FIND "${ISF_CAPTURE_JAVA_TEXT}" "${ISF_METHOD}(" ISF_JAVA_POSITION)
    if(ISF_JAVA_POSITION EQUAL -1)
        message(FATAL_ERROR "Java FaceCapture is missing ${ISF_METHOD}")
    endif()
    string(FIND "${ISF_JNI_TEXT}" "FaceCapture_${ISF_METHOD}" ISF_JNI_POSITION)
    if(ISF_JNI_POSITION EQUAL -1)
        message(FATAL_ERROR "JNI FaceCapture is missing ${ISF_METHOD}")
    endif()
endforeach()

string(FIND "${ISF_PROGRESS_JAVA_TEXT}" " trackCount" ISF_JAVA_TRACK_COUNT_POSITION)
string(FIND "${ISF_JNI_TEXT}" "\"trackCount\", progress.trackCount" ISF_JNI_TRACK_COUNT_POSITION)
if(ISF_JAVA_TRACK_COUNT_POSITION EQUAL -1 OR ISF_JNI_TRACK_COUNT_POSITION EQUAL -1)
    message(FATAL_ERROR "Capture progress trackCount is not bridged")
endif()

foreach(ISF_METHOD IN ITEMS nativeCreate nativeRelease)
    string(FIND "${ISF_SNAPSHOT_JAVA_TEXT}" "${ISF_METHOD}(" ISF_JAVA_POSITION)
    if(ISF_JAVA_POSITION EQUAL -1)
        message(FATAL_ERROR "Java FaceDetectionSnapshot is missing ${ISF_METHOD}")
    endif()
    string(FIND "${ISF_JNI_TEXT}" "FaceDetectionSnapshot_${ISF_METHOD}" ISF_JNI_POSITION)
    if(ISF_JNI_POSITION EQUAL -1)
        message(FATAL_ERROR "JNI FaceDetectionSnapshot is missing ${ISF_METHOD}")
    endif()
endforeach()

set(ISF_REQUIRED_C_APIS
        HFGetDefaultFaceCaptureConfig
        HFCreateFaceCaptureSession
        HFUpdateFaceCaptureSession
        HFUpdateFaceCaptureSessionWithSnapshot
        HFGetFaceCaptureResults
        HFFinishFaceCaptureSession
        HFResetFaceCaptureSession
        HFReleaseFaceCaptureSession
        HFExecuteFaceTrackSnapshot
        HFReleaseFaceResultSnapshot)
foreach(ISF_API IN LISTS ISF_REQUIRED_C_APIS)
    string(FIND "${ISF_JNI_TEXT}" "${ISF_API}(" ISF_API_POSITION)
    if(ISF_API_POSITION EQUAL -1)
        message(FATAL_ERROR "JNI capture adapter does not call ${ISF_API}")
    endif()
endforeach()

set(ISF_CONFIG_FIELDS
        filterMask outputCount minTrackCount stableDurationMs collectDurationMs maxCollectDurationMs
        trackLostGraceMs minCandidateIntervalMs minFaceWidthRatio maxFaceWidthRatio
        maxCenterOffsetX maxCenterOffsetY boundaryMarginRatio maxCenterMotionRatio
        maxSizeChangeRatio maxAbsYaw maxAbsPitch maxAbsRoll minQualityScore
        minSharpnessScore minBrightnessScore maxBrightnessScore)
foreach(ISF_FIELD IN LISTS ISF_CONFIG_FIELDS)
    string(FIND "${ISF_CONFIG_JAVA_TEXT}" " ${ISF_FIELD}" ISF_JAVA_FIELD_POSITION)
    string(FIND "${ISF_JNI_TEXT}" "\"${ISF_FIELD}\"" ISF_JNI_FIELD_POSITION)
    if(ISF_JAVA_FIELD_POSITION EQUAL -1 OR ISF_JNI_FIELD_POSITION EQUAL -1)
        message(FATAL_ERROR "Capture config field is not bridged: ${ISF_FIELD}")
    endif()
endforeach()

foreach(ISF_LIFECYCLE_MARKER IN ITEMS "implements AutoCloseable" "synchronized void close()" "private long nativeHandle")
    string(FIND "${ISF_CAPTURE_JAVA_TEXT}" "${ISF_LIFECYCLE_MARKER}" ISF_MARKER_POSITION)
    if(ISF_MARKER_POSITION EQUAL -1)
        message(FATAL_ERROR "FaceCapture lifecycle contract missing: ${ISF_LIFECYCLE_MARKER}")
    endif()
endforeach()

foreach(ISF_CONSTANT IN ITEMS FILTER_TRACK_COUNT REJECT_TRACK_COUNT_TOO_LOW)
    string(FIND "${ISF_CAPTURE_JAVA_TEXT}" "${ISF_CONSTANT}" ISF_CONSTANT_POSITION)
    if(ISF_CONSTANT_POSITION EQUAL -1)
        message(FATAL_ERROR "Java FaceCapture is missing ${ISF_CONSTANT}")
    endif()
endforeach()

message(STATUS "Verified Java/JNI face capture parity: 10 native methods, 22 config fields, complete lifecycle")
