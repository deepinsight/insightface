LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := inspireface_session_bridge
LOCAL_SRC_FILES := inspireface_session_bridge.cpp
LOCAL_CPPFLAGS := -std=c++17 -Wall -Wextra
LOCAL_LDLIBS := -ldl -llog
include $(BUILD_SHARED_LIBRARY)
