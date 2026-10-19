/**
 * Created by Jingyu Yan
 * @date 2024-11-26
 */

#ifndef INSPIRE_FACE_JNI_COMMON_H
#define INSPIRE_FACE_JNI_COMMON_H

#include <jni.h>
#include <string>

/**
 * @brief Convert jstring to std::string
 * @param env JNIEnv pointer
 * @param jstr jstring object
 * @return std::string
 */
inline std::string jstring2str(JNIEnv *env, jstring jstr) {
    if (!env || !jstr) {
        return {};
    }
    const char *characters = env->GetStringUTFChars(jstr, nullptr);
    if (!characters) {
        return {};
    }
    std::string result(characters);
    env->ReleaseStringUTFChars(jstr, characters);
    return result;
}

#endif  // INSPIRE_FACE_JNI_COMMON_H
