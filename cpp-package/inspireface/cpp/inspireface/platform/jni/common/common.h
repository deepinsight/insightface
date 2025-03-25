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
    char *rtn = NULL;
    jclass clsstring = env->FindClass("java/lang/String");
    jstring strencode = env->NewStringUTF("GB2312");
    jmethodID mid = env->GetMethodID(clsstring, "getBytes", "(Ljava/lang/String;)[B");
    jbyteArray barr = (jbyteArray)env->CallObjectMethod(jstr, mid, strencode);
    jsize alen = env->GetArrayLength(barr);
    jbyte *ba = env->GetByteArrayElements(barr, JNI_FALSE);
    if (alen > 0) {
        rtn = (char *)malloc(alen + 1);
        memcpy(rtn, ba, alen);
        rtn[alen] = 0;
    }
    env->ReleaseByteArrayElements(barr, ba, 0);
    std::string stemp(rtn);
    free(rtn);
    return stemp;
}

#endif  // INSPIRE_FACE_JNI_COMMON_H
