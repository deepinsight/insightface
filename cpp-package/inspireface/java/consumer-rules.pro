# Native symbols and descriptors are resolved by name from JNI.
-keep class com.insightface.sdk.inspireface.jni.Native { *; }
-keep class com.insightface.sdk.inspireface.jni.CPUEngine { native <methods>; }
-keep class com.insightface.sdk.inspireface.jni.NativeTypes$* { *; }
