if(NOT DEFINED ISF_READELF_EXECUTABLE OR ISF_READELF_EXECUTABLE STREQUAL "")
    message(FATAL_ERROR "ISF_READELF_EXECUTABLE is required")
endif()
if(NOT DEFINED ISF_NAPI_LIBRARY OR NOT EXISTS "${ISF_NAPI_LIBRARY}")
    message(FATAL_ERROR "Node-API library does not exist: ${ISF_NAPI_LIBRARY}")
endif()

execute_process(
        COMMAND "${ISF_READELF_EXECUTABLE}" -h -d -W -l --dyn-syms "${ISF_NAPI_LIBRARY}"
        RESULT_VARIABLE ISF_READELF_RESULT
        OUTPUT_VARIABLE ISF_READELF_OUTPUT
        ERROR_VARIABLE ISF_READELF_ERROR)
if(NOT ISF_READELF_RESULT EQUAL 0)
    message(FATAL_ERROR "Unable to inspect ${ISF_NAPI_LIBRARY}: ${ISF_READELF_ERROR}")
endif()

if(NOT ISF_READELF_OUTPUT MATCHES "Machine:[ ]+AArch64")
    message(FATAL_ERROR "HarmonyOS Node-API library is not AArch64")
endif()
if(NOT ISF_READELF_OUTPUT MATCHES "Shared library: \\[libace_napi\\.z\\.so\\]")
    message(FATAL_ERROR "HarmonyOS Node-API dependency libace_napi.z.so is missing")
endif()
if(NOT ISF_READELF_OUTPUT MATCHES "napi_module_register")
    message(FATAL_ERROR "HarmonyOS Node-API module registration is missing")
endif()
if(ISF_READELF_OUTPUT MATCHES "Shared library: \\[liblog\\.so\\]" OR
        ISF_READELF_OUTPUT MATCHES "Shared library: \\[libjnigraphics\\.so\\]")
    message(FATAL_ERROR "Android-only dependencies were linked into the HarmonyOS package")
endif()
if(ISF_READELF_OUTPUT MATCHES "GLOBAL[^\r\n]*[ \t]HF[A-Za-z0-9_]+([\r\n]|$)")
    message(FATAL_ERROR "HarmonyOS Node-API library leaks the core C API through its dynamic ABI")
endif()

string(REGEX MATCH "GNU_STACK[^\r\n]*" ISF_GNU_STACK_LINE "${ISF_READELF_OUTPUT}")
if(ISF_GNU_STACK_LINE STREQUAL "" OR ISF_GNU_STACK_LINE MATCHES "E")
    message(FATAL_ERROR "Invalid executable-stack state: ${ISF_GNU_STACK_LINE}")
endif()

message(STATUS "Verified HarmonyOS Node-API library: AArch64, registered, hidden core ABI, no Android dependencies, non-executable stack")
