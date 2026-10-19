if(NOT DEFINED ISF_READELF_EXECUTABLE OR ISF_READELF_EXECUTABLE STREQUAL "")
    message(FATAL_ERROR "ISF_READELF_EXECUTABLE is required")
endif()

if(NOT DEFINED ISF_ELF_FILE OR ISF_ELF_FILE STREQUAL "")
    message(FATAL_ERROR "ISF_ELF_FILE is required")
endif()

if(NOT EXISTS "${ISF_ELF_FILE}")
    message(FATAL_ERROR "ELF file does not exist: ${ISF_ELF_FILE}")
endif()

execute_process(
        COMMAND "${ISF_READELF_EXECUTABLE}" -W -l "${ISF_ELF_FILE}"
        RESULT_VARIABLE ISF_READELF_RESULT
        OUTPUT_VARIABLE ISF_READELF_OUTPUT
        ERROR_VARIABLE ISF_READELF_ERROR)

if(NOT ISF_READELF_RESULT EQUAL 0)
    message(FATAL_ERROR
            "Unable to inspect ${ISF_ELF_FILE}: ${ISF_READELF_ERROR}")
endif()

string(REGEX MATCH "GNU_STACK[^\r\n]*" ISF_GNU_STACK_LINE "${ISF_READELF_OUTPUT}")
if(ISF_GNU_STACK_LINE STREQUAL "")
    message(FATAL_ERROR
            "Missing GNU_STACK program header in ${ISF_ELF_FILE}")
endif()

# GNU readelf prints executable stack permissions as RWE (or as a separated E
# flag on some versions). GNU_STACK itself contains no letter E, so checking
# the extracted program-header line is stable across those formats.
if(ISF_GNU_STACK_LINE MATCHES "E")
    message(FATAL_ERROR
            "Executable stack detected in ${ISF_ELF_FILE}: ${ISF_GNU_STACK_LINE}")
endif()

message(STATUS "Verified non-executable stack: ${ISF_GNU_STACK_LINE}")
