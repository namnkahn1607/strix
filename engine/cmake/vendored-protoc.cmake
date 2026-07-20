# cmake/vendored-protoc.cmake

execute_process(
    COMMAND git rev-parse --show-toplevel
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE STRIX_REPO_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _strix_git_result
)
if(NOT _strix_git_result EQUAL 0)
    message(FATAL_ERROR "[vendored-protoc] Could not resolve repository root.")
endif()

# ==============================================================================
# 1. Read expected protoc version
# ==============================================================================
set(_strix_protoc_version_file "${STRIX_REPO_ROOT}/.protoc-version")
if (NOT EXISTS "${_strix_protoc_version_file}")
    message(FATAL_ERROR 
        "[vendored-protoc] ${_strix_protoc_version_file} not found. "
        "Run 'bash scripts/install/protoc.sh' from the repository root first.")
endif()

# ==============================================================================
# 2. Locate the vendored binary
# ==============================================================================
file(READ "${_strix_protoc_version_file}" _strix_protoc_expected_version)
string(STRIP "${_strix_protoc_expected_version}" _strix_protoc_expected_version)

find_program(PROTOC_EXECUTABLE protoc
    PATHS "${STRIX_REPO_ROOT}/tools/protoc/bin"
    NO_DEFAULT_PATH # No fallback to system protoc
)
if (NOT PROTOC_EXECUTABLE)
    message(FATAL_ERROR
        "[vendored-protoc] Binary executable not found at ${STRIX_REPO_ROOT}/tools/protoc/bin. "
        "Run 'bash scripts/install/protoc.sh' from the repository root first.")
endif()

# ==============================================================================
# 3. Extract the actual protoc version
# ==============================================================================
execute_process(
    COMMAND ${PROTOC_EXECUTABLE} --version
    OUTPUT_VARIABLE _strix_protoc_version_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" _strix_protoc_actual_version "${_strix_protoc_version_output}")

# ==============================================================================
# 4. Detech version mismatch
# ==============================================================================
if(NOT _strix_protoc_actual_version STREQUAL _strix_protoc_expected_version)
    message(FATAL_ERROR
        "[vendored-protoc] Version mismatch. Binary at ${PROTOC_EXECUTABLE} reports "
        "'${_strix_protoc_actual_version}', but ${_strix_protoc_version_file} pins "
        "'${_strix_protoc_expected_version}'. This usually means builtin-baseline in "
        "vcpkg.json changed without re-running 'bash scripts/install/protoc.sh'. "
        "Re-run it, then reconfigure.")
endif()
 
message(STATUS 
    "[vendored-protoc] Using protoc ${_strix_protoc_actual_version} at ${PROTOC_EXECUTABLE}")
