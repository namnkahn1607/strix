cmake_minimum_required(VERSION 3.22)
project(strix_engine VERSION 1.0)

enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/vendored-protoc.cmake)

# ==============================================================================
# 0. SHARED PUBLIC HEADER
# Single root for all module's public API: include/<module>/*.h
# ==============================================================================
set(PUBLIC_HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include")

# ==============================================================================
# 1. ORT LIBRARY
# ORT is vendored as a pre-built shared library under ort/, bypassing vcpkg.
# The library is compiled DYNAMICALLY using clang++-17.
# ==============================================================================
set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/ort")
add_library(ONNXRuntime SHARED IMPORTED)
set_target_properties(ONNXRuntime PROPERTIES
    IMPORTED_LOCATION "${ORT_ROOT}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include"
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${ORT_ROOT}/include"
)

# ==============================================================================
# 2. COMPILER OPTIONS
# Any module linking with this target will inherits its compile flags.
# ==============================================================================
add_library(engine_options INTERFACE)
target_compile_options(engine_options INTERFACE
    -march=x86-64-v3
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wunreachable-code
    -include "${CMAKE_CURRENT_SOURCE_DIR}/src/config/lockfree_invariants.h"
)

# ==============================================================================
# 3. AUTO-GENERATE CONTRACTS
# ==============================================================================
set(PROTO_DIR "${STRIX_REPO_ROOT}/api/proto/v1")
set(PROTO_FILE "${PROTO_DIR}/strix.proto")
set(PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Pre-make directory in case of non-existing
file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

# Get vendored protoc and grpc_cpp_plugin from vcpkg
set(PROTOC_EXEC "${PROTOC_EXECUTABLE}")
set(GRPC_PLUGIN_EXEC "$<TARGET_FILE:gRPC::grpc_cpp_plugin>")

# Execute command to generate contracts
add_custom_command(
    OUTPUT
        "${PROTO_GEN_DIR}/strix.pb.cc"
        "${PROTO_GEN_DIR}/strix.pb.h"
        "${PROTO_GEN_DIR}/strix.grpc.pb.cc"
        "${PROTO_GEN_DIR}/strix.grpc.pb.h"
    COMMAND ${PROTOC_EXEC}
    ARGS
        --grpc_out="${PROTO_GEN_DIR}"
        --cpp_out="${PROTO_GEN_DIR}"
        --plugin=protoc-gen-grpc="${GRPC_PLUGIN_EXEC}"
        -I "${PROTO_DIR}"
        "${PROTO_FILE}"
    DEPENDS
        "${PROTO_FILE}"
        "${PROTOC_EXEC}"
        gRPC::grpc_cpp_plugin
    COMMENT "Generating C++ gRPC stubs from strix.proto..."
)

# Package generated contracts into a static independent target
add_library(engine_contract STATIC
    "${PROTO_GEN_DIR}/strix.pb.cc"
    "${PROTO_GEN_DIR}/strix.grpc.pb.cc"
)
target_link_libraries(engine_contract PUBLIC gRPC::grpc++ protobuf::libprotobuf)
target_include_directories(engine_contract PUBLIC "${PROTO_GEN_DIR}")

# ==============================================================================
# 4. DATA PLANE MODULES
#   include/<module>/*.h : Public API
#   src/<module>/*.h     : Private API 
# ==============================================================================
set(ENGINE_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Module 1: Common
add_library(engine_common INTERFACE)
target_include_directories(engine_common INTERFACE ${PUBLIC_HEADER_ROOT})
target_link_libraries(engine_common INTERFACE engine_options)

# Module 2: Memory
add_library(engine_memory STATIC src/memory/memory_arena.cc)
target_include_directories(engine_memory
    PUBLIC ${PUBLIC_HEADER_ROOT}
    PRIVATE ${ENGINE_SOURCE_ROOT}/memory
)
target_link_libraries(engine_memory PRIVATE engine_options engine_common)

# Module 3: Inference
add_library(engine_inference STATIC src/inference/inference_model.cc)
target_include_directories(engine_inference
    PUBLIC ${PUBLIC_HEADER_ROOT}
    PRIVATE ${ENGINE_SOURCE_ROOT}/inference
)
target_link_libraries(engine_inference PRIVATE engine_options engine_common)
target_link_libraries(engine_inference PUBLIC ONNXRuntime)

# Module 4: Index
add_library(engine_index STATIC
    src/index/free_list.cc
    src/index/level0_ring.cc
    src/index/level1_ivf.cc
    src/index/recalibration.cc
    src/index/vector_index.cc
)
target_include_directories(engine_index
    PUBLIC ${PUBLIC_HEADER_ROOT}
    PRIVATE ${ENGINE_SOURCE_ROOT}/index
)
target_link_libraries(engine_index PRIVATE 
    engine_options
    engine_common
    engine_memory
)

# Module 5: Network
add_library(engine_service STATIC src/rpc/cache_service.cc)
target_include_directories(engine_service
    PUBLIC ${PUBLIC_HEADER_ROOT}
    PRIVATE ${ENGINE_SOURCE_ROOT}/rpc
)
target_link_libraries(engine_service PRIVATE 
    engine_options
    engine_common
    engine_memory
    engine_inference
    engine_index
    engine_contract
)

# ==============================================================================
# 5. FINAL EXECUTABLE
# ==============================================================================
add_executable(strix_engine src/main.cc)
target_link_libraries(strix_engine PRIVATE 
    engine_options 
    engine_common
    engine_memory 
    engine_inference 
    engine_index
    engine_service
    engine_contract
)

# Hardcode ORT library path into binary
set_target_properties(strix_engine PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN"
)

# ==============================================================================
# 6. UNIT TESTING
# ==============================================================================
find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

function(add_unit_test target_name module_name)
    cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
    add_executable(${target_name} ${ARG_SOURCES})
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/${module_name}
    )
    target_link_libraries(${target_name} PRIVATE
        ${ARG_LIBS}
        GTest::gtest_main
        GTest::gtest
    )
    gtest_discover_tests(${target_name})
endfunction()

# AVX2 dot product
add_unit_test(avx2_kernel_test index
    SOURCES src/index/tests/avx2_kernel_test.cc
    LIBS engine_common engine_index
)

# Inference model
add_unit_test(inference_test inference
    SOURCES src/inference/tests/inference_test.cc
    LIBS engine_common engine_inference
)

# Control block ops
add_unit_test(control_block_test memory
    SOURCES src/memory/tests/control_block_test.cc
    LIBS engine_memory
)

# Read/Write payload
add_unit_test(memory_arena_test memory
    SOURCES src/memory/tests/memory_arena_test.cc
    LIBS engine_common engine_memory
)

# Free-list
add_unit_test(free_list_test index
    SOURCES src/index/tests/free_list_test.cc
    LIBS engine_common engine_index
)

# Level-0 node buffer
add_unit_test(level0_ring_test index
    SOURCES src/index/tests/level0_ring_test.cc
    LIBS engine_common engine_index
)

# ==============================================================================
# 7. MICRO-BENCHMARK
# ==============================================================================
find_package(benchmark CONFIG REQUIRED)

function(add_micro_bench target_name module_name)
    cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
    add_executable(${target_name} ${ARG_SOURCES})
    target_include_directories(${target_name} PRIVATE
        ${ENGINE_SOURCE_ROOT}/${module_name}
    )
    target_link_libraries(${target_name} PRIVATE
        ${ARG_LIBS}
        benchmark::benchmark
    )
endfunction()

# AVX2 dot product
add_micro_bench(avx2_kernel_bench index
    SOURCES src/index/bench/avx2_kernel_bench.cc
    LIBS engine_options engine_common engine_index
)

# Inference model: Zero-shot
add_micro_bench(inference_bench inference
    SOURCES src/inference/bench/inference_bench.cc
    LIBS engine_options engine_common engine_inference
)

# Inference model: Custom ops
add_micro_bench(custom_ops_bench inference
    SOURCES src/inference/bench/custom_ops_bench.cc
    LIBS engine_options engine_common engine_inference
)

# Read/Write payload
add_micro_bench(memory_arena_bench memory
    SOURCES src/memory/bench/memory_arena_bench.cc
    LIBS engine_options engine_common engine_memory
)

# Free-list
add_micro_bench(free_list_bench index
    SOURCES src/index/bench/free_list_bench.cc
    LIBS engine_options engine_common engine_index
)

# Level-0 node buffer
add_micro_bench(level0_ring_bench index
    SOURCES src/index/bench/level0_ring_bench.cc
    LIBS engine_options engine_common engine_index
)

# Vector search
add_micro_bench(vector_search_bench index
    SOURCES src/index/bench/vector_search_bench.cc
    LIBS engine_options engine_common engine_memory engine_index
)

# ==============================================================================
# 8. INTEGRATION TESTING
# ==============================================================================
function(add_integration_test target_name)
    cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
    add_executable(${target_name} ${ARG_SOURCES})
    target_link_libraries(${target_name} PRIVATE
        ${ARG_LIBS}
        GTest::gtest_main
        GTest::gtest
    )
    gtest_discover_tests(${target_name})
endfunction()

# ==============================================================================
# 9. MACRO-BENCHMARK
# ==============================================================================
function(add_macro_bench target_name)
    cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
    add_executable(${target_name} ${ARG_SOURCES})
    target_link_libraries(${target_name} PRIVATE
        ${ARG_LIBS}
        benchmark::benchmark
    )
endfunction()
