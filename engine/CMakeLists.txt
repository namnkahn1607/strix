cmake_minimum_required(VERSION 3.28)
project(strix_engine VERSION 1.0)

enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(absl CONFIG REQUIRED)
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
# The library is compiled DYNAMICALLY using clang++-18.
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
    -include "${CMAKE_CURRENT_SOURCE_DIR}/src/misc/lockfree_invariants.h"
)

# ==============================================================================
# 3. AUTO-GENERATE CONTRACTS
# ==============================================================================
set(PROTO_DIR "${STRIX_REPO_ROOT}/proto/v1")
set(PROTO_FILE "${PROTO_DIR}/cache.proto")
set(PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Pre-make directory in case of non-existing
file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

# Get vendored protoc and grpc_cpp_plugin from vcpkg
set(PROTOC_EXEC "${PROTOC_EXECUTABLE}")
set(GRPC_PLUGIN_EXEC "$<TARGET_FILE:gRPC::grpc_cpp_plugin>")

# Execute command to generate contracts
add_custom_command(
    OUTPUT
        "${PROTO_GEN_DIR}/cache.pb.cc"
        "${PROTO_GEN_DIR}/cache.pb.h"
        "${PROTO_GEN_DIR}/cache.grpc.pb.cc"
        "${PROTO_GEN_DIR}/cache.grpc.pb.h"
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
    COMMENT "Generating C++ gRPC stubs from cache.proto..."
)

# Package generated contracts into a static independent target
add_library(engine_contract STATIC
    "${PROTO_GEN_DIR}/cache.pb.cc"
    "${PROTO_GEN_DIR}/cache.grpc.pb.cc"
)
target_link_libraries(engine_contract PUBLIC gRPC::grpc++ protobuf::libprotobuf)
target_include_directories(engine_contract PUBLIC "${PROTO_GEN_DIR}")

# ==============================================================================
# 4. DATA PLANE MODULES
#   include/<module>/*.h : Public API
#   src/<module>/*.h     : Private API
# ==============================================================================
set(ENGINE_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Module 1: Base
add_library(engine_base INTERFACE)
target_include_directories(engine_base INTERFACE ${PUBLIC_HEADER_ROOT})
target_link_libraries(engine_base INTERFACE engine_options)

# Module 2: Inference
add_library(engine_inference STATIC src/inference/sentence_encoder.cc)
target_include_directories(engine_inference
    PUBLIC ${PUBLIC_HEADER_ROOT}
    PRIVATE ${ENGINE_SOURCE_ROOT}/inference
)
target_link_libraries(engine_inference PRIVATE engine_options engine_base)
target_link_libraries(engine_inference PUBLIC ONNXRuntime)

# Module 3: ANN
add_library(engine_ann INTERFACE)
target_include_directories(engine_ann INTERFACE ${PUBLIC_HEADER_ROOT})
target_link_libraries(engine_ann INTERFACE engine_options engine_inference)

# Module 4: Worker
add_library(engine_worker STATIC src/worker/identity.cc)
target_include_directories(engine_worker
    PUBLIC ${PUBLIC_HEADER_ROOT}
    PRIVATE ${ENGINE_SOURCE_ROOT}/worker
)
target_link_libraries(engine_worker PRIVATE
    absl::log
    engine_options
    engine_base
)

# Module 5: Memory
add_library(engine_memory STATIC src/memory/arena.cc)
target_include_directories(engine_memory
    PUBLIC ${PUBLIC_HEADER_ROOT}
    PRIVATE ${ENGINE_SOURCE_ROOT}/memory
)
target_link_libraries(engine_memory PRIVATE
    absl::log
    engine_options
    engine_base
    engine_worker
)

# Module 6: Collection
add_library(engine_collection STATIC
    src/collection/collection.cc
    src/collection/internal/node_buf.cc
    src/collection/internal/ivf_table.cc
    src/collection/internal/recalibration.cc
)
target_include_directories(engine_collection
    PUBLIC ${PUBLIC_HEADER_ROOT}
    PRIVATE ${ENGINE_SOURCE_ROOT}/collection
)
target_link_libraries(engine_collection PRIVATE
    absl::log
    engine_options
    engine_base
    engine_memory
)

# Module 7: Network
add_library(engine_service STATIC src/rpc/cache_service.cc)
target_include_directories(engine_service
    PUBLIC ${PUBLIC_HEADER_ROOT}
    PRIVATE ${ENGINE_SOURCE_ROOT}/rpc
)
target_link_libraries(engine_service PRIVATE
    engine_options
    engine_base
    engine_inference
    engine_memory
    engine_collection
    engine_contract
)

# ==============================================================================
# 5. FINAL EXECUTABLE
# ==============================================================================
add_executable(strix_engine src/main.cc)
target_link_libraries(strix_engine PRIVATE
    engine_options
    engine_base
    engine_inference
    engine_ann
    engine_worker
    engine_memory
    engine_collection
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

# Dot product
add_unit_test(dot_product_test ann
    SOURCES src/ann/dot_product_test.cc
    LIBS engine_base engine_ann
)

# Memory arena byte fetch/commit
add_unit_test(arena_test memory
    SOURCES src/memory/arena_test.cc
    LIBS engine_base engine_memory
)

# Control block
add_unit_test(control_block_test memory
    SOURCES src/memory/control_block_test.cc
    LIBS engine_base engine_memory
)

# Inference model
add_unit_test(inference_test inference
    SOURCES src/inference/inference_test.cc
    LIBS engine_base engine_inference
)

# L0-tier node buffer
add_unit_test(node_buf_test collection
    SOURCES src/collection/node_buf_test.cc
    LIBS engine_base engine_collection
)

# Treiber stack
add_unit_test(tagged_treiber_test base
    SOURCES src/base/tagged_treiber_test.cc
    LIBS engine_base
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

# Dot product
add_micro_bench(dot_product_bench ann
    SOURCES src/ann/dot_product_bench.cc
    LIBS engine_options engine_base engine_ann
)

# Inference model: Custom ops
add_micro_bench(inference_bench inference
    SOURCES src/inference/inference_bench.cc
    LIBS absl::log engine_options engine_base engine_inference
)

# L0-tier node buffer
add_micro_bench(node_buf_bench collection
    SOURCES src/collection/node_buf_bench.cc
    LIBS engine_options engine_base engine_collection
)

# Tagged treiber stack
add_micro_bench(tagged_treiber_bench base
    SOURCES src/base/tagged_treiber_bench.cc
    LIBS engine_base
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
