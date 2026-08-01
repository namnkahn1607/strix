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
# 0. SHARED PUBLIC HEADER ROOT
# Single root for all module's public API: include/<module>/*.h
# ==============================================================================
set(PUBLIC_HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include")

# ==============================================================================
# 1. ORT LIBRARY PATH
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
# 2. GLOBAL COMPILER OPTIONS
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
# 5. TOTAL EXECUTABLE
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
# 6. TESTING EXECUTABLE(S)
# ==============================================================================
find_package(GTest CONFIG REQUIRED)

# Test AVX2 Dot Product
add_executable(avx2_kernel_test src/index/tests/avx2_kernel_test.cc)
target_link_libraries(avx2_kernel_test PRIVATE
    engine_common    
    engine_index
    GTest::gtest_main
    GTest::gtest
)

# Test Inference model all-MiniLM-L6-v2
add_executable(inference_test src/inference/tests/inference_test.cc)
target_link_libraries(inference_test PRIVATE
    engine_common
    engine_inference
    GTest::gtest_main
    GTest::gtest
)

# Test Control Block operations
add_executable(control_block_test src/memory/tests/control_block_test.cc)
target_link_libraries(control_block_test PRIVATE
    engine_memory
    GTest::gtest_main
    GTest::gtest
)

# Test Payload Read/Write of Memory Arena
add_executable(memory_arena_test src/memory/tests/memory_arena_test.cc)
target_link_libraries(memory_arena_test PRIVATE
    engine_common
    engine_memory
    GTest::gtest_main
    GTest::gtest
)

# Test Free List
add_executable(free_list_test src/index/tests/free_list_test.cc)
target_link_libraries(free_list_test PRIVATE
    engine_common
    engine_index
    GTest::gtest_main
    GTest::gtest
)

# Test MPSC L0 ring buffer
add_executable(level0_ring_test src/index/tests/level0_ring_test.cc)
target_link_libraries(level0_ring_test PRIVATE
    engine_common
    engine_index
    GTest::gtest_main
    GTest::gtest
)

include(GoogleTest)
gtest_discover_tests(avx2_kernel_test)
gtest_discover_tests(inference_test)
gtest_discover_tests(control_block_test)
gtest_discover_tests(memory_arena_test)
gtest_discover_tests(free_list_test)
gtest_discover_tests(level0_ring_test)

# ==============================================================================
# 7. BENCHMARK EXECUTABLE(S)
# ==============================================================================
find_package(benchmark CONFIG REQUIRED)

# Benchmark AVX2 Dot Product
add_executable(avx2_kernel_bench src/index/bench/avx2_kernel_bench.cc)
target_link_libraries(avx2_kernel_bench PRIVATE
    engine_options
    engine_common
    engine_index
    benchmark::benchmark
)

# Benchmark Inference model (Zero-shot)
add_executable(inference_bench src/inference/bench/inference_bench.cc)
target_link_libraries(inference_bench PRIVATE
    engine_options
    engine_common
    engine_inference
    benchmark::benchmark
)

# Benchmark Inference model (Custom Ops)
add_executable(custom_ops_bench src/inference/bench/custom_ops_bench.cc)
target_link_libraries(custom_ops_bench PRIVATE
    engine_options
    engine_common
    engine_inference
    benchmark::benchmark
)

# Benchmark Payload Read/Write of Memory Arena
add_executable(memory_arena_bench src/memory/bench/memory_arena_bench.cc)
target_link_libraries(memory_arena_bench PRIVATE
    engine_options
    engine_common
    engine_memory
    benchmark::benchmark
)

# Benchmark Free List operations
add_executable(free_list_bench src/index/bench/free_list_bench.cc)
target_link_libraries(free_list_bench PRIVATE
    engine_options
    engine_common
    engine_index
    benchmark::benchmark    
)

# Benchmark MPSC L0 ring buffer
add_executable(level0_ring_bench src/index/bench/level0_ring_bench.cc)
target_link_libraries(level0_ring_bench PRIVATE
    engine_options
    engine_common
    engine_index
    benchmark::benchmark
)

# Benchmark Vector searching subroutines
add_executable(vector_search_bench src/index/bench/vector_search_bench.cc)
target_link_libraries(vector_search_bench PRIVATE
    engine_options
    engine_common
    engine_memory
    engine_index
    benchmark::benchmark
)
