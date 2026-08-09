# cmake/vcpkg-triplets/clang-18-toolchain.cmake

set(CMAKE_C_COMPILER "/usr/bin/clang-18")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-18")
set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld-18")
