cmake_minimum_required(VERSION 3.13.4)
project(CudaLaunchParser)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force CMake to find MLIR in the LLVM 22 directory
set(MLIR_DIR "/usr/lib/llvm-22/lib/cmake/mlir")

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_executable(cuda-launch-parser main.cpp)

# Link Clang and the all-in-one MLIR library
target_link_libraries(cuda-launch-parser PRIVATE
  clangTooling
  clangFrontend
  clangAST
  clangBasic
  clangDriver
  clangSerialization
  clangParse
  clangSema
  clangAnalysis
  clangEdit
  clangLex
  LLVM
  MLIR
)