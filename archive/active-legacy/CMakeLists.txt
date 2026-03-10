cmake_minimum_required (VERSION 3.20)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES 86)

project (ece408project LANGUAGES CUDA CXX)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")

aux_source_directory(. DIR_SRCS)

add_subdirectory(src)

add_library(ece408net ece408net.cc ece408net.h)

add_executable(final final.cc "${PROJECT_SOURCE_DIR}/src/layer/custom/m3-competition.cu")
target_link_libraries(final ece408net MiniDNNLib cublas)
