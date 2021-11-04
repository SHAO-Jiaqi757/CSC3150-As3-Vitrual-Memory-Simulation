cmake_minimum_required(VERSION 3.19)
project(3150A3 CUDA)
set(CMAKE_CUDA_STANDARD 11)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "-G -g")
endif()

# add_library(cuda STATIC 
#     kernel.cu
#     user_program.cu
#     virtual_memory.cu
#     virtual_memory.h
#     main.cu
# )

add_executable(cuda main.cu user_program.cu virtual_memory.cu virtual_memory.h)

set_target_properties(
        cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)