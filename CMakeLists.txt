cmake_minimum_required(VERSION 3.18)
project(cuda_gpt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

enable_language(CUDA)

include_directories(include)

set(SOURCES
    main.cu
    src/tensor.cu
    src/attention.cu
    src/transformer.cu
    src/training.cu
)

set(SAFE_SOURCES
    safe_main.cu
    src/tensor.cu
    src/attention.cu
    src/transformer.cu
    src/training.cu
)

set(CHAT_SOURCES
    chat.cu
    src/tensor.cu
    src/attention.cu
    src/transformer.cu
    src/training.cu
)

add_executable(cuda_gpt ${SOURCES})
add_executable(cuda_gpt_safe ${SAFE_SOURCES})
add_executable(cuda_chat ${CHAT_SOURCES})

set_target_properties(cuda_gpt PROPERTIES
    CUDA_ARCHITECTURES "86"
)

set_target_properties(cuda_gpt_safe PROPERTIES
    CUDA_ARCHITECTURES "86"
)

set_target_properties(cuda_chat PROPERTIES
    CUDA_ARCHITECTURES "86"
)

# Suppress unused variable warnings for CUDA
target_compile_options(cuda_gpt PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=550>
)

target_compile_options(cuda_gpt_safe PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=550>
)

target_compile_options(cuda_chat PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=550>
)

# Find cuDNN
find_library(CUDNN_LIBRARY cudnn 
    HINTS 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib64 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib
    /usr/local/cuda/lib64
    /usr/lib/x86_64-linux-gnu
)

# Find cuDNN headers
find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    /usr/local/cuda/include
    /usr/include
)

if(CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
    message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
    target_include_directories(cuda_gpt PRIVATE ${CUDNN_INCLUDE_DIR})
    target_include_directories(cuda_chat PRIVATE ${CUDNN_INCLUDE_DIR})
    target_link_libraries(cuda_gpt cublas curand ${CUDNN_LIBRARY})
    target_link_libraries(cuda_gpt_safe cublas curand ${CUDNN_LIBRARY})
    target_link_libraries(cuda_chat cublas curand ${CUDNN_LIBRARY})
    target_compile_definitions(cuda_gpt PRIVATE USE_CUDNN)
    target_compile_definitions(cuda_gpt_safe PRIVATE USE_CUDNN)
    target_compile_definitions(cuda_chat PRIVATE USE_CUDNN)
else()
    message(WARNING "cuDNN not found, falling back to standard CUDA operations")
    target_link_libraries(cuda_gpt cublas curand)
    target_link_libraries(cuda_gpt_safe cublas curand)
    target_link_libraries(cuda_chat cublas curand)
endif()