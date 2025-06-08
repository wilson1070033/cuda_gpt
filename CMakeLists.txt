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

set(CHAT_SOURCES
    chat.cu
    src/tensor.cu
    src/attention.cu
    src/transformer.cu
    src/training.cu
)

add_executable(cuda_gpt ${SOURCES})
add_executable(cuda_chat ${CHAT_SOURCES})

set_target_properties(cuda_gpt PROPERTIES
    CUDA_ARCHITECTURES "86"
)

set_target_properties(cuda_chat PROPERTIES
    CUDA_ARCHITECTURES "86"
)

# Suppress unused variable warnings for CUDA
target_compile_options(cuda_gpt PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=550>
)

target_compile_options(cuda_chat PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=550>
)

target_link_libraries(cuda_gpt cublas curand)
target_link_libraries(cuda_chat cublas curand)