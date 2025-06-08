#include "tensor.h"
#include <iostream>
#include <random>
#include <cstring>
#include <curand_kernel.h>

__global__ void add_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void scale_kernel(const float* a, float scalar, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * scalar;
    }
}

__global__ void zero_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 0.0f;
    }
}

__global__ void setup_curand(curandState *state, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

__global__ void random_fill_kernel(float* data, curandState* state, int size, float mean, float std) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = mean + std * curand_normal(&state[idx]);
    }
}

Tensor::Tensor(const std::vector<int>& shape, bool on_device) 
    : shape(shape), on_device(on_device) {
    size = 1;
    for (int dim : shape) {
        size *= dim;
    }
    allocate_memory();
}

Tensor::~Tensor() {
    deallocate_memory();
}

void Tensor::allocate_memory() {
    if (on_device) {
        cudaMalloc(&data, size * sizeof(float));
    } else {
        data = new float[size];
    }
}

void Tensor::deallocate_memory() {
    if (data) {
        if (on_device) {
            cudaFree(data);
        } else {
            delete[] data;
        }
        data = nullptr;
    }
}

void Tensor::to_device() {
    if (!on_device) {
        float* device_data;
        cudaMalloc(&device_data, size * sizeof(float));
        cudaMemcpy(device_data, data, size * sizeof(float), cudaMemcpyHostToDevice);
        delete[] data;
        data = device_data;
        on_device = true;
    }
}

void Tensor::to_host() {
    if (on_device) {
        float* host_data = new float[size];
        cudaMemcpy(host_data, data, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(data);
        data = host_data;
        on_device = false;
    }
}

void Tensor::zero() {
    if (on_device) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        zero_kernel<<<grid_size, block_size>>>(data, size);
        cudaDeviceSynchronize();
    } else {
        std::memset(data, 0, size * sizeof(float));
    }
}

void Tensor::random_fill(float mean, float std) {
    if (on_device) {
        static curandState* d_state = nullptr;
        static int last_size = 0;
        
        if (d_state == nullptr || size > last_size) {
            if (d_state) cudaFree(d_state);
            cudaMalloc(&d_state, size * sizeof(curandState));
            last_size = size;
            
            int block_size = 256;
            int grid_size = (size + block_size - 1) / block_size;
            setup_curand<<<grid_size, block_size>>>(d_state, static_cast<unsigned long>(time(nullptr)), size);
            cudaDeviceSynchronize();
        }
        
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        random_fill_kernel<<<grid_size, block_size>>>(data, d_state, size, mean, std);
        cudaDeviceSynchronize();
    } else {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(mean, std);
        for (int i = 0; i < size; i++) {
            data[i] = dist(gen);
        }
    }
}

void Tensor::matmul(const Tensor& a, const Tensor& b, Tensor& c, cublasHandle_t handle) {
    if (a.shape.size() != 2 || b.shape.size() != 2 || c.shape.size() != 2) {
        throw std::runtime_error("Matrix multiplication requires 2D tensors");
    }
    
    int m = a.shape[0];
    int k = a.shape[1];
    int n = b.shape[1];
    
    if (b.shape[0] != k || c.shape[0] != m || c.shape[1] != n) {
        throw std::runtime_error("Matrix dimension mismatch");
    }
    
    const float alpha = 1.0f, beta = 0.0f;
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                b.data, n,
                a.data, k,
                &beta,
                c.data, n);
}

void Tensor::add(const Tensor& a, const Tensor& b, Tensor& c) {
    if (a.size != b.size || a.size != c.size) {
        throw std::runtime_error("Tensor size mismatch for addition");
    }
    
    if (a.on_device && b.on_device && c.on_device) {
        int block_size = 256;
        int grid_size = (a.size + block_size - 1) / block_size;
        add_kernel<<<grid_size, block_size>>>(a.data, b.data, c.data, a.size);
        cudaDeviceSynchronize();
    } else {
        for (int i = 0; i < a.size; i++) {
            c.data[i] = a.data[i] + b.data[i];
        }
    }
}

void Tensor::scale(const Tensor& a, float scalar, Tensor& c) {
    if (a.size != c.size) {
        throw std::runtime_error("Tensor size mismatch for scaling");
    }
    
    if (a.on_device && c.on_device) {
        int block_size = 256;
        int grid_size = (a.size + block_size - 1) / block_size;
        scale_kernel<<<grid_size, block_size>>>(a.data, scalar, c.data, a.size);
        cudaDeviceSynchronize();
    } else {
        for (int i = 0; i < a.size; i++) {
            c.data[i] = a.data[i] * scalar;
        }
    }
}

Tensor Tensor::clone() const {
    Tensor cloned(shape, on_device);
    cloned.copy_from(*this);
    return cloned;
}

void Tensor::copy_from(const Tensor& other) {
    if (size != other.size) {
        throw std::runtime_error("Cannot copy tensors of different sizes");
    }
    
    if (on_device && other.on_device) {
        cudaMemcpy(data, other.data, size * sizeof(float), cudaMemcpyDeviceToDevice);
    } else if (!on_device && !other.on_device) {
        std::memcpy(data, other.data, size * sizeof(float));
    } else if (on_device && !other.on_device) {
        cudaMemcpy(data, other.data, size * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(data, other.data, size * sizeof(float), cudaMemcpyDeviceToHost);
    }
}