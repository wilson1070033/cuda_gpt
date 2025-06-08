#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include <vector>

class Tensor {
public:
    float* data;
    std::vector<int> shape;
    int size;
    bool on_device;
    
    Tensor(const std::vector<int>& shape, bool on_device = true);
    ~Tensor();
    
    void to_device();
    void to_host();
    void zero();
    void random_fill(float mean = 0.0f, float std = 1.0f);
    
    static void matmul(const Tensor& a, const Tensor& b, Tensor& c, cublasHandle_t handle);
    static void add(const Tensor& a, const Tensor& b, Tensor& c);
    static void scale(const Tensor& a, float scalar, Tensor& c);
    
    Tensor clone() const;
    void copy_from(const Tensor& other);
    
    float* get_data() const { return data; }
    int get_size() const { return size; }
    std::vector<int> get_shape() const { return shape; }
    
private:
    void allocate_memory();
    void deallocate_memory();
};