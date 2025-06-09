#pragma once
#include "tensor.h"
#include <cublas_v2.h>

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

struct AttentionConfig {
    int d_model = 512;     // Model dimension
    int n_heads = 8;       // Number of attention heads
    int d_k = 64;          // Key/Query dimension (d_model / n_heads)
    int max_seq_len = 1024; // Maximum sequence length
    float dropout = 0.1f;
};

class MultiHeadAttention {
private:
    AttentionConfig config;
    Tensor W_q, W_k, W_v, W_o;  // Weight matrices
    Tensor bias_q, bias_k, bias_v, bias_o;  // Bias vectors
    cublasHandle_t cublas_handle;
    
#ifdef USE_CUDNN
    cudnnHandle_t cudnn_handle;
    cudnnTensorDescriptor_t scores_desc;
#endif
    
public:
    MultiHeadAttention(const AttentionConfig& config, cublasHandle_t handle);
    ~MultiHeadAttention();
    
    void forward(const Tensor& input, Tensor& output, const Tensor* mask = nullptr);
    void backward(const Tensor& grad_output, const Tensor& input, 
                  Tensor& grad_input, bool compute_input_grad = true);
    
    void initialize_weights();
    std::vector<Tensor*> get_parameters();
    
private:
    void scaled_dot_product_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                                      Tensor& output, const Tensor* mask = nullptr);
                                      
#ifdef USE_CUDNN
    void forward_cudnn(const Tensor& input, Tensor& output, const Tensor* mask = nullptr);
    void scaled_dot_product_attention_cudnn(const Tensor& Q, const Tensor& K, const Tensor& V,
                                           Tensor& output, const Tensor* mask = nullptr);
#endif
};