#pragma once
#include "attention.h"
#include "tensor.h"
#include <vector>
#include <memory>

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

struct ModelConfig {
    int vocab_size = 32000;
    int d_model = 512;
    int n_heads = 8;
    int n_layers = 6;
    int d_ff = 2048;        // Feed-forward hidden dimension
    int max_seq_len = 1024;
    float dropout = 0.1f;
    float layer_norm_eps = 1e-5f;
};

class LayerNorm {
private:
    Tensor gamma, beta;
    float eps;
    
#ifdef USE_CUDNN
    cudnnHandle_t cudnn_handle;
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnActivationDescriptor_t activation_desc;
#endif
    
public:
    LayerNorm(int d_model, float eps = 1e-5f);
    ~LayerNorm();
    void forward(const Tensor& input, Tensor& output);
    void backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input);
    std::vector<Tensor*> get_parameters();
    
#ifdef USE_CUDNN
    void forward_cudnn(const Tensor& input, Tensor& output);
#endif
};

class FeedForward {
private:
    Tensor W1, W2, bias1, bias2;
    ModelConfig config;
    cublasHandle_t cublas_handle;
    
#ifdef USE_CUDNN
    cudnnHandle_t cudnn_handle;
    cudnnTensorDescriptor_t input_desc, hidden_desc, output_desc;
    cudnnActivationDescriptor_t gelu_desc;
#endif
    
public:
    FeedForward(const ModelConfig& config, cublasHandle_t handle);
    ~FeedForward();
    void forward(const Tensor& input, Tensor& output);
    void forward_cudnn(const Tensor& input, Tensor& output);
    void backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input);
    std::vector<Tensor*> get_parameters();
};

class TransformerBlock {
private:
    std::unique_ptr<MultiHeadAttention> attention;
    std::unique_ptr<FeedForward> feed_forward;
    std::unique_ptr<LayerNorm> norm1, norm2;
    ModelConfig config;
    
public:
    TransformerBlock(const ModelConfig& config, cublasHandle_t handle);
    void forward(const Tensor& input, Tensor& output, const Tensor* mask = nullptr);
    void backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input);
    std::vector<Tensor*> get_parameters();
};

class GPTModel {
private:
    ModelConfig config;
    Tensor token_embedding, position_embedding;
    std::vector<std::unique_ptr<TransformerBlock>> layers;
    std::unique_ptr<LayerNorm> final_norm;
    Tensor lm_head;  // Language modeling head
    cublasHandle_t cublas_handle;
    
public:
    GPTModel(const ModelConfig& config);
    ~GPTModel();
    
    void forward(const Tensor& input_ids, Tensor& logits, const Tensor* mask = nullptr);
    void backward(const Tensor& grad_logits, const Tensor& input_ids, 
                  std::vector<Tensor>& grad_params);
    
    void initialize_weights();
    std::vector<Tensor*> get_all_parameters();
    
    // Inference methods
    std::vector<int> generate(const std::vector<int>& prompt, int max_new_tokens = 100);
    int sample_token(const Tensor& logits, float temperature = 1.0f);
};