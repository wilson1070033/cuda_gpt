#include "transformer.h"
#include <cmath>
#include <algorithm>
#include <random>

__global__ void layer_norm_kernel(const float* input, float* output, const float* gamma, const float* beta,
                                  int batch_size, int seq_len, int d_model, float eps) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    int offset = (batch_idx * seq_len + seq_idx) * d_model;
    const float* x = input + offset;
    float* y = output + offset;
    
    __shared__ float mean, var;
    __shared__ float sum, sum_sq;
    
    // Compute mean
    if (tid == 0) {
        sum = 0.0f;
        for (int i = 0; i < d_model; i++) {
            sum += x[i];
        }
        mean = sum / d_model;
    }
    __syncthreads();
    
    // Compute variance
    if (tid == 0) {
        sum_sq = 0.0f;
        for (int i = 0; i < d_model; i++) {
            float diff = x[i] - mean;
            sum_sq += diff * diff;
        }
        var = sum_sq / d_model;
    }
    __syncthreads();
    
    // Apply layer normalization
    float std_dev = sqrtf(var + eps);
    for (int i = tid; i < d_model; i += blockDim.x) {
        y[i] = gamma[i] * (x[i] - mean) / std_dev + beta[i];
    }
}

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

__global__ void add_positional_encoding_kernel(float* embeddings, const float* pos_encoding,
                                               int batch_size, int seq_len, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * seq_len * d_model;
    
    if (idx < total_size) {
        int pos_idx = (idx / d_model) % seq_len;
        int dim_idx = idx % d_model;
        int pos_encoding_idx = pos_idx * d_model + dim_idx;
        
        embeddings[idx] += pos_encoding[pos_encoding_idx];
    }
}

LayerNorm::LayerNorm(int d_model, float eps) 
    : gamma({d_model}), beta({d_model}), eps(eps) {
    gamma.random_fill(1.0f, 0.02f);  // Initialize gamma around 1
    beta.zero();  // Initialize beta to 0
}

void LayerNorm::forward(const Tensor& input, Tensor& output) {
    int batch_size = input.shape[0];
    int seq_len = input.shape[1];
    int d_model = input.shape[2];
    
    dim3 grid(batch_size, seq_len);
    dim3 block(32);
    
    layer_norm_kernel<<<grid, block>>>(input.data, output.data, gamma.data, beta.data,
                                       batch_size, seq_len, d_model, eps);
    cudaDeviceSynchronize();
}

std::vector<Tensor*> LayerNorm::get_parameters() {
    return {&gamma, &beta};
}

FeedForward::FeedForward(const ModelConfig& config, cublasHandle_t handle)
    : config(config), cublas_handle(handle),
      W1({config.d_model, config.d_ff}),
      W2({config.d_ff, config.d_model}),
      bias1({config.d_ff}),
      bias2({config.d_model}) {
    
    float scale1 = sqrtf(2.0f / config.d_model);
    float scale2 = sqrtf(2.0f / config.d_ff);
    
    W1.random_fill(0.0f, scale1);
    W2.random_fill(0.0f, scale2);
    bias1.zero();
    bias2.zero();
}

void FeedForward::forward(const Tensor& input, Tensor& output) {
    int batch_size = input.shape[0];
    int seq_len = input.shape[1];
    int d_model = input.shape[2];
    
    Tensor hidden({batch_size, seq_len, config.d_ff});
    
    // First linear layer: input -> hidden
    for (int b = 0; b < batch_size; b++) {
        Tensor input_batch({seq_len, d_model});
        Tensor hidden_batch({seq_len, config.d_ff});
        
        input_batch.data = const_cast<float*>(input.data) + b * seq_len * d_model;
        hidden_batch.data = hidden.data + b * seq_len * config.d_ff;
        
        Tensor::matmul(input_batch, W1, hidden_batch, cublas_handle);
        Tensor::add(hidden_batch, bias1, hidden_batch);
    }
    
    // Apply GELU activation
    int block_size = 256;
    int grid_size = (hidden.size + block_size - 1) / block_size;
    gelu_kernel<<<grid_size, block_size>>>(hidden.data, hidden.data, hidden.size);
    cudaDeviceSynchronize();
    
    // Second linear layer: hidden -> output
    for (int b = 0; b < batch_size; b++) {
        Tensor hidden_batch({seq_len, config.d_ff});
        Tensor output_batch({seq_len, d_model});
        
        hidden_batch.data = hidden.data + b * seq_len * config.d_ff;
        output_batch.data = output.data + b * seq_len * d_model;
        
        Tensor::matmul(hidden_batch, W2, output_batch, cublas_handle);
        Tensor::add(output_batch, bias2, output_batch);
    }
}

std::vector<Tensor*> FeedForward::get_parameters() {
    return {&W1, &W2, &bias1, &bias2};
}

TransformerBlock::TransformerBlock(const ModelConfig& config, cublasHandle_t handle)
    : config(config) {
    
    AttentionConfig attn_config;
    attn_config.d_model = config.d_model;
    attn_config.n_heads = config.n_heads;
    attn_config.d_k = config.d_model / config.n_heads;
    attn_config.max_seq_len = config.max_seq_len;
    attn_config.dropout = config.dropout;
    
    attention = std::make_unique<MultiHeadAttention>(attn_config, handle);
    feed_forward = std::make_unique<FeedForward>(config, handle);
    norm1 = std::make_unique<LayerNorm>(config.d_model, config.layer_norm_eps);
    norm2 = std::make_unique<LayerNorm>(config.d_model, config.layer_norm_eps);
}

void TransformerBlock::forward(const Tensor& input, Tensor& output, const Tensor* mask) {
    int batch_size = input.shape[0];
    int seq_len = input.shape[1];
    int d_model = input.shape[2];
    
    // Self-attention with residual connection
    Tensor norm1_output({batch_size, seq_len, d_model});
    Tensor attn_output({batch_size, seq_len, d_model});
    Tensor residual1({batch_size, seq_len, d_model});
    
    // Layer norm before attention
    norm1->forward(input, norm1_output);
    
    // Self-attention
    attention->forward(norm1_output, attn_output, mask);
    
    // Residual connection
    Tensor::add(input, attn_output, residual1);
    
    // Feed-forward with residual connection
    Tensor norm2_output({batch_size, seq_len, d_model});
    Tensor ff_output({batch_size, seq_len, d_model});
    
    // Layer norm before feed-forward
    norm2->forward(residual1, norm2_output);
    
    // Feed-forward
    feed_forward->forward(norm2_output, ff_output);
    
    // Final residual connection
    Tensor::add(residual1, ff_output, output);
}

std::vector<Tensor*> TransformerBlock::get_parameters() {
    std::vector<Tensor*> params;
    
    auto attn_params = attention->get_parameters();
    auto ff_params = feed_forward->get_parameters();
    auto norm1_params = norm1->get_parameters();
    auto norm2_params = norm2->get_parameters();
    
    params.insert(params.end(), attn_params.begin(), attn_params.end());
    params.insert(params.end(), ff_params.begin(), ff_params.end());
    params.insert(params.end(), norm1_params.begin(), norm1_params.end());
    params.insert(params.end(), norm2_params.begin(), norm2_params.end());
    
    return params;
}

GPTModel::GPTModel(const ModelConfig& config)
    : config(config),
      token_embedding({config.vocab_size, config.d_model}),
      position_embedding({config.max_seq_len, config.d_model}),
      lm_head({config.d_model, config.vocab_size}) {
    
    // Initialize cuBLAS
    cublasCreate(&cublas_handle);
    
    // Create transformer layers
    for (int i = 0; i < config.n_layers; i++) {
        layers.push_back(std::make_unique<TransformerBlock>(config, cublas_handle));
    }
    
    // Final layer norm
    final_norm = std::make_unique<LayerNorm>(config.d_model, config.layer_norm_eps);
    
    initialize_weights();
}

GPTModel::~GPTModel() {
    cublasDestroy(cublas_handle);
}

void GPTModel::initialize_weights() {
    float scale = sqrtf(2.0f / config.d_model);
    
    token_embedding.random_fill(0.0f, scale);
    position_embedding.random_fill(0.0f, scale);
    lm_head.random_fill(0.0f, scale);
}

void GPTModel::forward(const Tensor& input_ids, Tensor& logits, const Tensor* mask) {
    int batch_size = input_ids.shape[0];
    int seq_len = input_ids.shape[1];
    
    // Get token embeddings and add positional embeddings
    Tensor embeddings({batch_size, seq_len, config.d_model});
    
    // Simple embedding lookup - create random embeddings for now
    embeddings.random_fill(0.0f, 0.1f);
    
    // Add positional encodings
    int block_size = 256;
    int grid_size = (embeddings.size + block_size - 1) / block_size;
    add_positional_encoding_kernel<<<grid_size, block_size>>>(
        embeddings.data, position_embedding.data, batch_size, seq_len, config.d_model);
    cudaDeviceSynchronize();
    
    // Pass through transformer layers
    Tensor layer_input = embeddings.clone();
    Tensor layer_output({batch_size, seq_len, config.d_model});
    
    for (int i = 0; i < config.n_layers; i++) {
        layers[i]->forward(layer_input, layer_output, mask);
        layer_input.copy_from(layer_output);
    }
    
    // Final layer normalization
    Tensor norm_output({batch_size, seq_len, config.d_model});
    final_norm->forward(layer_input, norm_output);
    
    // Language modeling head
    for (int b = 0; b < batch_size; b++) {
        Tensor norm_batch({seq_len, config.d_model});
        Tensor logits_batch({seq_len, config.vocab_size});
        
        norm_batch.data = norm_output.data + b * seq_len * config.d_model;
        logits_batch.data = logits.data + b * seq_len * config.vocab_size;
        
        Tensor::matmul(norm_batch, lm_head, logits_batch, cublas_handle);
    }
}

std::vector<Tensor*> GPTModel::get_all_parameters() {
    std::vector<Tensor*> all_params;
    
    all_params.push_back(&token_embedding);
    all_params.push_back(&position_embedding);
    all_params.push_back(&lm_head);
    
    for (auto& layer : layers) {
        auto layer_params = layer->get_parameters();
        all_params.insert(all_params.end(), layer_params.begin(), layer_params.end());
    }
    
    auto final_norm_params = final_norm->get_parameters();
    all_params.insert(all_params.end(), final_norm_params.begin(), final_norm_params.end());
    
    return all_params;
}

int GPTModel::sample_token(const Tensor& logits, float temperature) {
    // Simple top-k sampling implementation
    std::vector<float> probs(config.vocab_size);
    
    // Copy logits to host
    Tensor host_logits = logits.clone();
    host_logits.to_host();
    
    // Apply temperature and softmax
    float max_logit = *std::max_element(host_logits.data, host_logits.data + config.vocab_size);
    float sum_exp = 0.0f;
    
    for (int i = 0; i < config.vocab_size; i++) {
        probs[i] = expf((host_logits.data[i] - max_logit) / temperature);
        sum_exp += probs[i];
    }
    
    for (int i = 0; i < config.vocab_size; i++) {
        probs[i] /= sum_exp;
    }
    
    // Sample from the distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    
    return dist(gen);
}