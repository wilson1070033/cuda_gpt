#include "attention.h"
#include <cmath>
#include <iostream>

__global__ void softmax_kernel(float* input, int batch_size, int seq_len, int d_k) {
    int batch_idx = blockIdx.y;
    int head_idx = blockIdx.z;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || row >= seq_len) return;
    
    // Calculate offset for this specific attention head and batch
    int offset = (batch_idx * gridDim.z + head_idx) * seq_len * seq_len + row * seq_len;
    float* row_data = input + offset;
    
    // Find maximum for numerical stability
    __shared__ float max_val;
    __shared__ float sum_exp;
    
    if (tid == 0) {
        max_val = row_data[0];
        for (int i = 1; i < seq_len; i++) {
            max_val = fmaxf(max_val, row_data[i]);
        }
    }
    __syncthreads();
    
    // Compute sum of exponentials
    if (tid == 0) {
        sum_exp = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            sum_exp += expf(row_data[i] - max_val);
        }
    }
    __syncthreads();
    
    // Apply softmax
    for (int i = tid; i < seq_len; i += blockDim.x) {
        row_data[i] = expf(row_data[i] - max_val) / sum_exp;
    }
}

__global__ void apply_mask_kernel(float* attention_scores, const float* mask, 
                                  int batch_size, int n_heads, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * n_heads * seq_len * seq_len;
    
    if (idx >= total_size) return;
    
    int batch_idx = idx / (n_heads * seq_len * seq_len);
    int remaining = idx % (n_heads * seq_len * seq_len);
    int head_idx = remaining / (seq_len * seq_len);
    remaining = remaining % (seq_len * seq_len);
    (void)head_idx; // Suppress unused variable warning
    int i = remaining / seq_len;
    int j = remaining % seq_len;
    
    int mask_idx = batch_idx * seq_len * seq_len + i * seq_len + j;
    
    if (mask[mask_idx] == 0.0f) {
        attention_scores[idx] = -1e9f;  // Large negative value for masked positions
    }
}

__global__ void transpose_kernel(const float* input, float* output, 
                                 int batch_size, int seq_len, int n_heads, int d_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * seq_len * n_heads * d_k;
    
    if (idx >= total_size) return;
    
    int batch_idx = idx / (seq_len * n_heads * d_k);
    int remaining = idx % (seq_len * n_heads * d_k);
    int seq_idx = remaining / (n_heads * d_k);
    remaining = remaining % (n_heads * d_k);
    int head_idx = remaining / d_k;
    int d_idx = remaining % d_k;
    
    // From [batch, seq, n_heads, d_k] to [batch, n_heads, seq, d_k]
    int output_idx = batch_idx * n_heads * seq_len * d_k + 
                     head_idx * seq_len * d_k + 
                     seq_idx * d_k + d_idx;
    
    output[output_idx] = input[idx];
}

MultiHeadAttention::MultiHeadAttention(const AttentionConfig& config, cublasHandle_t handle)
    : config(config), cublas_handle(handle),
      W_q({config.d_model, config.d_model}),
      W_k({config.d_model, config.d_model}),
      W_v({config.d_model, config.d_model}),
      W_o({config.d_model, config.d_model}),
      bias_q({config.d_model}),
      bias_k({config.d_model}),
      bias_v({config.d_model}),
      bias_o({config.d_model}) {
    
    initialize_weights();
    
#ifdef USE_CUDNN
    // Initialize cuDNN
    cudnnCreate(&cudnn_handle);
    cudnnCreateTensorDescriptor(&scores_desc);
#endif
}

MultiHeadAttention::~MultiHeadAttention() {
#ifdef USE_CUDNN
    cudnnDestroyTensorDescriptor(scores_desc);
    cudnnDestroy(cudnn_handle);
#endif
}

void MultiHeadAttention::initialize_weights() {
    float scale = sqrtf(2.0f / config.d_model);
    
    W_q.random_fill(0.0f, scale);
    W_k.random_fill(0.0f, scale);
    W_v.random_fill(0.0f, scale);
    W_o.random_fill(0.0f, scale);
    
    bias_q.zero();
    bias_k.zero();
    bias_v.zero();
    bias_o.zero();
}

void MultiHeadAttention::forward(const Tensor& input, Tensor& output, const Tensor* mask) {
#ifdef USE_CUDNN
    forward_cudnn(input, output, mask);
#else
    // Standard CUDA implementation without cuDNN
    int batch_size = input.shape[0];
    int seq_len = input.shape[1];
    int d_model = input.shape[2];
    
    // Create Q, K, V matrices
    Tensor Q({batch_size, seq_len, d_model}), K({batch_size, seq_len, d_model}), V({batch_size, seq_len, d_model});
    
    // Standard matrix multiplication implementation
    Tensor::matmul(input, W_q, Q, cublas_handle);
    Tensor::add(Q, bias_q, Q);
    
    Tensor::matmul(input, W_k, K, cublas_handle);
    Tensor::add(K, bias_k, K);
    
    Tensor::matmul(input, W_v, V, cublas_handle);
    Tensor::add(V, bias_v, V);
    
    // Apply attention
    scaled_dot_product_attention(Q, K, V, output, mask);
#endif
}

#ifdef USE_CUDNN
void MultiHeadAttention::forward_cudnn(const Tensor& input, Tensor& output, const Tensor* mask) {
    int batch_size = input.shape[0];
    int seq_len = input.shape[1];
    int d_model = input.shape[2];
    
    // Create Q, K, V matrices
    Tensor Q({batch_size, seq_len, d_model}), K({batch_size, seq_len, d_model}), V({batch_size, seq_len, d_model});
    
    // Compute Q = input * W_q + bias_q
    // For simplicity, we'll do this for each batch item separately
    for (int b = 0; b < batch_size; b++) {
        // Create views into the tensors for this batch
        Tensor input_batch({seq_len, d_model});
        Tensor Q_batch({seq_len, d_model});
        Tensor K_batch({seq_len, d_model});
        Tensor V_batch({seq_len, d_model});
        
        // Copy batch data (simplified - in practice you'd use tensor slicing)
        cudaMemcpy(input_batch.data, input.data + b * seq_len * d_model, 
                   seq_len * d_model * sizeof(float), cudaMemcpyDeviceToDevice);
        
        // Q = input * W_q
        Tensor::matmul(input_batch, W_q, Q_batch, cublas_handle);
        Tensor::add(Q_batch, bias_q, Q_batch);
        
        // K = input * W_k
        Tensor::matmul(input_batch, W_k, K_batch, cublas_handle);
        Tensor::add(K_batch, bias_k, K_batch);
        
        // V = input * W_v
        Tensor::matmul(input_batch, W_v, V_batch, cublas_handle);
        Tensor::add(V_batch, bias_v, V_batch);
        
        // Copy back to main tensors
        cudaMemcpy(Q.data + b * seq_len * d_model, Q_batch.data,
                   seq_len * d_model * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(K.data + b * seq_len * d_model, K_batch.data,
                   seq_len * d_model * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(V.data + b * seq_len * d_model, V_batch.data,
                   seq_len * d_model * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    // Apply scaled dot-product attention with cuDNN
    scaled_dot_product_attention_cudnn(Q, K, V, output, mask);
}
#endif

void MultiHeadAttention::scaled_dot_product_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                                                      Tensor& output, const Tensor* mask) {
    int batch_size = Q.shape[0];
    int seq_len = Q.shape[1];
    int d_model = Q.shape[2];
    
    // Reshape Q, K, V to [batch_size, n_heads, seq_len, d_k]
    Tensor Q_heads({batch_size, config.n_heads, seq_len, config.d_k});
    Tensor K_heads({batch_size, config.n_heads, seq_len, config.d_k});
    Tensor V_heads({batch_size, config.n_heads, seq_len, config.d_k});
    
    // Transpose and reshape (simplified implementation)
    int block_size = 256;
    int grid_size = (batch_size * seq_len * config.n_heads * config.d_k + block_size - 1) / block_size;
    
    transpose_kernel<<<grid_size, block_size>>>(Q.data, Q_heads.data, batch_size, seq_len, config.n_heads, config.d_k);
    transpose_kernel<<<grid_size, block_size>>>(K.data, K_heads.data, batch_size, seq_len, config.n_heads, config.d_k);
    transpose_kernel<<<grid_size, block_size>>>(V.data, V_heads.data, batch_size, seq_len, config.n_heads, config.d_k);
    
    // Compute attention scores: Q * K^T
    Tensor scores({batch_size, config.n_heads, seq_len, seq_len});
    
    // Scale factor
    float scale = 1.0f / sqrtf(config.d_k);
    
    // For each batch and head, compute Q * K^T
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < config.n_heads; h++) {
            int offset_qk = (b * config.n_heads + h) * seq_len * config.d_k;
            int offset_scores = (b * config.n_heads + h) * seq_len * seq_len;
            
            Tensor Q_bh({seq_len, config.d_k});
            Tensor K_bh({seq_len, config.d_k});
            Tensor scores_bh({seq_len, seq_len});
            
            Q_bh.data = Q_heads.data + offset_qk;
            K_bh.data = K_heads.data + offset_qk;
            scores_bh.data = scores.data + offset_scores;
            
            // Compute Q * K^T (note: K needs to be transposed)
            const float alpha = scale, beta = 0.0f;
            cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        seq_len, seq_len, config.d_k,
                        &alpha,
                        K_bh.data, config.d_k,
                        Q_bh.data, config.d_k,
                        &beta,
                        scores_bh.data, seq_len);
        }
    }
    
    // Apply mask if provided
    if (mask) {
        int total_elements = batch_size * config.n_heads * seq_len * seq_len;
        int grid_size_mask = (total_elements + block_size - 1) / block_size;
        apply_mask_kernel<<<grid_size_mask, block_size>>>(scores.data, mask->data, 
                                                          batch_size, config.n_heads, seq_len);
    }
    
    // Apply softmax
    dim3 grid_softmax(seq_len, batch_size, config.n_heads);
    dim3 block_softmax(32);
    softmax_kernel<<<grid_softmax, block_softmax>>>(scores.data, batch_size, seq_len, config.d_k);
    
    // Compute final output: attention * V
    Tensor attention_output({batch_size, config.n_heads, seq_len, config.d_k});
    
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < config.n_heads; h++) {
            int offset_scores = (b * config.n_heads + h) * seq_len * seq_len;
            int offset_v = (b * config.n_heads + h) * seq_len * config.d_k;
            int offset_out = (b * config.n_heads + h) * seq_len * config.d_k;
            
            Tensor scores_bh({seq_len, seq_len});
            Tensor V_bh({seq_len, config.d_k});
            Tensor out_bh({seq_len, config.d_k});
            
            scores_bh.data = scores.data + offset_scores;
            V_bh.data = V_heads.data + offset_v;
            out_bh.data = attention_output.data + offset_out;
            
            // Compute scores * V
            const float alpha = 1.0f, beta = 0.0f;
            cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        config.d_k, seq_len, seq_len,
                        &alpha,
                        V_bh.data, config.d_k,
                        scores_bh.data, seq_len,
                        &beta,
                        out_bh.data, config.d_k);
        }
    }
    
    // Reshape back to [batch_size, seq_len, d_model] and apply output projection
    Tensor reshaped_output({batch_size, seq_len, d_model});
    
    // Transpose back (inverse of previous transpose)
    transpose_kernel<<<grid_size, block_size>>>(attention_output.data, reshaped_output.data, 
                                                batch_size, config.n_heads, seq_len, config.d_k);
    
    // Apply output projection: output = reshaped_output * W_o + bias_o
    for (int b = 0; b < batch_size; b++) {
        Tensor reshaped_batch({seq_len, d_model});
        Tensor output_batch({seq_len, d_model});
        
        reshaped_batch.data = reshaped_output.data + b * seq_len * d_model;
        output_batch.data = output.data + b * seq_len * d_model;
        
        Tensor::matmul(reshaped_batch, W_o, output_batch, cublas_handle);
        Tensor::add(output_batch, bias_o, output_batch);
    }
}

#ifdef USE_CUDNN
void MultiHeadAttention::scaled_dot_product_attention_cudnn(const Tensor& Q, const Tensor& K, const Tensor& V,
                                                           Tensor& output, const Tensor* mask) {
    int batch_size = Q.shape[0];
    int seq_len = Q.shape[1];
    int d_model = Q.shape[2];
    
    // Reshape Q, K, V to [batch_size, n_heads, seq_len, d_k]
    Tensor Q_heads({batch_size, config.n_heads, seq_len, config.d_k});
    Tensor K_heads({batch_size, config.n_heads, seq_len, config.d_k});
    Tensor V_heads({batch_size, config.n_heads, seq_len, config.d_k});
    
    // Transpose and reshape (simplified implementation)
    int block_size = 256;
    int grid_size = (batch_size * seq_len * config.n_heads * config.d_k + block_size - 1) / block_size;
    
    transpose_kernel<<<grid_size, block_size>>>(Q.data, Q_heads.data, batch_size, seq_len, config.n_heads, config.d_k);
    transpose_kernel<<<grid_size, block_size>>>(K.data, K_heads.data, batch_size, seq_len, config.n_heads, config.d_k);
    transpose_kernel<<<grid_size, block_size>>>(V.data, V_heads.data, batch_size, seq_len, config.n_heads, config.d_k);
    
    // Compute attention scores: Q * K^T
    Tensor scores({batch_size, config.n_heads, seq_len, seq_len});
    
    // Scale factor
    float scale = 1.0f / sqrtf(config.d_k);
    
    // For each batch and head, compute Q * K^T
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < config.n_heads; h++) {
            int offset_qk = (b * config.n_heads + h) * seq_len * config.d_k;
            int offset_scores = (b * config.n_heads + h) * seq_len * seq_len;
            
            Tensor Q_bh({seq_len, config.d_k});
            Tensor K_bh({seq_len, config.d_k});
            Tensor scores_bh({seq_len, seq_len});
            
            Q_bh.data = Q_heads.data + offset_qk;
            K_bh.data = K_heads.data + offset_qk;
            scores_bh.data = scores.data + offset_scores;
            
            // Compute Q * K^T (note: K needs to be transposed)
            const float alpha = scale, beta = 0.0f;
            cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        seq_len, seq_len, config.d_k,
                        &alpha,
                        K_bh.data, config.d_k,
                        Q_bh.data, config.d_k,
                        &beta,
                        scores_bh.data, seq_len);
        }
    }
    
    // Apply mask if provided
    if (mask) {
        int total_elements = batch_size * config.n_heads * seq_len * seq_len;
        int grid_size_mask = (total_elements + block_size - 1) / block_size;
        apply_mask_kernel<<<grid_size_mask, block_size>>>(scores.data, mask->data, 
                                                          batch_size, config.n_heads, seq_len);
    }
    
    // Apply softmax using standard CUDA (cuDNN version temporarily disabled)
    int total_elements = batch_size * config.n_heads * seq_len * seq_len;
    int softmax_block_size = 256;
    int softmax_grid_size = (total_elements + softmax_block_size - 1) / softmax_block_size;
    softmax_kernel<<<softmax_grid_size, softmax_block_size>>>(scores.data, batch_size, seq_len, config.d_k);
    cudaDeviceSynchronize();
    
    // Compute final output: attention * V
    Tensor attention_output({batch_size, config.n_heads, seq_len, config.d_k});
    
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < config.n_heads; h++) {
            int offset_scores = (b * config.n_heads + h) * seq_len * seq_len;
            int offset_v = (b * config.n_heads + h) * seq_len * config.d_k;
            int offset_out = (b * config.n_heads + h) * seq_len * config.d_k;
            
            Tensor scores_bh({seq_len, seq_len});
            Tensor V_bh({seq_len, config.d_k});
            Tensor out_bh({seq_len, config.d_k});
            
            scores_bh.data = scores.data + offset_scores;
            V_bh.data = V_heads.data + offset_v;
            out_bh.data = attention_output.data + offset_out;
            
            // Compute scores * V
            const float alpha = 1.0f, beta = 0.0f;
            cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        config.d_k, seq_len, seq_len,
                        &alpha,
                        V_bh.data, config.d_k,
                        scores_bh.data, seq_len,
                        &beta,
                        out_bh.data, config.d_k);
        }
    }
    
    // Reshape back to [batch_size, seq_len, d_model] and apply output projection
    Tensor reshaped_output({batch_size, seq_len, d_model});
    
    // Transpose back (inverse of previous transpose)
    transpose_kernel<<<grid_size, block_size>>>(attention_output.data, reshaped_output.data, 
                                                batch_size, config.n_heads, seq_len, config.d_k);
    
    // Apply output projection: output = reshaped_output * W_o + bias_o
    for (int b = 0; b < batch_size; b++) {
        Tensor reshaped_batch({seq_len, d_model});
        Tensor output_batch({seq_len, d_model});
        
        reshaped_batch.data = reshaped_output.data + b * seq_len * d_model;
        output_batch.data = output.data + b * seq_len * d_model;
        
        Tensor::matmul(reshaped_batch, W_o, output_batch, cublas_handle);
        Tensor::add(output_batch, bias_o, output_batch);
    }
}
#endif

std::vector<Tensor*> MultiHeadAttention::get_parameters() {
    return {&W_q, &W_k, &W_v, &W_o, &bias_q, &bias_k, &bias_v, &bias_o};
}