#pragma once
#include "transformer.h"
#include <vector>
#include <string>
#include <unordered_map>

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

struct TrainingConfig {
    float learning_rate = 3e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;
    int batch_size = 8;
    int seq_length = 512;
    int max_epochs = 10;
    int warmup_steps = 1000;
    int save_every = 1000;
    std::string data_path = "data/";
    std::string checkpoint_path = "checkpoints/";
};

class AdamOptimizer {
private:
    std::vector<Tensor> m_tensors;  // First moment estimates
    std::vector<Tensor> v_tensors;  // Second moment estimates
    TrainingConfig config;
    int step_count;
    
public:
    AdamOptimizer(const std::vector<Tensor*>& parameters, const TrainingConfig& config);
    void step(const std::vector<Tensor*>& parameters, const std::vector<Tensor>& gradients);
    void zero_grad(const std::vector<Tensor*>& parameters);
    float get_lr() const;
};

class Tokenizer {
private:
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> token_to_id;
    int pad_token_id = 0;
    int eos_token_id = 1;
    int unk_token_id = 2;
    
public:
    Tokenizer();
    void load_vocab(const std::string& vocab_path);
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& tokens);
    int get_vocab_size() const { return vocab.size(); }
};

class DataLoader {
private:
    std::vector<std::vector<int>> sequences;
    TrainingConfig config;
    int current_idx;
    
public:
    DataLoader(const TrainingConfig& config, const Tokenizer& tokenizer);
    void load_data(const std::string& data_path);
    bool get_next_batch(Tensor& input_ids, Tensor& labels);
    void shuffle();
    int get_num_batches() const;
};

class Trainer {
private:
    std::unique_ptr<GPTModel> model;
    std::unique_ptr<AdamOptimizer> optimizer;
    std::unique_ptr<DataLoader> dataloader;
    std::unique_ptr<Tokenizer> tokenizer;
    TrainingConfig config;
    
#ifdef USE_CUDNN
    cudnnHandle_t cudnn_handle;
    cudnnTensorDescriptor_t tensor_desc;
    cudnnActivationDescriptor_t activation_desc;
    cudnnDropoutDescriptor_t dropout_desc;
    void* dropout_states;
    size_t dropout_state_size;
#endif
    
public:
    Trainer(const ModelConfig& model_config, const TrainingConfig& train_config);
    ~Trainer();
    void train();
    void evaluate();
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);
    
private:
    float compute_loss(const Tensor& logits, const Tensor& labels);
    void compute_loss_gradient(const Tensor& logits, const Tensor& labels, Tensor& grad_logits);
    void init_cudnn();
    void cleanup_cudnn();
};