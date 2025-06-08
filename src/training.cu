#include "training.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__global__ void adam_update_kernel(float* param, const float* grad, float* m, float* v,
                                   float lr, float beta1, float beta2, float eps, 
                                   float beta1_corrected, float beta2_corrected, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad[idx];
        
        // Update biased second raw moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad[idx] * grad[idx];
        
        // Compute bias-corrected first moment estimate
        float m_hat = m[idx] / (1.0f - beta1_corrected);
        
        // Compute bias-corrected second raw moment estimate
        float v_hat = v[idx] / (1.0f - beta2_corrected);
        
        // Update parameters
        param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

__global__ void zero_gradients_kernel(float* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 0.0f;
    }
}

__global__ void cross_entropy_loss_kernel(const float* logits, const int* labels, float* loss,
                                          int batch_size, int seq_len, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tokens = batch_size * seq_len;
    
    if (idx < total_tokens) {
        int batch_idx = idx / seq_len;
        int seq_idx = idx % seq_len;
        (void)batch_idx; (void)seq_idx; // Suppress unused variable warnings
        
        int logits_offset = idx * vocab_size;
        int label = labels[idx];
        
        if (label >= 0 && label < vocab_size) {  // Ignore padding tokens (negative labels)
            // Find max for numerical stability
            float max_logit = logits[logits_offset];
            for (int i = 1; i < vocab_size; i++) {
                max_logit = fmaxf(max_logit, logits[logits_offset + i]);
            }
            
            // Compute log softmax
            float sum_exp = 0.0f;
            for (int i = 0; i < vocab_size; i++) {
                sum_exp += expf(logits[logits_offset + i] - max_logit);
            }
            float log_sum_exp = logf(sum_exp) + max_logit;
            
            // Cross entropy loss
            loss[idx] = -(logits[logits_offset + label] - log_sum_exp);
        } else {
            loss[idx] = 0.0f;  // Ignore this token
        }
    }
}

__global__ void cross_entropy_gradient_kernel(const float* logits, const int* labels, float* grad_logits,
                                              int batch_size, int seq_len, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * vocab_size;
    
    if (idx < total_elements) {
        int token_idx = idx / vocab_size;
        int vocab_idx = idx % vocab_size;
        
        int batch_idx = token_idx / seq_len;
        int seq_idx = token_idx % seq_len;
        (void)batch_idx; (void)seq_idx; // Suppress unused variable warnings
        
        int label = labels[token_idx];
        
        if (label >= 0 && label < vocab_size) {  // Valid token
            int logits_offset = token_idx * vocab_size;
            
            // Compute softmax for this token
            float max_logit = logits[logits_offset];
            for (int i = 1; i < vocab_size; i++) {
                max_logit = fmaxf(max_logit, logits[logits_offset + i]);
            }
            
            float sum_exp = 0.0f;
            for (int i = 0; i < vocab_size; i++) {
                sum_exp += expf(logits[logits_offset + i] - max_logit);
            }
            
            float softmax = expf(logits[idx] - max_logit) / sum_exp;
            
            // Gradient of cross entropy loss
            if (vocab_idx == label) {
                grad_logits[idx] = softmax - 1.0f;
            } else {
                grad_logits[idx] = softmax;
            }
        } else {
            grad_logits[idx] = 0.0f;  // Ignore this token
        }
    }
}

AdamOptimizer::AdamOptimizer(const std::vector<Tensor*>& parameters, const TrainingConfig& config)
    : config(config), step_count(0) {
    
    // Initialize momentum and velocity tensors
    for (auto* param : parameters) {
        m_tensors.emplace_back(param->get_shape());
        v_tensors.emplace_back(param->get_shape());
        
        m_tensors.back().zero();
        v_tensors.back().zero();
    }
}

void AdamOptimizer::step(const std::vector<Tensor*>& parameters, const std::vector<Tensor>& gradients) {
    step_count++;
    
    float beta1_corrected = powf(config.beta1, step_count);
    float beta2_corrected = powf(config.beta2, step_count);
    float lr = get_lr();
    
    for (size_t i = 0; i < parameters.size(); i++) {
        int size = parameters[i]->get_size();
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        
        adam_update_kernel<<<grid_size, block_size>>>(
            parameters[i]->get_data(),
            gradients[i].get_data(),
            m_tensors[i].get_data(),
            v_tensors[i].get_data(),
            lr, config.beta1, config.beta2, config.eps,
            beta1_corrected, beta2_corrected, size
        );
    }
    
    cudaDeviceSynchronize();
}

void AdamOptimizer::zero_grad(const std::vector<Tensor*>& parameters) {
    for (auto* param : parameters) {
        int size = param->get_size();
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        
        zero_gradients_kernel<<<grid_size, block_size>>>(param->get_data(), size);
    }
    cudaDeviceSynchronize();
}

float AdamOptimizer::get_lr() const {
    if (step_count <= config.warmup_steps) {
        // Linear warmup
        return config.learning_rate * static_cast<float>(step_count) / config.warmup_steps;
    } else {
        // Cosine decay (simplified)
        return config.learning_rate * 0.5f * (1.0f + cosf(M_PI * step_count / 10000.0f));
    }
}

Tokenizer::Tokenizer() {
    // Initialize basic vocabulary
    vocab = {"<pad>", "<eos>", "<unk>"};
    token_to_id["<pad>"] = 0;
    token_to_id["<eos>"] = 1;
    token_to_id["<unk>"] = 2;
    
    // Add basic ASCII characters
    for (char c = 32; c < 127; c++) {
        std::string token(1, c);
        vocab.push_back(token);
        token_to_id[token] = vocab.size() - 1;
    }
    
    // Add common words (simplified)
    std::vector<std::string> common_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "this", "that", "these", "those", "here", "there", "where", "when", "why", "how",
        "what", "who", "which", "whose", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "can",
        "hello", "world", "computer", "program", "code", "function", "variable", "class"
    };
    
    for (const auto& word : common_words) {
        if (token_to_id.find(word) == token_to_id.end()) {
            vocab.push_back(word);
            token_to_id[word] = vocab.size() - 1;
        }
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> tokens;
    std::istringstream iss(text);
    std::string word;
    
    while (iss >> word) {
        // Convert to lowercase
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        
        // Remove punctuation (simplified)
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
        
        if (!word.empty()) {
            auto it = token_to_id.find(word);
            if (it != token_to_id.end()) {
                tokens.push_back(it->second);
            } else {
                tokens.push_back(unk_token_id);  // Unknown token
            }
        }
    }
    
    return tokens;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) {
    std::string text;
    for (int token_id : tokens) {
        if (token_id >= 0 && token_id < static_cast<int>(vocab.size())) {
            if (!text.empty()) text += " ";
            text += vocab[token_id];
        }
    }
    return text;
}

DataLoader::DataLoader(const TrainingConfig& config, const Tokenizer& tokenizer)
    : config(config), current_idx(0) {
}

void DataLoader::load_data(const std::string& data_path) {
    std::ifstream file(data_path);
    if (!file.is_open()) {
        std::cout << "Warning: Could not open " << data_path << std::endl;
        std::cout << "Creating dummy conversational data..." << std::endl;
        // Create dummy conversational data
        for (int i = 0; i < 100; i++) {
            std::vector<int> dummy_seq(config.seq_length);
            for (int j = 0; j < config.seq_length; j++) {
                dummy_seq[j] = rand() % 100 + 3;  // Smaller vocab range
            }
            sequences.push_back(dummy_seq);
        }
        return;
    }
    
    std::cout << "Loading conversational training data from " << data_path << std::endl;
    
    std::string line;
    Tokenizer tokenizer;
    
    while (std::getline(file, line) && sequences.size() < 10000) {  // Limit dataset size
        auto tokens = tokenizer.encode(line);
        if (tokens.size() >= config.seq_length) {
            // Split long sequences
            for (size_t i = 0; i + config.seq_length <= tokens.size(); i += config.seq_length) {
                std::vector<int> seq(tokens.begin() + i, tokens.begin() + i + config.seq_length);
                sequences.push_back(seq);
            }
        } else if (tokens.size() > 10) {  // Only use sequences with reasonable length
            // Pad short sequences
            tokens.resize(config.seq_length, 0);  // Pad with pad token
            sequences.push_back(tokens);
        }
    }
    
    std::cout << "Loaded " << sequences.size() << " sequences." << std::endl;
}

bool DataLoader::get_next_batch(Tensor& input_ids, Tensor& labels) {
    if (current_idx + config.batch_size > sequences.size()) {
        return false;  // No more batches
    }
    
    // Copy data to tensors
    input_ids.to_host();
    labels.to_host();
    
    for (int b = 0; b < config.batch_size; b++) {
        const auto& seq = sequences[current_idx + b];
        
        for (int s = 0; s < config.seq_length; s++) {
            int input_idx = b * config.seq_length + s;
            
            if (s < config.seq_length - 1) {
                input_ids.get_data()[input_idx] = static_cast<float>(seq[s]);
                labels.get_data()[input_idx] = static_cast<float>(seq[s + 1]);
            } else {
                input_ids.get_data()[input_idx] = static_cast<float>(seq[s]);
                labels.get_data()[input_idx] = 1.0f;  // EOS token
            }
        }
    }
    
    input_ids.to_device();
    labels.to_device();
    
    current_idx += config.batch_size;
    return true;
}

int DataLoader::get_num_batches() const {
    return (sequences.size() + config.batch_size - 1) / config.batch_size;
}

void DataLoader::shuffle() {
    std::shuffle(sequences.begin(), sequences.end(), std::mt19937{std::random_device{}()});
    current_idx = 0;
}

Trainer::Trainer(const ModelConfig& model_config, const TrainingConfig& train_config)
    : config(train_config) {
    
    model = std::make_unique<GPTModel>(model_config);
    tokenizer = std::make_unique<Tokenizer>();
    dataloader = std::make_unique<DataLoader>(train_config, *tokenizer);
    
    auto parameters = model->get_all_parameters();
    optimizer = std::make_unique<AdamOptimizer>(parameters, train_config);
    
    // Load training data - check multiple possible paths
    std::string data_file;
    if (!train_config.data_path.empty()) {
        data_file = train_config.data_path + "train.txt";
    } else {
        // Try different paths depending on where program is run from
        std::vector<std::string> possible_paths = {
            "data/train.txt",           // From project root
            "../data/train.txt",        // From build directory
            "../../data/train.txt"      // From nested build directory
        };
        
        data_file = "data/train.txt"; // Default
        for (const auto& path : possible_paths) {
            std::ifstream test_file(path);
            if (test_file.is_open()) {
                data_file = path;
                test_file.close();
                break;
            }
        }
    }
    dataloader->load_data(data_file);
}

void Trainer::train() {
    auto parameters = model->get_all_parameters();
    
    for (int epoch = 0; epoch < config.max_epochs; epoch++) {
        std::cout << "Epoch " << epoch + 1 << "/" << config.max_epochs << std::endl;
        
        dataloader->shuffle();
        float total_loss = 0.0f;
        int num_batches = 0;
        
        Tensor input_ids({config.batch_size, config.seq_length});
        Tensor labels({config.batch_size, config.seq_length});
        Tensor logits({config.batch_size, config.seq_length, tokenizer->get_vocab_size()});
        
        std::cout << "Starting training batches..." << std::endl;
        
        while (dataloader->get_next_batch(input_ids, labels)) {
            std::cout << "Processing batch " << num_batches + 1 << std::endl;
            
            try {
                // Forward pass
                model->forward(input_ids, logits);
                
                // Compute loss
                float batch_loss = compute_loss(logits, labels);
                total_loss += batch_loss;
                num_batches++;
                
                std::cout << "Batch " << num_batches << ", Loss: " << batch_loss << std::endl;
                
                // Skip backward pass for now to avoid complexity
                // Just do a simple forward pass test
                
            } catch (const std::exception& e) {
                std::cerr << "Error in batch " << num_batches + 1 << ": " << e.what() << std::endl;
                break;
            }
            
            // Limit to just a few batches for testing
            if (num_batches >= 3) break;
        }
        
        float avg_loss = total_loss / num_batches;
        std::cout << "Average loss for epoch " << epoch + 1 << ": " << avg_loss << std::endl;
    }
}

float Trainer::compute_loss(const Tensor& logits, const Tensor& labels) {
    int batch_size = logits.shape[0];
    int seq_len = logits.shape[1];
    int vocab_size = logits.shape[2];
    
    Tensor loss_per_token({batch_size, seq_len});
    Tensor labels_int({batch_size, seq_len});
    
    // Convert float labels to int (simplified)
    Tensor labels_copy = labels.clone();
    labels_copy.to_host();
    labels_int.to_host();
    for (int i = 0; i < batch_size * seq_len; i++) {
        reinterpret_cast<int*>(labels_int.get_data())[i] = static_cast<int>(labels_copy.get_data()[i]);
    }
    labels_int.to_device();
    
    int block_size = 256;
    int grid_size = (batch_size * seq_len + block_size - 1) / block_size;
    
    cross_entropy_loss_kernel<<<grid_size, block_size>>>(
        logits.get_data(),
        reinterpret_cast<int*>(labels_int.get_data()),
        loss_per_token.get_data(),
        batch_size, seq_len, vocab_size
    );
    
    // Sum up the losses
    loss_per_token.to_host();
    float total_loss = 0.0f;
    int valid_tokens = 0;
    
    for (int i = 0; i < batch_size * seq_len; i++) {
        int label = static_cast<int>(labels_copy.get_data()[i]);
        if (label >= 0) {  // Valid token
            total_loss += loss_per_token.get_data()[i];
            valid_tokens++;
        }
    }
    
    return valid_tokens > 0 ? total_loss / valid_tokens : 0.0f;
}

void Trainer::compute_loss_gradient(const Tensor& logits, const Tensor& labels, Tensor& grad_logits) {
    int batch_size = logits.shape[0];
    int seq_len = logits.shape[1];
    int vocab_size = logits.shape[2];
    
    Tensor labels_int({batch_size, seq_len});
    
    // Convert float labels to int
    Tensor labels_copy = labels.clone();
    labels_copy.to_host();
    labels_int.to_host();
    for (int i = 0; i < batch_size * seq_len; i++) {
        reinterpret_cast<int*>(labels_int.get_data())[i] = static_cast<int>(labels_copy.get_data()[i]);
    }
    labels_int.to_device();
    
    int block_size = 256;
    int grid_size = (batch_size * seq_len * vocab_size + block_size - 1) / block_size;
    
    cross_entropy_gradient_kernel<<<grid_size, block_size>>>(
        logits.get_data(),
        reinterpret_cast<int*>(labels_int.get_data()),
        grad_logits.get_data(),
        batch_size, seq_len, vocab_size
    );
    
    cudaDeviceSynchronize();
}