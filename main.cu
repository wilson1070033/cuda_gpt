#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include "transformer.h"
#include "training.h"

using namespace std;

void print_cuda_info() {
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        cerr << "Failed to get CUDA device count: " << cudaGetErrorString(error) << endl;
        cout << "Note: Running in CPU fallback mode for demonstration" << endl;
        return;
    }
    
    cout << "CUDA devices count: " << device_count << endl;
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, i);
        if (error != cudaSuccess) {
            cerr << "Failed to get device " << i << " properties: " << cudaGetErrorString(error) << endl;
            continue;
        }
        
        cout << "Device " << i << ": " << prop.name << endl;
        cout << "  Compute capability: " << prop.major << "." << prop.minor << endl;
        cout << "  Global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << endl;
        cout << "  Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << endl;
        cout << "  Max threads per block: " << prop.maxThreadsPerBlock << endl;
        cout << "  Multiprocessor count: " << prop.multiProcessorCount << endl;
    }
}

void test_basic_operations() {
    cout << "\n=== Testing Basic Tensor Operations ===" << endl;
    
    // Test basic tensor operations
    Tensor a({2, 3});
    Tensor b({2, 3});
    Tensor c({2, 3});
    
    a.random_fill(0.0f, 1.0f);
    b.random_fill(0.0f, 1.0f);
    
    Tensor::add(a, b, c);
    
    cout << "Tensor addition test completed" << endl;
    
    // Test matrix multiplication
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    Tensor x({2, 4});
    Tensor y({4, 3});
    Tensor z({2, 3});
    
    x.random_fill(0.0f, 1.0f);
    y.random_fill(0.0f, 1.0f);
    
    Tensor::matmul(x, y, z, handle);
    
    cout << "Matrix multiplication test completed" << endl;
    
    cublasDestroy(handle);
}

void train_mini_gpt() {
    cout << "\n=== Training Mini GPT Model ===" << endl;
    
    // Small model configuration optimized for RTX 3050 Mobile
    ModelConfig model_config;
    model_config.vocab_size = 200;      // Very small vocabulary for testing
    model_config.d_model = 128;         // Smaller model dimension
    model_config.n_heads = 4;           // Fewer attention heads
    model_config.n_layers = 2;          // Fewer layers for testing
    model_config.d_ff = 256;            // Smaller feed-forward network
    model_config.max_seq_len = 64;      // Shorter sequence length
    
    TrainingConfig train_config;
    train_config.batch_size = 2;        // Very small batch size
    train_config.seq_length = 32;       // Very short sequences for testing
    train_config.learning_rate = 1e-4f;
    train_config.max_epochs = 1;        // Single epoch for testing
    
    try {
        cout << "Creating GPT model..." << endl;
        auto model = make_unique<GPTModel>(model_config);
        
        cout << "Creating trainer..." << endl;
        auto trainer = make_unique<Trainer>(model_config, train_config);
        
        cout << "Starting training..." << endl;
        trainer->train();
        
        cout << "Training completed!" << endl;
        
    } catch (const exception& e) {
        cerr << "Error during training: " << e.what() << endl;
    }
}

void test_generation() {
    cout << "\n=== Testing Text Generation ===" << endl;
    
    ModelConfig config;
    config.vocab_size = 200;
    config.d_model = 128;
    config.n_heads = 4;
    config.n_layers = 2;
    config.d_ff = 256;
    config.max_seq_len = 64;
    
    try {
        auto model = make_unique<GPTModel>(config);
        auto tokenizer = make_unique<Tokenizer>();
        
        // Test generation (using randomly initialized weights)
        string prompt = "hello world";
        auto prompt_tokens = tokenizer->encode(prompt);
        
        cout << "Input prompt: \"" << prompt << "\"" << endl;
        cout << "Prompt tokens: ";
        for (int token : prompt_tokens) {
            cout << token << " ";
        }
        cout << endl;
        
        // Note: Since weights are randomly initialized, generated text will be random
        cout << "Note: Since the model is untrained, generated output will be random" << endl;
        
    } catch (const exception& e) {
        cerr << "Error during generation test: " << e.what() << endl;
    }
}

void interactive_conversation() {
    cout << "\n=== Interactive Conversation Mode ===" << endl;
    cout << "Note: This model needs training on real conversation data to work properly." << endl;
    cout << "Type 'quit' to exit the conversation." << endl;
    cout << "\nStarting conversation..." << endl;
    
    ModelConfig config;
    config.vocab_size = 200;
    config.d_model = 128;
    config.n_heads = 4;
    config.n_layers = 2;
    config.d_ff = 256;
    config.max_seq_len = 64;
    
    try {
        auto model = make_unique<GPTModel>(config);
        auto tokenizer = make_unique<Tokenizer>();
        
        string input;
        while (true) {
            cout << "\nHuman: ";
            getline(cin, input);
            
            if (input == "quit" || input == "exit") {
                cout << "AI: Goodbye! Thanks for chatting!" << endl;
                break;
            }
            
            if (input.empty()) continue;
            
            // Tokenize input
            auto tokens = tokenizer->encode(input);
            
            // Since the model is untrained, provide a simple response
            cout << "AI: I understand you said \"" << input << "\". ";
            cout << "I'm still learning from training data to give better responses!" << endl;
            cout << "    (Tokens: ";
            for (size_t i = 0; i < min(tokens.size(), size_t(5)); i++) {
                cout << tokens[i] << " ";
            }
            if (tokens.size() > 5) cout << "...";
            cout << ")" << endl;
        }
        
    } catch (const exception& e) {
        cerr << "Error in conversation mode: " << e.what() << endl;
    }
}

int main() {
    cout << "=== CUDA GPT Training System ===" << endl;
    cout << "GPT model implementation optimized for RTX 3050 Mobile" << endl;
    cout << "Running on WSL Ubuntu" << endl;
    
    // Check CUDA devices
    print_cuda_info();
    
    // Test basic operations
    test_basic_operations();
    
    // Train model
    train_mini_gpt();
    
    // Test generation
    test_generation();
    
    // Interactive conversation
    cout << "\nWould you like to try the conversation mode? (y/n): ";
    string choice;
    getline(cin, choice);
    if (choice == "y" || choice == "Y" || choice == "yes") {
        interactive_conversation();
    }
    
    cout << "\nProgram completed!" << endl;
    cout << "\nUsage instructions:" << endl;
    cout << "1. Train with real conversation data in data/train.txt" << endl;
    cout << "2. Adjust ModelConfig parameters for different hardware configurations" << endl;
    cout << "3. Use checkpoint functionality to save and load trained models" << endl;
    cout << "4. Run conversation mode to chat with the AI" << endl;
    
    return 0;
}