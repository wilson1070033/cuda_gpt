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
    
    // Check if CUDA is available
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    bool use_cuda = (error == cudaSuccess && device_count > 0);
    
    cout << "Using " << (use_cuda ? "CUDA" : "CPU") << " for tensor operations" << endl;
    
    // Test basic tensor operations
    Tensor a({2, 3}, use_cuda);
    Tensor b({2, 3}, use_cuda);
    Tensor c({2, 3}, use_cuda);
    
    a.random_fill(0.0f, 1.0f);
    b.random_fill(0.0f, 1.0f);
    
    Tensor::add(a, b, c);
    
    cout << "Tensor addition test completed" << endl;
    
    if (use_cuda) {
        // Test matrix multiplication only if CUDA is available
        cublasHandle_t handle;
        if (cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS) {
            Tensor x({2, 4}, use_cuda);
            Tensor y({4, 3}, use_cuda);
            Tensor z({2, 3}, use_cuda);
            
            x.random_fill(0.0f, 1.0f);
            y.random_fill(0.0f, 1.0f);
            
            Tensor::matmul(x, y, z, handle);
            
            cout << "Matrix multiplication test completed" << endl;
            
            cublasDestroy(handle);
        } else {
            cout << "cuBLAS initialization failed, skipping matrix multiplication test" << endl;
        }
    } else {
        cout << "Skipping matrix multiplication test (requires CUDA)" << endl;
    }
}

void train_smart_gpt() {
    cout << "\n=== Training Smart GPT Model ===" << endl;
    
    // Check if CUDA is available for training
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    bool use_cuda = (error == cudaSuccess && device_count > 0);
    
    if (!use_cuda) {
        cout << "Note: CUDA not available. Training would be extremely slow on CPU." << endl;
        cout << "Skipping training demonstration for now." << endl;
        return;
    }
    
    // Enhanced model configuration for intelligence
    ModelConfig model_config;
    model_config.vocab_size = 2000;     // Much larger vocabulary for better understanding
    model_config.d_model = 512;         // Larger model dimension for more capacity
    model_config.n_heads = 8;           // More attention heads for better pattern recognition
    model_config.n_layers = 6;          // More layers for deeper understanding
    model_config.d_ff = 2048;           // Larger feed-forward network
    model_config.max_seq_len = 1024;    // Longer sequences for better context
    model_config.dropout = 0.1f;        // Add dropout for regularization
    
    TrainingConfig train_config;
    train_config.batch_size = 8;        // Larger batch size for stable gradients
    train_config.seq_length = 512;      // Longer sequences for better context
    train_config.learning_rate = 3e-4f; // Optimized learning rate
    train_config.max_epochs = 20;       // More epochs for better learning
    train_config.warmup_steps = 2000;   // More warmup for stability
    
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
    
    // Check if CUDA is available
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    bool use_cuda = (error == cudaSuccess && device_count > 0);
    
    if (!use_cuda) {
        cout << "Note: CUDA not available. Skipping GPU-based model test." << endl;
        cout << "Testing basic tokenization instead..." << endl;
        
        try {
            auto tokenizer = make_unique<Tokenizer>();
            string prompt = "hello world";
            auto prompt_tokens = tokenizer->encode(prompt);
            
            cout << "Input prompt: \"" << prompt << "\"" << endl;
            cout << "Prompt tokens: ";
            for (int token : prompt_tokens) {
                cout << token << " ";
            }
            cout << endl;
            cout << "Tokenization test completed" << endl;
            
        } catch (const exception& e) {
            cerr << "Error during tokenization test: " << e.what() << endl;
        }
        return;
    }
    
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
    
    // Check if CUDA is available
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    bool use_cuda = (error == cudaSuccess && device_count > 0);
    
    try {
        unique_ptr<GPTModel> model = nullptr;
        auto tokenizer = make_unique<Tokenizer>();
        
        if (use_cuda) {
            ModelConfig config;
            config.vocab_size = 200;
            config.d_model = 128;
            config.n_heads = 4;
            config.n_layers = 2;
            config.d_ff = 256;
            config.max_seq_len = 64;
            
            model = make_unique<GPTModel>(config);
            cout << "Using GPU-accelerated model" << endl;
        } else {
            cout << "Using CPU-only tokenization mode" << endl;
        }
        
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
    train_smart_gpt();
    
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