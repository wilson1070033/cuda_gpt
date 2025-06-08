#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "transformer.h"
#include "training.h"

using namespace std;

class ConversationalAI {
private:
    unique_ptr<GPTModel> model;
    unique_ptr<Tokenizer> tokenizer;
    ModelConfig config;
    
public:
    ConversationalAI() {
        // Initialize model configuration
        config.vocab_size = 200;
        config.d_model = 128;
        config.n_heads = 4;
        config.n_layers = 2;
        config.d_ff = 256;
        config.max_seq_len = 64;
        
        model = make_unique<GPTModel>(config);
        tokenizer = make_unique<Tokenizer>();
        
        cout << "AI: Hello! I'm a mini GPT model. I'm still learning, so my responses will be simple." << endl;
        cout << "AI: You can ask me about AI, programming, or just chat. Type 'quit' to exit." << endl;
    }
    
    string generateResponse(const string& input) {
        // Simple pattern matching for demo since model is untrained
        string lower_input = input;
        transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        // Greetings
        if (lower_input.find("hello") != string::npos || lower_input.find("hi") != string::npos) {
            return "Hello! How can I help you today?";
        }
        
        // Questions about AI
        if (lower_input.find("what is ai") != string::npos || lower_input.find("what's ai") != string::npos) {
            return "AI stands for Artificial Intelligence. It's technology that enables machines to simulate human intelligence!";
        }
        
        if (lower_input.find("cuda") != string::npos) {
            return "CUDA is NVIDIA's parallel computing platform that allows us to use GPU acceleration for tasks like machine learning!";
        }
        
        if (lower_input.find("how are you") != string::npos) {
            return "I'm doing well! I'm a neural network running on your GPU. How are you doing?";
        }
        
        if (lower_input.find("programming") != string::npos || lower_input.find("code") != string::npos) {
            return "I love talking about programming! Are you working on any coding projects?";
        }
        
        if (lower_input.find("joke") != string::npos) {
            return "Why don't scientists trust atoms? Because they make up everything! 😄";
        }
        
        if (lower_input.find("sad") != string::npos || lower_input.find("upset") != string::npos) {
            return "I'm sorry to hear you're feeling down. Sometimes talking helps. What's on your mind?";
        }
        
        if (lower_input.find("thank") != string::npos) {
            return "You're very welcome! I'm happy to help. Is there anything else you'd like to know?";
        }
        
        if (lower_input.find("cool") != string::npos || lower_input.find("awesome") != string::npos) {
            return "I'm glad you think so! Technology and AI are really fascinating fields.";
        }
        
        // Default response
        vector<string> default_responses = {
            "That's interesting! Tell me more about that.",
            "I see. What made you think about that?",
            "That's a good point. I'm still learning about many topics.",
            "Interesting perspective! I'd love to learn more about your thoughts.",
            "I'm processing what you said. My neural network is still growing!"
        };
        
        return default_responses[rand() % default_responses.size()];
    }
    
    void startConversation() {
        string input;
        
        while (true) {
            cout << "\nHuman: ";
            getline(cin, input);
            
            if (input.empty()) continue;
            
            if (input == "quit" || input == "exit" || input == "bye") {
                cout << "AI: Goodbye! Thanks for chatting with me. Have a great day! 🤖" << endl;
                break;
            }
            
            // Tokenize input (for demonstration)
            auto tokens = tokenizer->encode(input);
            
            // Generate response
            string response = generateResponse(input);
            cout << "AI: " << response << endl;
            
            // Show tokenization info occasionally
            if (input.length() > 10) {
                cout << "    (Input tokenized to " << tokens.size() << " tokens)" << endl;
            }
        }
    }
};

int main() {
    cout << "=== CUDA GPT Conversational AI ===" << endl;
    cout << "Mini GPT model optimized for RTX 3050 Mobile" << endl;
    cout << "Running on WSL Ubuntu with CUDA acceleration" << endl;
    cout << "\nInitializing AI..." << endl;
    
    // Initialize CUDA (optional for conversation demo)
    cudaError_t error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        cout << "Note: CUDA not available (" << cudaGetErrorString(error) << ")" << endl;
        cout << "Running conversation demo without GPU acceleration" << endl;
    } else {
        cout << "CUDA initialization successful!" << endl;
    }
    
    try {
        ConversationalAI ai;
        ai.startConversation();
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}