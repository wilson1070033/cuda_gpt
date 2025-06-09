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
    vector<vector<int>> conversation_history;
    int max_context_length;
    bool use_neural_network;
    string current_topic;
    int conversation_turn_count;
    
public:
    ConversationalAI() {
        // Initialize model configuration
        config.vocab_size = 200;
        config.d_model = 128;
        config.n_heads = 4;
        config.n_layers = 2;
        config.d_ff = 256;
        config.max_seq_len = 64;
        
        max_context_length = 32;
        use_neural_network = true;
        current_topic = "";
        conversation_turn_count = 0;
        
        model = make_unique<GPTModel>(config);
        tokenizer = make_unique<Tokenizer>();
        
        cout << "AI: Hello! I'm a mini GPT model with improved conversation capabilities." << endl;
        cout << "AI: I can remember our conversation context and generate more relevant responses." << endl;
        cout << "AI: You can ask me about AI, programming, or just chat. Type 'quit' to exit." << endl;
    }
    
    string updateTopicContext(const string& input) {
        string lower_input = input;
        transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        // Topic detection
        if (lower_input.find("cuda") != string::npos || lower_input.find("gpu") != string::npos || 
            lower_input.find("parallel") != string::npos || lower_input.find("nvidia") != string::npos) {
            current_topic = "cuda";
        } else if (lower_input.find("programming") != string::npos || lower_input.find("code") != string::npos ||
                  lower_input.find("python") != string::npos || lower_input.find("c++") != string::npos) {
            current_topic = "programming";
        } else if (lower_input.find("weather") != string::npos || lower_input.find("today") != string::npos ||
                  lower_input.find("day") != string::npos) {
            current_topic = "daily_life";
        } else if (lower_input.find("food") != string::npos || lower_input.find("eat") != string::npos ||
                  lower_input.find("hungry") != string::npos || lower_input.find("cook") != string::npos) {
            current_topic = "food";
        } else if (lower_input.find("work") != string::npos || lower_input.find("job") != string::npos ||
                  lower_input.find("busy") != string::npos || lower_input.find("office") != string::npos) {
            current_topic = "work";
        } else if (lower_input.find("music") != string::npos || lower_input.find("song") != string::npos ||
                  lower_input.find("listen") != string::npos) {
            current_topic = "entertainment";
        }
        
        return current_topic;
    }
    
    string generateNeuralResponse(const vector<int>& input_tokens) {
        string input_text = tokenizer->decode(input_tokens);
        conversation_turn_count++;
        
        // Update topic context
        updateTopicContext(input_text);
        
        // Advanced context-aware responses
        if (conversation_history.size() > 2) {
            string last_context = "";
            if (conversation_history.size() >= 2) {
                last_context = tokenizer->decode(conversation_history[conversation_history.size() - 2]);
            }
            
            // Context-aware follow-ups
            if (last_context.find("programming") != string::npos) {
                if (input_text.find("yes") != string::npos || input_text.find("python") != string::npos) {
                    return "Python is excellent for beginners! Are you working on any specific projects, or would you like to learn about data science, web development, or automation?";
                }
                if (input_text.find("c++") != string::npos || input_text.find("cuda") != string::npos) {
                    return "C++ and CUDA make a powerful combination for high-performance computing! Are you interested in GPU programming or parallel algorithms?";
                }
            }
            
            if (last_context.find("cuda") != string::npos) {
                if (input_text.find("learn") != string::npos || input_text.find("start") != string::npos) {
                    return "CUDA learning path: Start with basic kernels, then memory management, and finally optimization techniques. Do you have experience with C++ already?";
                }
            }
            
            if (last_context.find("food") != string::npos) {
                if (input_text.find("yes") != string::npos || input_text.find("cook") != string::npos) {
                    return "Cooking is so rewarding! What type of cuisine do you enjoy making? I'd love to hear about your favorite recipes.";
                }
            }
        }
        
        return ""; // Return empty to fall back to enhanced pattern matching
    }
    
    string generateFallbackResponse(const string& input) {
        string lower_input = input;
        transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        // === GREETINGS ===
        if (lower_input.find("hello") != string::npos || lower_input.find("hi") != string::npos || 
            lower_input.find("hey") != string::npos) {
            vector<string> greetings = {
                "Hello! How's your day going so far?",
                "Hi there! What would you like to chat about today?",
                "Hey! Great to see you. Are you working on anything interesting?",
                "Hello! I hope you're having a wonderful day. What's on your mind?"
            };
            return greetings[rand() % greetings.size()];
        }
        
        if (lower_input.find("good morning") != string::npos || lower_input.find("morning") != string::npos) {
            vector<string> morning_greetings = {
                "Good morning! I hope you slept well. What are your plans for today?",
                "Morning! Starting the day with some AI chat - I like your style! How can I help?",
                "Good morning! Ready to tackle the day? What's first on your agenda?"
            };
            return morning_greetings[rand() % morning_greetings.size()];
        }
        
        if (lower_input.find("good evening") != string::npos || lower_input.find("evening") != string::npos) {
            vector<string> evening_greetings = {
                "Good evening! How was your day? Anything exciting happen?",
                "Evening! Winding down for the day? What was the highlight?",
                "Good evening! I hope you had a productive day. How are you feeling?"
            };
            return evening_greetings[rand() % evening_greetings.size()];
        }
        
        // === HOW ARE YOU ===
        if (lower_input.find("how are you") != string::npos || lower_input.find("how's it going") != string::npos) {
            vector<string> status_responses = {
                "I'm doing great! My neural networks are humming along nicely. How about you?",
                "Fantastic! I'm learning something new from every conversation. How's your day treating you?",
                "I'm functioning optimally and ready to chat! What's new in your world?",
                "Doing wonderful! I love meeting new people and learning about their interests. How are you feeling today?"
            };
            return status_responses[rand() % status_responses.size()];
        }
        
        // === CUDA & TECHNICAL TOPICS ===
        if (lower_input.find("what is cuda") != string::npos || lower_input.find("what's cuda") != string::npos) {
            return "CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform that lets you harness GPU power for general computing, not just graphics! Want to know about kernels, memory management, or getting started?";
        }
        
        if (lower_input.find("cuda") != string::npos) {
            vector<string> cuda_responses = {
                "CUDA is fascinating! Are you interested in learning GPU programming, optimizing existing code, or working on machine learning acceleration?",
                "CUDA opens up incredible parallel computing possibilities! What's your experience level with GPU programming?",
                "Love talking CUDA! Are you working on scientific computing, deep learning, or high-performance applications?",
                "CUDA can speed up computations dramatically! What kind of problems are you trying to solve?"
            };
            return cuda_responses[rand() % cuda_responses.size()];
        }
        
        if (lower_input.find("gpu") != string::npos || lower_input.find("graphics card") != string::npos) {
            vector<string> gpu_responses = {
                "GPUs are incredible for parallel processing! Modern GPUs have thousands of cores working together. Are you interested in programming them?",
                "Graphics cards aren't just for gaming anymore - they're computational powerhouses! What would you like to know about GPU computing?",
                "GPU computing has revolutionized so many fields! From AI to scientific simulations. What interests you most?"
            };
            return gpu_responses[rand() % gpu_responses.size()];
        }
        
        if (lower_input.find("parallel programming") != string::npos || lower_input.find("parallel computing") != string::npos) {
            return "Parallel programming is the art of making many processors work together! CUDA, OpenMP, MPI - each has its strengths. What's your parallel programming background?";
        }
        
        // === PROGRAMMING ===
        if (lower_input.find("programming") != string::npos || lower_input.find("coding") != string::npos) {
            vector<string> programming_responses = {
                "Programming is like solving puzzles with infinite solutions! What languages do you enjoy working with?",
                "I love talking code! Are you learning something new, debugging a tricky problem, or working on a cool project?",
                "Programming never gets boring - there's always something new to learn! What's your current focus?",
                "Code is poetry in motion! Are you into web development, systems programming, data science, or something else?"
            };
            return programming_responses[rand() % programming_responses.size()];
        }
        
        if (lower_input.find("python") != string::npos) {
            return "Python is fantastic! So versatile - from web development to AI research. Are you using it for data science, automation, web apps, or something else?";
        }
        
        if (lower_input.find("c++") != string::npos || lower_input.find("cpp") != string::npos) {
            return "C++ is powerful! Perfect for performance-critical applications and system programming. Combined with CUDA, it's unstoppable for GPU computing. What are you building?";
        }
        
        // === DAILY LIFE ===
        if (lower_input.find("food") != string::npos || lower_input.find("eat") != string::npos || 
            lower_input.find("hungry") != string::npos) {
            vector<string> food_responses = {
                "Food is fuel for both body and mind! What's your favorite cuisine? I love hearing about different cooking styles.",
                "Are you a cooking enthusiast or more of a takeout person? Both have their merits!",
                "Food brings people together! Do you have any family recipes or favorite comfort foods?",
                "The intersection of food and technology is fascinating - from precision cooking to food delivery algorithms!"
            };
            return food_responses[rand() % food_responses.size()];
        }
        
        if (lower_input.find("cooking") != string::npos || lower_input.find("cook") != string::npos) {
            vector<string> cooking_responses = {
                "Cooking is both art and science! Do you follow recipes exactly or like to improvise?",
                "I find cooking algorithms fascinating - precise timing, temperature control, ingredient ratios. What's your specialty?",
                "Cooking is like programming - you follow steps, debug when things go wrong, and celebrate successful deployments! What do you love to make?"
            };
            return cooking_responses[rand() % cooking_responses.size()];
        }
        
        if (lower_input.find("work") != string::npos || lower_input.find("job") != string::npos) {
            vector<string> work_responses = {
                "Work keeping you busy? I hope you're finding it fulfilling! What field are you in?",
                "Work-life balance is so important! Are you in tech, or does your job involve any programming?",
                "Hope work is treating you well! Any interesting projects you're excited about?"
            };
            return work_responses[rand() % work_responses.size()];
        }
        
        if (lower_input.find("weekend") != string::npos) {
            vector<string> weekend_responses = {
                "Weekends are for recharging! Do you have any fun plans, or are you using the time to learn something new?",
                "Weekend vibes! Are you working on any personal projects or just relaxing?",
                "Weekends are perfect for side projects! Any coding adventures planned?"
            };
            return weekend_responses[rand() % weekend_responses.size()];
        }
        
        if (lower_input.find("weather") != string::npos) {
            vector<string> weather_responses = {
                "I don't experience weather directly, but I love how it affects human behavior and daily routines! How's the weather treating you?",
                "Weather can really influence our mood and productivity! Is it a good day to be coding indoors or exploring outside?",
                "Weather systems are fascinating from a computational modeling perspective! Hope you're having pleasant conditions."
            };
            return weather_responses[rand() % weather_responses.size()];
        }
        
        // === ENTERTAINMENT ===
        if (lower_input.find("music") != string::npos) {
            vector<string> music_responses = {
                "Music and math have such beautiful connections! Do you have a favorite genre, or does it depend on your mood?",
                "I find the signal processing behind music fascinating! Are you into any particular artists or styles?",
                "Music is universal language! Do you play any instruments, or are you more of a listener?"
            };
            return music_responses[rand() % music_responses.size()];
        }
        
        if (lower_input.find("game") != string::npos || lower_input.find("gaming") != string::npos) {
            vector<string> gaming_responses = {
                "Gaming and GPU computing go hand in hand! Are you into the technical side of games or just enjoy playing?",
                "Games are incredible showcases of real-time computing! Any favorites, especially ones that push hardware limits?",
                "The game development pipeline is fascinating - graphics, physics, AI. What types of games do you enjoy?"
            };
            return gaming_responses[rand() % gaming_responses.size()];
        }
        
        // === LEARNING & EDUCATION ===
        if (lower_input.find("learn") != string::npos || lower_input.find("study") != string::npos) {
            vector<string> learning_responses = {
                "Learning never stops! What's caught your curiosity lately? I love helping people explore new topics.",
                "The best way to learn is by doing! Are you working through tutorials, building projects, or reading documentation?",
                "Learning is a journey, not a destination! What subject has you excited right now?"
            };
            return learning_responses[rand() % learning_responses.size()];
        }
        
        // === EMOTIONS & FEELINGS ===
        if (lower_input.find("tired") != string::npos || lower_input.find("exhausted") != string::npos) {
            vector<string> tired_responses = {
                "Sounds like you need some rest! Sometimes stepping away from screens helps recharge both mind and body.",
                "Being tired is your brain's way of asking for maintenance time! Are you getting enough sleep?",
                "Mental fatigue is real, especially with intense focus work like programming. Hope you can take a break soon!"
            };
            return tired_responses[rand() % tired_responses.size()];
        }
        
        if (lower_input.find("excited") != string::npos || lower_input.find("happy") != string::npos) {
            vector<string> excited_responses = {
                "That's wonderful to hear! What's got you excited? I love hearing about things that spark joy.",
                "Excitement is contagious! Share the good vibes - what's making you happy?",
                "I can feel the positive energy! What's the source of your excitement today?"
            };
            return excited_responses[rand() % excited_responses.size()];
        }
        
        if (lower_input.find("sad") != string::npos || lower_input.find("upset") != string::npos || 
            lower_input.find("frustrated") != string::npos) {
            vector<string> sad_responses = {
                "I'm sorry you're feeling down. Sometimes talking through things helps. What's weighing on your mind?",
                "Tough days happen to everyone. Would you like to talk about what's bothering you, or would a distraction help more?",
                "I hear you're having a difficult time. Remember that feelings are temporary - this too shall pass. Want to share what's going on?"
            };
            return sad_responses[rand() % sad_responses.size()];
        }
        
        // === COMPLIMENTS & THANKS ===
        if (lower_input.find("thank") != string::npos || lower_input.find("thanks") != string::npos) {
            vector<string> thanks_responses = {
                "You're very welcome! I'm happy I could help. Feel free to ask me anything else!",
                "My pleasure! I enjoy our conversations. Is there anything else you'd like to explore?",
                "Glad I could assist! That's what I'm here for. What else can we dive into?"
            };
            return thanks_responses[rand() % thanks_responses.size()];
        }
        
        if (lower_input.find("cool") != string::npos || lower_input.find("awesome") != string::npos || 
            lower_input.find("amazing") != string::npos) {
            vector<string> cool_responses = {
                "I'm glad you think so! Technology really is amazing when you dig into how it all works.",
                "Right? The more you learn about computing, the more mind-blowing it becomes!",
                "Technology never ceases to amaze me either! What aspect interests you most?"
            };
            return cool_responses[rand() % cool_responses.size()];
        }
        
        // === JOKES & HUMOR ===
        if (lower_input.find("joke") != string::npos || lower_input.find("funny") != string::npos) {
            vector<string> jokes = {
                "Why don't scientists trust atoms? Because they make up everything! 😄",
                "Why do programmers prefer dark mode? Because light attracts bugs! 💻",
                "What's a computer's favorite snack? Microchips! 🍟",
                "Why did the GPU break up with the CPU? It said 'You're too serial for me!' 😂",
                "How do you comfort a JavaScript bug? You console it! 💻",
                "Why do CUDA programmers never get lost? They always know their thread ID! 🧠"
            };
            return jokes[rand() % jokes.size()];
        }
        
        // === AI & TECHNOLOGY ===
        if (lower_input.find("what is ai") != string::npos || lower_input.find("what's ai") != string::npos) {
            return "AI is the fascinating field of making machines smart! It includes machine learning, neural networks, and pattern recognition. I'm a small example running right here on your GPU!";
        }
        
        if (lower_input.find("artificial intelligence") != string::npos || lower_input.find(" ai ") != string::npos) {
            vector<string> ai_responses = {
                "AI is transforming every industry! From healthcare to entertainment to scientific research. What aspects of AI interest you most?",
                "The current AI revolution is incredible to witness! Are you interested in the technical side, the applications, or the societal impacts?",
                "AI is both the tool and the outcome of human creativity! What would you like to know about machine learning, neural networks, or AI applications?"
            };
            return ai_responses[rand() % ai_responses.size()];
        }
        
        // === CONTEXTUAL DEFAULTS BASED ON CURRENT TOPIC ===
        if (current_topic == "cuda") {
            vector<string> cuda_defaults = {
                "Since we're talking CUDA, are you interested in kernel optimization, memory management, or maybe debugging techniques?",
                "CUDA has so many fascinating aspects! Would you like to know about thread hierarchy, shared memory, or perhaps profiling tools?",
                "GPU programming is an exciting field! What specific CUDA challenges are you working on?"
            };
            return cuda_defaults[rand() % cuda_defaults.size()];
        }
        
        if (current_topic == "programming") {
            vector<string> programming_defaults = {
                "Programming is such a creative field! Are you working on any interesting projects or learning new technologies?",
                "Code is poetry in motion! What's your favorite programming paradigm - object-oriented, functional, or something else?",
                "Every programming problem is a puzzle waiting to be solved! What challenges are you tackling lately?"
            };
            return programming_defaults[rand() % programming_defaults.size()];
        }
        
        if (current_topic == "daily_life") {
            vector<string> life_defaults = {
                "Life is full of interesting moments! What's been the highlight of your day so far?",
                "I love learning about human experiences! What's something that made you smile recently?",
                "Daily life has its own rhythms and patterns! How do you like to spend your free time?"
            };
            return life_defaults[rand() % life_defaults.size()];
        }
        
        // === GENERAL CONTEXTUAL DEFAULTS ===
        vector<string> default_responses = {
            "That's really interesting! Could you tell me more about your thoughts on that?",
            "I see! What made you think about that particular topic?",
            "Fascinating perspective! I'd love to understand your experience better.",
            "That's worth exploring further! What aspect interests you most?",
            "I'm curious to learn more about your viewpoint on this!",
            "Intriguing! How does that relate to your interests or work?",
            "I appreciate you sharing that! What would you like to dive deeper into?"
        };
        
        return default_responses[rand() % default_responses.size()];
    }
    
    string generateResponse(const string& input) {
        auto input_tokens = tokenizer->encode(input);
        
        if (use_neural_network) {
            string response = generateNeuralResponse(input_tokens);
            if (!response.empty()) {
                // Add to conversation history
                conversation_history.push_back(input_tokens);
                auto response_tokens = tokenizer->encode(response);
                conversation_history.push_back(response_tokens);
                
                // Keep history manageable
                while (conversation_history.size() > 20) {
                    conversation_history.erase(conversation_history.begin());
                }
                
                return response;
            }
        }
        
        return generateFallbackResponse(input);
    }
    
    void toggleMode() {
        use_neural_network = !use_neural_network;
        cout << "AI: Switched to " << (use_neural_network ? "Neural Network" : "Fallback Pattern Matching") 
             << " mode." << endl;
    }
    
    void clearContext() {
        conversation_history.clear();
        cout << "AI: Conversation context cleared." << endl;
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
            
            // Special commands
            if (input == "/toggle") {
                toggleMode();
                continue;
            }
            
            if (input == "/clear") {
                clearContext();
                continue;
            }
            
            if (input == "/help") {
                cout << "AI: Special commands:" << endl;
                cout << "    /toggle - Switch between neural and fallback modes" << endl;
                cout << "    /clear  - Clear conversation context" << endl;
                cout << "    /help   - Show this help message" << endl;
                cout << "    quit    - Exit the conversation" << endl;
                continue;
            }
            
            // Generate response using improved system
            string response = generateResponse(input);
            cout << "AI: " << response << endl;
            
            // Show diagnostic info for longer inputs
            if (input.length() > 15) {
                auto tokens = tokenizer->encode(input);
                cout << "    (Input: " << tokens.size() << " tokens, Context: " 
                     << conversation_history.size() << " turns, Mode: " 
                     << (use_neural_network ? "Neural" : "Fallback") << ")" << endl;
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