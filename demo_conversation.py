#!/usr/bin/env python3
"""
Demo conversation script showing how the CUDA GPT chat works
This simulates the conversation functionality without requiring CUDA
"""

class DemoConversationalAI:
    def __init__(self):
        print("=== CUDA GPT Conversational AI Demo ===")
        print("Mini GPT model optimized for RTX 3050 Mobile")
        print("Running on WSL Ubuntu with CUDA acceleration")
        print("\nAI: Hello! I'm a mini GPT model. I'm still learning, so my responses will be simple.")
        print("AI: You can ask me about AI, programming, or just chat. Type 'quit' to exit.")
    
    def generate_response(self, input_text):
        """Generate response based on input patterns"""
        lower_input = input_text.lower()
        
        # Greetings
        if "hello" in lower_input or "hi" in lower_input:
            return "Hello! How can I help you today?"
        
        # Questions about AI
        if "what is ai" in lower_input or "what's ai" in lower_input:
            return "AI stands for Artificial Intelligence. It's technology that enables machines to simulate human intelligence!"
        
        if "cuda" in lower_input:
            return "CUDA is NVIDIA's parallel computing platform that allows us to use GPU acceleration for tasks like machine learning!"
        
        if "how are you" in lower_input:
            return "I'm doing well! I'm a neural network running on your GPU. How are you doing?"
        
        if "programming" in lower_input or "code" in lower_input:
            return "I love talking about programming! Are you working on any coding projects?"
        
        if "joke" in lower_input:
            return "Why don't scientists trust atoms? Because they make up everything! 😄"
        
        if "sad" in lower_input or "upset" in lower_input:
            return "I'm sorry to hear you're feeling down. Sometimes talking helps. What's on your mind?"
        
        if "thank" in lower_input:
            return "You're very welcome! I'm happy to help. Is there anything else you'd like to know?"
        
        if "cool" in lower_input or "awesome" in lower_input:
            return "I'm glad you think so! Technology and AI are really fascinating fields."
        
        # Default responses
        import random
        default_responses = [
            "That's interesting! Tell me more about that.",
            "I see. What made you think about that?",
            "That's a good point. I'm still learning about many topics.",
            "Interesting perspective! I'd love to learn more about your thoughts.",
            "I'm processing what you said. My neural network is still growing!"
        ]
        
        return random.choice(default_responses)
    
    def start_conversation(self):
        """Start interactive conversation"""
        while True:
            try:
                user_input = input("\nHuman: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("AI: Goodbye! Thanks for chatting with me. Have a great day! 🤖")
                    break
                
                # Generate response
                response = self.generate_response(user_input)
                print(f"AI: {response}")
                
                # Show tokenization info for longer inputs
                if len(user_input) > 10:
                    token_count = len(user_input.split()) + 2  # Simulate tokenization
                    print(f"    (Input tokenized to {token_count} tokens)")
                    
            except KeyboardInterrupt:
                print("\nAI: Goodbye! Thanks for chatting with me. Have a great day! 🤖")
                break
            except EOFError:
                print("\nAI: Goodbye! Thanks for chatting with me. Have a great day! 🤖")
                break

def demo_conversation():
    """Run a demo conversation with predefined inputs"""
    ai = DemoConversationalAI()
    
    # Predefined conversation
    demo_inputs = [
        "hello",
        "what is ai",
        "that's really cool!",
        "tell me about cuda",
        "how are you doing today?",
        "can you tell me a joke?",
        "thank you for the chat",
        "quit"
    ]
    
    print("\n" + "="*50)
    print("DEMO CONVERSATION:")
    print("="*50)
    
    for user_input in demo_inputs:
        print(f"\nHuman: {user_input}")
        
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("AI: Goodbye! Thanks for chatting with me. Have a great day! 🤖")
            break
            
        response = ai.generate_response(user_input)
        print(f"AI: {response}")
        
        if len(user_input) > 10:
            token_count = len(user_input.split()) + 2
            print(f"    (Input tokenized to {token_count} tokens)")

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Demo conversation (automated)")
    print("2. Interactive conversation")
    
    try:
        choice = input("Enter 1 or 2: ").strip()
        
        if choice == "1":
            demo_conversation()
        elif choice == "2":
            ai = DemoConversationalAI()
            ai.start_conversation()
        else:
            print("Invalid choice. Running demo...")
            demo_conversation()
            
    except KeyboardInterrupt:
        print("\nGoodbye!")