#!/usr/bin/env python3
"""
Generate conversational training data for CUDA GPT model
Creates realistic dialogue patterns for training
"""

import random

# Conversation templates
greetings = [
    ("Hello, how are you today?", "I'm doing well, thank you for asking! How can I help you today?"),
    ("Hi there!", "Hello! It's nice to meet you. What would you like to talk about?"),
    ("Good morning!", "Good morning! I hope you're having a wonderful day. How can I assist you?"),
    ("Hey, what's up?", "Hello! Not much, just here to help. What's on your mind?"),
    ("How's it going?", "Things are going great! Thanks for asking. What can I do for you?")
]

questions_answers = [
    ("What is AI?", "AI stands for Artificial Intelligence. It's technology that enables machines to simulate human intelligence and perform tasks like learning, reasoning, and problem-solving."),
    ("How does machine learning work?", "Machine learning uses algorithms to analyze data, identify patterns, and make predictions or decisions without being explicitly programmed for each specific task."),
    ("What is CUDA?", "CUDA is a parallel computing platform developed by NVIDIA that allows developers to use GPU acceleration for general-purpose computing tasks."),
    ("What's the weather like?", "I don't have access to real-time weather data, but I'd be happy to help you with other questions or tasks!"),
    ("Can you help me with programming?", "Absolutely! I'd be glad to help with programming questions. What language or topic would you like assistance with?"),
    ("What's your favorite color?", "I don't have personal preferences like favorite colors, but I find the concept of color perception fascinating! What's your favorite color?"),
    ("How old are you?", "I don't have an age in the traditional sense since I'm an AI, but I'm always learning and growing with each conversation!"),
    ("Do you like music?", "I don't experience music the way humans do, but I find the mathematical patterns and emotional expressions in music incredibly interesting!"),
    ("What can you do?", "I can help with various tasks like answering questions, explaining concepts, helping with programming, having conversations, and much more! What would you like help with?")
]

casual_conversations = [
    ("Tell me a joke", "Why don't scientists trust atoms? Because they make up everything! I hope that brought a smile to your face."),
    ("I'm feeling sad today", "I'm sorry to hear you're feeling sad. Sometimes talking about what's bothering you can help. Would you like to share what's on your mind?"),
    ("That's really cool!", "I'm glad you think so! It's always great to share excitement about interesting things. What specifically caught your attention?"),
    ("I don't understand", "No problem at all! Everyone learns at their own pace. Could you tell me which part you'd like me to explain differently?"),
    ("Thank you so much!", "You're very welcome! I'm happy I could help. Feel free to ask if you need anything else."),
    ("This is confusing", "I understand that some topics can be confusing. Let me try to break it down into simpler parts. What specific aspect would you like me to clarify?"),
    ("You're really helpful", "Thank you for the kind words! It makes me happy to know I'm being helpful. Is there anything else you'd like to explore?")
]

technical_questions = [
    ("How do neural networks work?", "Neural networks are inspired by the human brain. They consist of interconnected nodes (neurons) that process information through weighted connections, learning patterns from data through training."),
    ("What is deep learning?", "Deep learning uses neural networks with multiple layers to automatically learn complex patterns in data. The 'deep' refers to the many layers that allow the network to learn hierarchical representations."),
    ("Explain transformers in AI", "Transformers are a type of neural network architecture that uses self-attention mechanisms to process sequences of data, like text. They're the foundation of modern language models like GPT."),
    ("What is GPU computing?", "GPU computing uses graphics processing units to perform parallel calculations. GPUs excel at handling many simple operations simultaneously, making them ideal for machine learning and scientific computing."),
    ("How does training work?", "Training involves showing the AI many examples so it can learn patterns. The model adjusts its internal parameters based on mistakes, gradually improving its performance on the task.")
]

def generate_conversation_file():
    """Generate comprehensive training data file"""
    
    conversations = []
    
    # Add multiple instances of each conversation type
    for _ in range(10):  # Repeat conversations for better learning
        conversations.extend(greetings)
        conversations.extend(questions_answers)
        conversations.extend(casual_conversations)
        conversations.extend(technical_questions)
    
    # Shuffle for better training
    random.shuffle(conversations)
    
    # Write to file
    with open('data/train.txt', 'w', encoding='utf-8') as f:
        for human_msg, ai_msg in conversations:
            f.write(f"Human: {human_msg}\n")
            f.write(f"AI: {ai_msg}\n")
            f.write("\n")  # Empty line between conversations
    
    print(f"Generated {len(conversations)} conversation pairs")
    print("Training data saved to data/train.txt")

if __name__ == "__main__":
    generate_conversation_file()