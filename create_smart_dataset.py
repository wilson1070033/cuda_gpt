#!/usr/bin/env python3
"""
Script to create an intelligent training dataset by combining all available data
"""

import os

def create_smart_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    # Input files
    files_to_combine = [
        'smart_train.txt',      # New intelligent conversations
        'new_train.txt',        # Math, code, daily, CUDA topics
        'train.txt',            # Original conversations
    ]
    
    output_file = os.path.join(data_dir, 'intelligent_train.txt')
    
    print("Creating intelligent training dataset...")
    
    total_size = 0
    conversation_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in files_to_combine:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                print(f"Adding content from {filename}")
                with open(filepath, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    if content:
                        # Count conversations (Human: lines)
                        conversation_count += content.count('Human:')
                        
                        outfile.write(content)
                        if not content.endswith('\n'):
                            outfile.write('\n')
                        outfile.write('\n')  # Extra separation
                        
                        file_size = os.path.getsize(filepath)
                        total_size += file_size
                        print(f"  - Added {file_size:,} bytes, {content.count('Human:')} conversations")
            else:
                print(f"Warning: {filename} not found, skipping")
    
    final_size = os.path.getsize(output_file)
    print(f"\nIntelligent training dataset created: {output_file}")
    print(f"Total size: {final_size:,} bytes")
    print(f"Total conversations: {conversation_count}")
    
    # Calculate vocabulary richness
    with open(output_file, 'r', encoding='utf-8') as f:
        text = f.read().lower()
        words = set(text.split())
        print(f"Unique words: {len(words):,}")
        print(f"Total words: {len(text.split()):,}")
        print(f"Vocabulary richness: {len(words)/len(text.split()):.3f}")

if __name__ == "__main__":
    create_smart_dataset()