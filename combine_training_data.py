#!/usr/bin/env python3
"""
Script to combine the original train.txt with new_train.txt
"""

import os

def combine_training_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    old_file = os.path.join(data_dir, 'train.txt')
    new_file = os.path.join(data_dir, 'new_train.txt')
    combined_file = os.path.join(data_dir, 'combined_train.txt')
    
    print("Combining training data files...")
    
    with open(combined_file, 'w', encoding='utf-8') as outfile:
        # First add the new expanded data
        if os.path.exists(new_file):
            print(f"Adding content from {new_file}")
            with open(new_file, 'r', encoding='utf-8') as infile:
                content = infile.read()
                outfile.write(content)
                if not content.endswith('\n'):
                    outfile.write('\n')
                outfile.write('\n')  # Extra newline for separation
        
        # Then add the original data
        if os.path.exists(old_file):
            print(f"Adding content from {old_file}")
            with open(old_file, 'r', encoding='utf-8') as infile:
                content = infile.read()
                outfile.write(content)
    
    print(f"Combined training data saved to {combined_file}")
    
    # Get file sizes for comparison
    if os.path.exists(old_file):
        old_size = os.path.getsize(old_file)
        print(f"Original train.txt size: {old_size} bytes")
    
    if os.path.exists(new_file):
        new_size = os.path.getsize(new_file)
        print(f"New train.txt size: {new_size} bytes")
    
    combined_size = os.path.getsize(combined_file)
    print(f"Combined file size: {combined_size} bytes")

if __name__ == "__main__":
    combine_training_data()