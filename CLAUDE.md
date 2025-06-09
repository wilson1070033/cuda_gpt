# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System

This project uses CMake with CUDA support. Two executables are built:

### Primary Build Commands
```bash
# Standard build process
mkdir build && cd build  # or cd cmake-build-debug
cmake ..
make -j4

# Direct execution
./cuda_gpt      # Main training and demo program
./cuda_chat     # Interactive conversation mode
```

### Manual Compilation (fallback)
```bash
nvcc -I./include -std=c++17 -O3 -lcublas -lcurand main.cu src/*.cu -o cuda_gpt
```

### Training Data Setup
```bash
# Generate comprehensive intelligent training data
python3 generate_training_data.py    # Creates original data/train.txt
python3 create_smart_dataset.py      # Creates intelligent_train.txt with 326+ conversations
# Creates data/intelligent_train.txt with comprehensive conversations including:
# - Advanced science and mathematics (quantum computing, relativity, DNA)
# - Programming and technology (AI/ML, blockchain, neural networks)
# - Philosophy and psychology (consciousness, creativity, leadership)
# - Daily conversation patterns and problem-solving
# - CUDA and GPU computing topics
```

## Architecture Overview

### Core Design Principles
- **GPU-First Architecture**: All tensor operations run on CUDA
- **Educational Implementation**: Code prioritizes clarity over maximum optimization
- **Memory-Constrained Design**: Optimized for RTX 3050 Mobile (4GB VRAM)
- **Modular Components**: Clean separation between tensor ops, attention, transformer, and training

### Key Architectural Components

**Tensor System** (`tensor.h/cu`):
- Automatic host/device memory management with `to_host()` and `to_device()`
- cuBLAS integration for matrix operations
- Shape-aware tensor operations with bounds checking

**Attention Mechanism** (`attention.h/cu`):
- Scaled dot-product attention with causal masking
- Multi-head parallel computation
- Attention weights computed as Q·K^T/√d_k

**Transformer Architecture** (`transformer.h/cu`):
- Pre-norm layer configuration (LayerNorm before attention/FFN)
- Residual connections with proper gradient flow
- GELU activation in feed-forward networks
- Sinusoidal positional encodings

**Training System** (`training.h/cu`):
- cuDNN-accelerated training for optimized performance
- Adam optimizer with bias correction and learning rate scheduling
- Cross-entropy loss with numerical stability (log-softmax)
- Enhanced tokenizer with expanded vocabulary mapping
- DataLoader supports multiple training data formats with auto-detection

### Memory Management Strategy
- **Lazy transfers**: Data stays on GPU until explicitly moved to host
- **Shape consistency**: All operations verify tensor dimensions
- **cuBLAS handles**: Shared across components to avoid repeated initialization/destruction
- **Error propagation**: CUDA errors properly checked and reported

## Model Configuration Patterns

The codebase uses a configuration-driven approach for different GPU memory constraints:

```cpp
// RTX 3050 Mobile (4GB) - Enhanced Smart Model
ModelConfig config;
config.vocab_size = 2000;     // Large vocabulary for intelligent conversations
config.d_model = 512;         // Large model dimension for better understanding
config.n_heads = 8;           // Multi-head attention for pattern recognition  
config.n_layers = 6;          // Deep layers for complex reasoning
config.d_ff = 2048;           // Large feed-forward for capacity
config.dropout = 0.1f;        // Regularization

// RTX 3060+ (8GB+) - Maximum Intelligence
config.vocab_size = 5000;
config.d_model = 1024;
config.n_heads = 16;
config.n_layers = 12;
config.d_ff = 4096;
```

## Data Flow Architecture

**Training Pipeline**:
1. `DataLoader` reads from `data/combined_train.txt` (expanded dataset) or fallbacks
2. `Tokenizer` converts text to integer sequences with enhanced vocabulary
3. `GPTModel.forward()` processes batches through transformer layers with cuDNN optimization
4. `Trainer.compute_loss()` calculates cross-entropy loss with improved numerical stability
5. `AdamOptimizer` updates parameters with dynamic learning rate scheduling

**Conversation Pipeline** (chat.cu):
1. Pattern matching on input text (since model is untrained)
2. Response generation based on keyword detection
3. Token count display for educational purposes

## Important Implementation Details

### CUDA Kernel Organization
- **Layer normalization**: Custom kernel with shared memory for mean/variance
- **Attention masking**: Parallel causal mask application  
- **Loss computation**: Separate kernels for forward and gradient computation
- **Optimizer updates**: Vectorized Adam update kernels
- **cuDNN Integration**: Optimized deep learning primitives for improved performance

### cuDNN Acceleration Features
- **Automatic Detection**: CMake automatically detects and links cuDNN when available
- **Optimized Operations**: Enhanced activation functions, dropout, and convolution operations
- **Memory Management**: Efficient tensor descriptor and workspace management
- **Performance Monitoring**: Built-in timing and performance metrics during training

### Numerical Stability Measures
- Softmax uses max subtraction for overflow prevention
- Layer norm includes epsilon (1e-5) for division stability
- Loss computation handles padding tokens (negative labels)

### Error Handling Strategy
- CUDA errors checked with `cudaGetErrorString()`
- Tensor size mismatches caught with descriptive messages
- File I/O errors gracefully fallback to dummy data
- Model initialization errors prevent undefined behavior

## Conversation Training Data Structure

The training data follows a simple format in `data/train.txt`:
```
Human: [question/statement]
AI: [response]