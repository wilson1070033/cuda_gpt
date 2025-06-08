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
# Generate conversation training data
python3 generate_training_data.py
# Creates data/train.txt with 260+ conversation pairs
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
- Adam optimizer with bias correction
- Cross-entropy loss with numerical stability (log-softmax)
- Basic tokenizer with vocabulary mapping
- DataLoader supports both file input and dummy data generation

### Memory Management Strategy
- **Lazy transfers**: Data stays on GPU until explicitly moved to host
- **Shape consistency**: All operations verify tensor dimensions
- **cuBLAS handles**: Shared across components to avoid repeated initialization/destruction
- **Error propagation**: CUDA errors properly checked and reported

## Model Configuration Patterns

The codebase uses a configuration-driven approach for different GPU memory constraints:

```cpp
// RTX 3050 Mobile (4GB) - Current default
ModelConfig config;
config.vocab_size = 200;
config.d_model = 128;
config.n_heads = 4;
config.n_layers = 2;

// RTX 3060+ (6GB+) - Scaling up
config.vocab_size = 1000;
config.d_model = 256;
config.n_heads = 8;
config.n_layers = 4;
```

## Data Flow Architecture

**Training Pipeline**:
1. `DataLoader` reads from `data/train.txt` or generates dummy data
2. `Tokenizer` converts text to integer sequences
3. `GPTModel.forward()` processes batches through transformer layers
4. `Trainer.compute_loss()` calculates cross-entropy loss
5. `AdamOptimizer` updates parameters

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