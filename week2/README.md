# Week 2: Core Optimization Techniques

## 📋 Week Overview
This week focuses on making transformer inference faster through core optimization techniques. We'll implement KV caching, explore attention optimizations, and learn about kernel fusion and batch size optimization.

## 🎯 Learning Objectives
By the end of this week, you will be able to:
1. Implement and optimize KV caching with different strategies
2. Apply memory-efficient attention variants
3. Profile and optimize kernel operations
4. Determine optimal batch sizes for different hardware

## 📚 Core Concepts

### 1. KV Caching Deep Dive
- **Standard KV caching**: Cache key-value pairs across tokens
- **PagedAttention**: vLLM's memory-efficient caching
- **Rotary Position Embedding caching**: Pre-compute rotary embeddings
- **Cache eviction strategies**: LRU, FIFO, adaptive
- **Multi-query & grouped-query attention**: Reduced cache size

### 2. Attention Optimization
- **FlashAttention**: IO-aware exact attention
- **Memory-efficient attention**: Approximate and sparse variants
- **Sliding window attention**: Fixed context window
- **Linear attention**: O(n) complexity alternatives
- **Kernel fusion**: Combining attention operations

### 3. Batch Size Optimization
- **Memory vs throughput trade-off**: Finding the sweet spot
- **Dynamic batching**: Variable sequence lengths
- **Continuous batching**: vLLM's approach
- **Micro-batching**: Gradient accumulation for training

### 4. Operator Fusion & Kernel Optimization
- **GEMM optimization**: Matrix multiplication tuning
- **Activation fusion**: Combining linear + activation
- **LayerNorm optimization**: Fused normalization
- **Custom CUDA kernels**: When and how to write them

## 📖 Required Reading

### Papers
1. **"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"** (Dao et al., 2022)
   - Core FlashAttention algorithm
   - IO complexity analysis

2. **"Self-Attention Does Not Need O(n²) Memory"** (Rabe & Staats, 2021)
   - Memory-efficient attention formulation
   - Trade-offs and implementations

3. **"vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention"** (Kwon et al., 2023)
   - PagedAttention algorithm
   - Memory management for serving

### Blog Posts & Tutorials
1. [FlashAttention Explained](https://arxiv.org/abs/2205.14135)
2. [KV Caching in Transformers](https://kipp.ly/blog/kv-cache/)
3. [Optimizing Transformer Inference](https://huggingface.co/blog/optimize-llm)

## 💻 Hands-on Labs

### Lab 2.1: KV Caching Implementation
```python
# Implement different KV caching strategies
# Compare memory usage and performance
```

**Objectives:**
- Implement standard KV caching
- Add PagedAttention-style caching
- Profile memory usage across strategies
- Implement cache eviction policies

### Lab 2.2: FlashAttention Implementation
```python
# Implement FlashAttention from scratch
# Compare with standard attention
```

**Objectives:**
- Understand tiling and IO-aware computation
- Implement forward and backward passes
- Benchmark against PyTorch attention
- Profile memory bandwidth usage

### Lab 2.3: Batch Size Optimization
```python
# Find optimal batch sizes for different hardware
# Implement dynamic batching
```

**Objectives:**
- Profile throughput vs batch size
- Implement dynamic batching scheduler
- Test on different GPU architectures
- Create optimization guidelines

## 🧮 Mathematical Foundations

### KV Cache Memory Calculation
For a model with:
- Layers: L
- Hidden size: d
- Heads: h
- Head dimension: d_h = d/h
- Sequence length: n
- Batch size: b
- Data type: bytes_per_param

**KV Cache per layer:**
- Keys: b × n × h × d_h × bytes_per_param
- Values: b × n × h × d_h × bytes_per_param
- Total per layer: 2 × b × n × h × d_h × bytes_per_param

**Total cache:** L × 2 × b × n × h × d_h × bytes_per_param

### FlashAttention Complexity
**Standard attention:**
- FLOPs: O(n²d)
- Memory: O(n² + nd)

**FlashAttention:**
- FLOPs: O(n²d) (same)
- HBM accesses: O(nd²) vs O(n²d²)
- Speedup: Up to 7.6× on GPT-2

### Batch Size Optimization Formula
**Throughput model:**
T(b) = (latency_constant + b × latency_per_example)⁻¹ × b

**Optimal batch size:**
b_opt = √(latency_constant / latency_per_example)

## 🔬 Advanced Topics

### 1. Multi-Query Attention (MQA)
- Single key-value head shared across query heads
- Reduces KV cache by factor of h
- Quality trade-offs and when to use

### 2. Grouped-Query Attention (GQA)
- Intermediate between MHA and MQA
- Groups of query heads share key-value heads
- Better quality than MQA with reduced cache

### 3. Sliding Window Attention
- Fixed context window that slides
- O(n × w) complexity for window size w
- Suitable for long sequences

### 4. Linear Attention Variants
- Performer, Linformer, Linear Transformer
- O(n) or O(n log n) complexity
- Approximation quality analysis

## 📊 Performance Benchmarks

### KV Cache Strategies
We'll benchmark:
1. **Standard caching**: Baseline
2. **PagedAttention**: vLLM implementation
3. **MQA/GQA**: Reduced cache variants
4. **Dynamic caching**: Adaptive strategies

**Metrics:**
- Memory usage vs sequence length
- Throughput at different batch sizes
- Cache hit/miss rates
- End-to-end latency

### Attention Optimization Comparison
Compare:
1. **PyTorch attention**: Baseline
2. **FlashAttention**: IO-optimized
3. **Memory-efficient**: Approximate variants
4. **Custom kernels**: Hand-tuned implementations

## 🚀 Production Considerations

### 1. Mixed Precision Inference
- FP16, BF16, FP8 precision levels
- Accuracy vs speed trade-offs
- Hardware support requirements

### 2. Kernel Auto-tuning
- Automatic kernel selection
- Hardware-specific optimizations
- Runtime adaptation

### 3. Profiling Tools
- NVIDIA Nsight Systems
- PyTorch Profiler
- Custom profiling instrumentation

### 4. Deployment Optimization
- Model compilation (TorchScript, ONNX)
- TensorRT optimization
- Server configuration tuning

## 📝 Weekly Deliverables

### 1. Code Submission
- Complete all three labs
- Include performance benchmarks
- Add unit tests for optimization functions

### 2. Optimization Report
- Analysis of KV caching strategies
- FlashAttention implementation notes
- Batch size optimization findings
- Recommendations for different scenarios

### 3. Research Implementation
- Implement one advanced attention variant
- Compare with standard attention
- Document trade-offs and use cases

## 🔧 Setup Instructions

### Additional Dependencies
```bash
# Install FlashAttention
pip install flash-attn --no-build-isolation

# Install Triton for custom kernels
pip install triton

# Install benchmarking tools
pip install pyinstrument line_profiler

# Install vLLM for comparison
pip install vllm
```

### Hardware Requirements
This week requires GPU acceleration:
- **Minimum**: NVIDIA GPU with 8GB VRAM
- **Recommended**: NVIDIA GPU with 16GB+ VRAM
- **Cloud option**: Google Colab Pro+ or AWS p3.2xlarge

### Verification Script
```python
import torch
import flash_attn

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"FlashAttention available: {hasattr(flash_attn, 'flash_attn_qkvpacked_func')}")

# Test basic operations
x = torch.randn(2, 32, 512, device='cuda')
y = torch.randn(2, 32, 512, device='cuda')
z = torch.matmul(x, y.transpose(-1, -2))
print(f"Matrix multiplication test passed: {z.shape == (2, 32, 32)}")
```

## 🎯 Success Criteria

You've successfully completed Week 2 if you can:
1. Implement and optimize KV caching for different scenarios
2. Explain the trade-offs of different attention optimizations
3. Determine optimal batch sizes for given hardware constraints
4. Profile and identify bottlenecks in transformer inference

## 📚 Additional Resources

### Videos
1. [FlashAttention: Fast and Memory-Efficient Exact Attention](https://www.youtube.com/watch?v=GQjX5F2fNpQ)
2. [Optimizing Transformer Inference on GPUs](https://www.youtube.com/watch?v=MV2c6t6HZ8M)
3. [vLLM: Efficient LLM Serving](https://www.youtube.com/watch?v=4CxXozqZQqo)

### Tools
1. [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention)
2. [vLLM GitHub](https://github.com/vllm-project/vllm)
3. [Triton Language Tutorial](https://triton-lang.org/main/getting-started/tutorials/index.html)

### Community
1. [PyTorch Optimization Forum](https://discuss.pytorch.org/c/optimization/11)
2. [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
3. [MLSys Papers](https://mlsys.org/)

---

**Estimated Time Commitment:** 15-18 hours  
**Difficulty Level:** ⭐⭐⭐☆☆ (Intermediate optimization)  
**Next Week:** Advanced Sampling & Decoding