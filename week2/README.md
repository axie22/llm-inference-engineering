# Week 2: Core Optimization Techniques

## 📋 Week Overview
This week focuses on making transformer inference faster through core optimization techniques. We'll implement KV caching, explore attention optimizations, and learn about kernel fusion and batch size optimization. Updated with modern advances (2024-2025) including FlashAttention-2, PagedAttention v2, and efficient attention variants.

## 🎯 Learning Objectives
By the end of this week, you will be able to:
1. Implement and optimize KV caching with different strategies (standard, paged, sliding window)
2. Apply memory-efficient attention variants (FlashAttention-2, memory-efficient, sliding window)
3. Profile and optimize kernel operations using CUDA and Triton
4. Determine optimal batch sizes for different hardware and workloads
5. Understand recent advances in attention optimization (MQA, GQA, Mamba)

## 📚 Core Concepts

### 1. KV Caching Deep Dive
- **Standard KV caching**: Cache key-value pairs across tokens
- **PagedAttention**: vLLM's memory-efficient caching (v2 updates)
- **Rotary Position Embedding caching**: Pre-compute rotary embeddings
- **Cache eviction strategies**: LRU, FIFO, adaptive
- **Multi-query & grouped-query attention**: Reduced cache size
- **Sliding window caching**: Limit cache size to recent tokens
- **Dynamic cache allocation**: Adapt cache size per request

### 2. Attention Optimization
- **FlashAttention-2**: Faster and more parallel version (2023)
- **Memory-efficient attention**: Approximate and sparse variants
- **Sliding window attention**: Fixed context window (Longformer, StreamingLLM)
- **Linear attention**: O(n) complexity alternatives (Performer, Linformer)
- **Kernel fusion**: Combining attention operations (Triton)
- **Grouped-query attention (GQA)**: Balancing quality and memory
- **Multi-query attention (MQA)**: Extreme memory reduction

### 3. Batch Size Optimization
- **Memory vs throughput trade-off**: Finding the sweet spot
- **Dynamic batching**: Variable sequence lengths (ORCA, vLLM)
- **Continuous batching**: vLLM's approach with iteration-level scheduling
- **Micro-batching**: Gradient accumulation for training
- **Adaptive batching**: Based on request latency SLOs

### 4. Operator Fusion & Kernel Optimization
- **GEMM optimization**: Matrix multiplication tuning (cuBLAS, CUTLASS)
- **Activation fusion**: Combining linear + activation (SiLU, GeLU)
- **LayerNorm optimization**: Fused normalization (Apex, xFormers)
- **Custom CUDA kernels**: When and how to write them
- **Triton kernels**: High-level GPU programming for ML

## 📖 Required Reading

### Foundational Papers
1. **"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"** (Dao et al., 2022)
   - Core FlashAttention algorithm
   - IO complexity analysis

2. **"FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"** (Dao et al., 2023)
   - Improvements over FlashAttention
   - Better utilization of GPU warps

3. **"vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention"** (Kwon et al., 2023)
   - PagedAttention algorithm
   - Memory management for serving

4. **"Self-Attention Does Not Need O(n²) Memory"** (Rabe & Staats, 2021)
   - Memory-efficient attention formulation
   - Trade-offs and implementations

### Modern Reading (2024-2025)
1. **"PagedAttention v2: Faster and More Memory-Efficient"** (vLLM team, 2024)
   - Improvements to page table management
   - Support for variable block sizes

2. **"MQA and GQA: Efficient Attention for Large Language Models"** (Google, 2024)
   - Comparative analysis of multi-query and grouped-query attention
   - Quality vs memory trade-offs

3. **"StreamingLLM: Efficient Streaming Language Models with Infinite Context"** (Xiao et al., 2024)
   - Sliding window attention with attention sink
   - Practical deployment for streaming applications

4. **"Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"** (Tillet et al., 2023)
   - Writing efficient GPU kernels without CUDA

### Blog Posts & Tutorials
1. [FlashAttention-2 Explained](https://hazyresearch.stanford.edu/blog/2023-07-12-flashattention-2)
2. [KV Caching in Transformers](https://kipp.ly/blog/kv-cache/)
3. [Optimizing Transformer Inference](https://huggingface.co/blog/optimize-llm)
4. [vLLM Internals](https://blog.vllm.ai/2023/06/20/vllm-internals.html)
5. [Grouped-Query Attention](https://www.answer.ai/posts/2024-03-04-grouped-query-attention.html)

## 🖼️ Visual Aids

### KV Caching Diagram
```
Standard KV Cache:
┌─────────────────────────────────────────────┐
│ Layer 1: [K1 V1][K2 V2]...[Kn Vn]           │
│ Layer 2: [K1 V1][K2 V2]...[Kn Vn]           │
│ ...                                          │
│ Layer L: [K1 V1][K2 V2]...[Kn Vn]           │
└─────────────────────────────────────────────┘
Memory: O(L * n * h * d_h)

PagedAttention:
┌─────────────────────────────────────────────┐
│ Pages: [K1‑16 V1‑16] [K17‑32 V17‑32] ...    │
│ Page Table: Batch→[Page IDs]                │
│ Free List: Unused pages                      │
└─────────────────────────────────────────────┘
Memory: O(active tokens * h * d_h)
```

### FlashAttention Tiling
```
HBM (High Bandwidth Memory)
   ↓
Tile Q (SRAM) → Compute attention with tile K, V
   ↓
Write back to HBM with rescaling
```

### Attention Variants Comparison
| Variant           | Memory | Time  | Quality | Use Case          |
|-------------------|--------|-------|---------|-------------------|
| Full Attention    | O(n²)  | O(n²) | Best    | Short sequences   |
| FlashAttention-2  | O(n)   | O(n²) | Exact   | General           |
| Sliding Window    | O(w*n) | O(w*n)| Good    | Streaming         |
| Linear Attention  | O(n)   | O(n)  | Approx  | Very long seq     |
| MQA               | O(n/h) | O(n²) | Reduced | Memory-bound      |
| GQA               | O(n/g) | O(n²) | Better  | Balanced          |

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

**Modern Extensions:**
- Integrate with Rotary Position Embeddings
- Add sliding window cache limits
- Benchmark against vLLM's implementation

### Lab 2.2: FlashAttention Implementation
```python
# Implement FlashAttention-2 from scratch
# Compare with standard attention
```

**Objectives:**
- Understand tiling and IO-aware computation
- Implement forward and backward passes
- Benchmark against PyTorch attention
- Profile memory bandwidth usage

**Modern Extensions:**
- Add support for GQA/MQA
- Implement kernel fusion with Triton
- Compare with xFormers library

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

**Modern Extensions:**
- Implement continuous batching (iteration-level)
- Add latency SLO awareness
- Integrate with vLLM's scheduler

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

**With GQA (groups g):**
- Keys/Values per group: b × n × g × d_h × bytes_per_param
- Total cache: L × 2 × b × n × g × d_h × bytes_per_param

### FlashAttention-2 Complexity
**Standard attention:**
- FLOPs: O(n²d)
- Memory: O(n² + nd)
- HBM accesses: O(n²d²)

**FlashAttention-2:**
- FLOPs: O(n²d) (same)
- HBM accesses: O(nd²)
- Speedup: Up to 2× over FlashAttention, 9× over baseline

### Batch Size Optimization Formula
**Throughput model:**
T(b) = (latency_constant + b × latency_per_example)⁻¹ × b

**Optimal batch size:**
b_opt = √(latency_constant / latency_per_example)

**With continuous batching:**
T(b) = (max(latency_i))⁻¹ × Σb_i

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
- Suitable for long sequences (StreamingLLM)

### 4. Linear Attention Variants
- Performer, Linformer, Linear Transformer
- O(n) or O(n log n) complexity
- Approximation quality analysis

### 5. Mamba: Selective State Spaces
- Linear-time sequence modeling
- Hardware-aware algorithm
- Comparison to attention-based models

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
2. **FlashAttention-2**: IO-optimized
3. **Memory-efficient**: Approximate variants
4. **Custom kernels**: Hand-tuned implementations

### Hardware-Specific Optimization
- **NVIDIA H100**: Tensor Core utilization
- **AMD MI300X**: Matrix Core optimizations
- **AWS Inferentia2**: Neuron SDK
- **Google TPUv5**: XLA compilation

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
- TensorRT-LLM optimization
- Server configuration tuning

### 5. Serving Frameworks
- vLLM (PagedAttention)
- TensorRT-LLM (Kernel fusion)
- TGI (Hugging Face)
- Ray Serve (Distributed)

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
# Install FlashAttention-2
pip install flash-attn --no-build-isolation

# Install Triton for custom kernels
pip install triton

# Install benchmarking tools
pip install pyinstrument line_profiler

# Install vLLM for comparison
pip install vllm

# Install xFormers for attention variants
pip install xformers

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Hardware Requirements
This week requires GPU acceleration:
- **Minimum**: NVIDIA GPU with 8GB VRAM (RTX 3070+)
- **Recommended**: NVIDIA GPU with 16GB+ VRAM (RTX 4090, A100)
- **Cloud option**: Google Colab Pro+, AWS p3.2xlarge, Lambda Labs

### Verification Script
```python
import torch
import flash_attn

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"FlashAttention-2 available: {hasattr(flash_attn, 'flash_attn_varlen_func')}")

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
5. Apply modern optimization techniques (FlashAttention-2, GQA, PagedAttention)

## 📚 Additional Resources

### Videos
1. [FlashAttention-2: Faster Attention with Better Parallelism](https://www.youtube.com/watch?v=H6QcPkGc8bQ)
2. [Optimizing Transformer Inference on GPUs](https://www.youtube.com/watch?v=MV2c6t6HZ8M)
3. [vLLM: Efficient LLM Serving](https://www.youtube.com/watch?v=4CxXozqZQqo)
4. [Grouped-Query Attention Explained](https://www.youtube.com/watch?v=8c8zJ3nN0x4)

### Tools
1. [FlashAttention-2 GitHub](https://github.com/Dao-AILab/flash-attention)
2. [vLLM GitHub](https://github.com/vllm-project/vllm)
3. [Triton Language Tutorial](https://triton-lang.org/main/getting-started/tutorials/index.html)
4. [xFormers GitHub](https://github.com/facebookresearch/xformers)

### Community
1. [PyTorch Optimization Forum](https://discuss.pytorch.org/c/optimization/11)
2. [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
3. [MLSys Papers](https://mlsys.org/)
4. [Hugging Face Forums](https://discuss.huggingface.co/)

---

**Estimated Time Commitment:** 15-20 hours  
**Difficulty Level:** ⭐⭐⭐☆☆ (Intermediate optimization)  
**Next Week:** Advanced Sampling & Decoding

---
*Last Updated: April 2026*  
*Enhanced with modern reading material and graphics*