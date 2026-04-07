# Week 1: Foundations of LLM Inference
## From First Principles to Production-Ready Understanding

## 📋 Week Overview
This week establishes the **foundational knowledge** needed to understand, analyze, and optimize LLM inference. We'll move from first principles to practical implementation, building an intuition for the computational and memory characteristics that govern transformer performance. Unlike high-level overviews, we dive deep into the **actual operations**, **memory access patterns**, and **hardware constraints** that shape real-world inference systems.

### 🎯 What Makes This Week Different?
- **From scratch implementations** that reveal underlying complexity
- **Hardware-aware analysis** connecting algorithms to silicon
- **Production perspective** focusing on measurable metrics
- **Mathematical rigor** with practical interpretations

## 🎯 Learning Objectives
By the end of this week, you will be able to:

### Core Competencies
1. **Explain** the computational complexity of each transformer component with concrete FLOP counts
2. **Calculate** memory requirements for different model sizes and batch configurations
3. **Profile** inference performance using industry-standard metrics and tools
4. **Identify** the hardware constraints that fundamentally shape optimization strategies

### Practical Skills
5. **Implement** a transformer layer from scratch with correct tensor shapes
6. **Measure** and interpret PyTorch profiler output for bottleneck identification
7. **Optimize** batch size selection based on memory-throughput trade-offs
8. **Estimate** inference cost ($/million tokens) for different hardware configurations

## 📚 Core Concepts Deep Dive

### 1. Transformer Architecture: Beyond the Diagram
The transformer isn't just boxes and arrows—it's a specific sequence of tensor operations with precise memory access patterns.

#### Self-Attention: The Heart of the Matter
**Query-Key-Value mechanics:**
```python
# Not just matrix multiplication—understand the data movement
# Q, K, V projections: [batch, seq_len, d_model] → [batch, seq_len, d_model]
# Each is an independent linear layer, but they can be fused
Q = linear_q(x)  # Weight matrix shape: [d_model, d_model]
K = linear_k(x)  # Same weights shape, different values
V = linear_v(x)

# Attention scores: [batch, num_heads, seq_len, seq_len]
# This is where the n² complexity comes from
scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_head)

# Softmax: Expensive but parallelizable across positions
attn_weights = F.softmax(scores, dim=-1)

# Weighted sum: [batch, num_heads, seq_len, d_head]
context = torch.matmul(attn_weights, V)
```

**Key Insight:** The attention mechanism has O(n²d) computational complexity but also O(n²) memory complexity for the attention matrix. For n=4096 and d=4096, this is 67M FLOPs for QK^T but stores a 4096×4096 = 16.8M element matrix (67MB in FP32).

#### Multi-Head Attention: Parallelism vs Cohesion
- **Independent computation**: Each head processes different subspaces
- **Concatenation overhead**: Memory layout impacts performance
- **Head dimension trade-off**: Smaller d_head = more heads but less capacity per head

**Practical consideration:** Typical configurations:
- GPT-3: 96 heads, d_head=128 (d_model=12288)
- LLaMA 2 7B: 32 heads, d_head=128 (d_model=4096)
- Smaller models: 12-16 heads, d_head=64

#### Feed-Forward Networks: The Memory Bandwidth Challenge
The FFN is often the memory bandwidth bottleneck:
```python
# Two linear transformations with ReLU in between
# Typically: d_model → 4×d_model → d_model
hidden = self.fc1(x)    # [batch, seq_len, 4*d_model]
hidden = F.relu(hidden) # Elementwise, cheap
output = self.fc2(hidden) # [batch, seq_len, d_model]
```

**Memory analysis:** The intermediate activation of size [batch, seq_len, 4*d_model] can be huge. For batch=8, seq_len=2048, d_model=4096: 8×2048×16384 = 268M elements = 1GB in FP32!

#### Layer Normalization & Residual Connections
- **LayerNorm**: Statistics computation (mean, variance) + affine transform
- **Pre-norm vs Post-norm**: Affects gradient flow and training stability
- **Residual connections**: Enable deeper networks but require keeping input in memory

### 2. Computational Complexity: Beyond Big-O Notation
Big-O tells us scaling behavior, but constants matter for practical optimization.

#### Attention Complexity Breakdown
Given:
- Sequence length: n
- Hidden dimension: d  
- Number of heads: h
- Head dimension: d_h = d/h

**Exact FLOP counts (forward pass only):**
1. **Q/K/V projections**: 3 × n × d × d = 3nd² FLOPs
2. **QK^T multiplication**: n × n × d_h = n²d_h FLOPs  
3. **Softmax**: ~3n² FLOPs (exp, sum, division)
4. **Attention × V**: n × n × d_h = n²d_h FLOPs
5. **Output projection**: n × d_h × d = nd² FLOPs

**Total**: 4nd² + 2n²d_h + 3n² ≈ 4nd² + 2n²d FLOPs

**Example for GPT-3 (n=2048, d=12288, h=96):**
- Projections: 3×2048×12288² ≈ 926 GFLOPs
- Attention: 2×2048²×128 ≈ 1.07 GFLOPs  
- Ratio: Attention is only ~0.1% of total! The real cost is in the projections.

**Takeaway:** For long sequences, attention dominates. For typical lengths, linear layers dominate.

#### Feed-Forward Network Complexity
Standard FFN with expansion factor 4:
- First layer: n × d × 4d = 4nd² FLOPs
- Second layer: n × 4d × d = 4nd² FLOPs
- **Total**: 8nd² FLOPs

**Comparison:** One attention layer = 4nd² FLOPs, one FFN layer = 8nd² FLOPs. FFN is 2× more expensive!

#### Memory Hierarchy: The 1000× Speed Difference
Understanding memory is crucial for optimization:

| Memory Type | Size | Bandwidth | Latency | Use Case |
|-------------|------|-----------|---------|----------|
| GPU Registers | ~256KB | ~10 TB/s | 1 cycle | Tensor cores, thread-local |
| Shared Memory | 64-164KB | ~3 TB/s | ~20 cycles | Block cooperation, tiling |
| L1/L2 Cache | 192KB/48MB | ~2 TB/s | ~100 cycles | Reuse, spatial locality |
| HBM2/GDDR6 | 16-80GB | 0.8-2 TB/s | ~300 cycles | Main GPU memory |
| CPU RAM | 32-512GB | 50-200 GB/s | ~1000 cycles | Overflow, CPU tensors |
| NVMe SSD | 1-8TB | 3-7 GB/s | ~100μs | Model checkpointing |
| Network | - | 1-100 Gb/s | ~1ms | Distributed inference |

**Key insight:** A memory access to HBM is ~300× slower than a register access. Optimization = minimizing HBM accesses.

### 3. Inference Metrics: What Actually Matters
Different stakeholders care about different metrics:

#### Latency Metrics
- **Time to First Token (TTFT)**: Critical for interactive applications
  - Includes prompt processing, KV cache population
  - Typically 100ms-1s for reasonable models
- **Time Per Output Token (TPOT)**: Determines generation speed
  - Post-first token generation
  - Typically 10-100ms/token
- **End-to-End Latency**: TTFT + (output_tokens × TPOT)

**Industry benchmarks:**
- ChatGPT: ~300ms TTFT, ~50ms TPOT
- Local 7B model (RTX 4090): ~500ms TTFT, ~20ms TPOT
- Cloud API (GPT-4): ~700ms TTFT, ~80ms TPOT

#### Throughput Metrics
- **Tokens per Second**: Raw generation speed
- **Requests per Second**: Accounts for varying sequence lengths
- **Concurrent Users Supported**: Function of throughput and latency

**Throughput-Latency Trade-off:**
```python
# Little's Law: L = λ × W
# L = average number of requests in system
# λ = request arrival rate (RPS)
# W = average response time

# Example: If W = 2s and we want L ≤ 10 (manageable queue)
# Then λ ≤ 10 / 2 = 5 RPS maximum
```

#### Memory Metrics
- **Peak Memory**: Maximum memory used during inference
- **Persistent Memory**: Model weights + KV cache
- **Memory Bandwidth Utilization**: % of theoretical maximum

**Memory calculation example (GPT-3 175B):**
- Weights (FP16): 175B × 2 bytes = 350GB
- KV cache (n=2048, batch=1): 96 layers × 2 × 2048 × 128 × 2 bytes ≈ 100MB
- **Total:** ~351GB (doesn't fit on most GPUs!)

#### Cost Metrics
- **$/million tokens**: Standard pricing metric
- **Energy/token**: Important for sustainability
- **Hardware amortization**: Cost spread over lifetime

**Cloud pricing examples:**
- GPT-4: ~$30/million tokens input, $60/million tokens output
- Claude 3 Opus: ~$75/million tokens
- Self-hosted 7B model: ~$0.10/million tokens (electricity only)

## 📖 Required Reading with Guided Questions

### Papers

#### 1. **"Attention Is All You Need"** (Vaswani et al., 2017)
**Focus sections:** 3.1-3.3 (Architecture), 5 (Experiments)

**Key questions to answer while reading:**
1. Why did the authors choose dot-product attention over additive attention?
2. How does multi-head attention provide benefits over single-head?
3. What positional encoding schemes were considered and why was sinusoidal chosen?
4. How does the transformer's complexity compare to RNNs/CNNs for different sequence lengths?

**Practical insights:**
- The paper reports 3.5× faster training than LSTMs—think about inference implications
- Dropout rates (0.1) and label smoothing (0.1) affect activation distributions
- Beam search size 4 with length penalty α=0.6 was used—how does this affect inference?

#### 2. **"Efficient Transformers: A Survey"** (Tay et al., 2020)
**Focus sections:** 1-3 (Introduction, Taxonomy, Efficient Attention)

**Key questions:**
1. What are the three main approaches to transformer efficiency?
2. How do different sparse attention patterns trade off quality and speed?
3. What hardware considerations are mentioned for different optimizations?

**Taxonomy to understand:**
- Fixed patterns (Local, Strided, Global)
- Learnable patterns (Reformer, Sparse Transformer)
- Memory compression (Linear attention, Nyström approximation)

#### 3. **"The Hardware Lottery"** (Hooker, 2020)
**Focus:** Entire paper (it's short and impactful)

**Key questions:**
1. What historical examples show hardware influencing algorithm success?
2. How does this apply to transformer inference today?
3. What hardware developments might change optimization priorities?

**Relate to current context:**
- Tensor cores favor certain matrix sizes (multiples of 8 for FP16)
- HBM bandwidth limits attention-heavy models
- Specialized AI chips (TPU, Cerebras, Groq) change cost functions

### Blog Posts & Tutorials with Exercises

#### 1. [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
**Exercise:** Run the code and modify:
- Change number of heads from 8 to 4 and 16, measure speed difference
- Implement different positional encodings (learned, rotary)
- Add profiling to identify the slowest operation

#### 2. [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
**Exercise:** Create your own diagram for:
- Tensor shapes at each step for a specific configuration
- Memory allocation over time during inference
- Data flow between CPU and GPU

#### 3. [Transformer Inference Arithmetic](https://kipp.ly/blog/transformer-inference-arithmetic/)
**Exercise:** Extend the calculations:
- Add KV cache memory to the formulas
- Calculate for mixed precision (FP16 weights, FP32 accumulators)
- Model memory vs compute bound regimes on different GPUs

## 💻 Hands-on Labs with Extended Explanations

### Lab 1.1: Transformer Implementation from Scratch
**Not just implementation—understanding:**

#### Part A: Basic Implementation
```python
# We'll implement with extensive comments explaining WHY each operation exists
class EducationalMultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        # Why assert divisibility? Because we need equal head dimensions
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Why separate projections? Could fuse for efficiency
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model) 
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Why sqrt(d_head) scaling? Variance stabilization
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape for multi-head
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: [batch, heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Why softmax on last dimension? Attend over keys for each query position
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape back: [batch, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return context
```

#### Part B: Memory Analysis
```python
def analyze_memory_usage(model, batch_size=1, seq_len=512):
    """Analyze memory usage of attention mechanism"""
    
    # Calculate theoretical memory
    d_model = model.d_model
    num_heads = model.num_heads
    head_dim = model.head_dim
    
    print(f"Configuration: batch={batch_size}, seq_len={seq_len}, d_model={d_model}, heads={num_heads}")
    print(f"Head dimension: {head_dim}")
    
    # Q/K/V projections memory
    proj_memory = 3 * batch_size * seq_len * d_model * 2  # FP16
    
    # Attention matrix memory (the n² problem!)
    attn_matrix_memory = batch_size * num_heads * seq_len * seq_len * 2
    
    # Context memory
    context_memory = batch_size * seq_len * d_model * 2
    
    print(f"\nMemory breakdown (FP16):")
    print(f"  Q/K/V projections: {proj_memory / 1024**2:.2f} MB")
    print(f"  Attention matrix:   {attn_matrix_memory / 1024**2:.2f} MB")
    print(f"  Context output:     {context_memory / 1024**2:.2f} MB")
    print(f"  Total temporary:    {(proj_memory + attn_matrix_memory + context_memory) / 1024**2:.2f} MB")
    
    # Compare with typical GPU memory
    if torch.cuda.is_available():
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        print(f"\nGPU memory available: {free_mem / 1024**3:.2f} GB")
        utilization = (proj_memory + attn_matrix_memory + context_memory) / free_mem * 100
        print(f"  This operation uses: {utilization:.1f}% of available memory")
```

#### Part C: Performance Profiling
```python
def profile_attention_speed(model, input_sizes):
    """Profile attention at different sequence lengths"""
    
    results = []
    
    for seq_len in input_sizes:
        x = torch.randn(1, seq_len, model.d_model).cuda()
        
        # Warmup
        for _ in range(10):
            _ = model(x)
        
        # Profile
        torch.cuda.synchronize()
        start = time.time()
        
        iterations = 100
        for _ in range(iterations):
            _ = model(x)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        avg_ms = elapsed / iterations * 1000
        results.append((seq_len, avg_ms))
        
        print(f"seq_len={seq_len:4d}: {avg_ms:.2f} ms")
    
    return results
```

#### Part D: Experimental Variations
**Experiment 1: Head count vs speed**
- Test with 1, 4, 8, 16, 32 heads (adjust d_model accordingly)
- Measure memory usage and speed
- **Finding:** More heads = more parallelization but more memory overhead

**Experiment 2: Sequence length scaling**
- Test from 128 to 4096 tokens
- Plot time vs seq_len, confirm O(n²) behavior
- **Finding:** Attention dominates after ~1024 tokens

**Experiment 3: Batch size scaling**
- Test batch sizes 1, 2, 4, 8, 16
- Measure throughput (tokens/sec) vs latency
- **Finding:** Optimal batch size depends on memory bandwidth

### Lab 1.2: Inference Profiling with Real Tools
**Beyond simple timing—professional profiling:**

#### Part A: PyTorch Profiler Deep Dive
```python
def comprehensive_profile(model, input_data):
    """Use PyTorch profiler to get detailed insights"""
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for step in range(5):
            output = model(input_data)
            prof.step()
    
    # Print key metrics
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # Memory statistics
    print("\nMemory Statistics:")
    mem_events = [evt for evt in prof.events() if '[memory]' in evt.name]
    for evt in mem_events[:10]:
        print(f"{evt.name}: {evt.cuda_memory_usage / 1024**2:.1f} MB")
```

#### Part B: Bottleneck Identification
```python
def identify_bottlenecks(profile_data):
    """Analyze profiler output to find bottlenecks"""
    
    bottlenecks = []
    
    # Check for memory-bound operations
    high_memory_ops = [op for op in profile_data if op.cuda_memory_usage > 100e6]  # >100MB
    
    # Check for compute-bound operations  
    high_compute_ops = [op for op in profile_data if op.cuda_time_total > 10]  # >10ms
    
    # Check for kernel launch overhead
    many_launches = [op for op in profile_data if op.count > 1000]
    
    print("Potential bottlenecks identified:")
    if high_memory_ops:
        print(f"  Memory-bound operations: {len(high_memory_ops)} ops using >100MB")
        for op in high_memory_ops[:3]:
            print(f"    - {op.name}: {op.cuda_memory_usage/1024**2:.0f}MB")
    
    if high_compute_ops:
        print(f"  Compute-bound operations: {len(high_compute_ops)} ops taking >10ms")
        for op in high_compute_ops[:3]:
            print(f"    - {op.name}: {op.cuda_time_total:.1f}ms")
    
    return bottlenecks
```

#### Part C: FLOP Counting
```python
def calculate_flops(model, input_shape):
    """Calculate theoretical FLOPs for a forward pass"""
    
    batch_size, seq_len, d_model = input_shape
    
    # Attention FLOPs (from our earlier formula)
    attention_flops = 4 * seq_len * d_model**2 + 2 * seq_len**2 * d_model
    
    # FFN FLOPs (assuming expansion factor 4)
    ffn_flops = 8 * seq_len * d_model**2
    
    # LayerNorm FLOPs (approx)
    layernorm_flops = 3 * seq_len * d_model  # mean, variance, normalize
    
    total_flops = attention_flops + ffn_flops + layernorm_flops
    
    print(f"FLOP Analysis for shape {input_shape}:")
    print(f"  Attention: {attention_flops / 1e9:.2f} GFLOPs ({attention_flops/total_flops*100:.1f}%)")
    print(f"  FFN:       {ffn_flops / 1e9:.2f} GFLOPs ({ffn_flops/total_flops*100:.1f}%)")
    print(f"  LayerNorm: {layernorm_flops / 1e9:.2f} GFLOPs ({layernorm_flops/total_flops*100:.1f}%)")
    print(f"  Total:     {total_flops / 1e9:.2f} GFLOPs")
    
    # Compare with theoretical GPU performance
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        peak_tflops = device.multi_processor_count * device.clock_rate * 2 / 1e3  # Approximate
        
        # Assuming we achieve 50% of peak (realistic for transformers)
        achievable_tflops = peak_tflops * 0.5
        
        theoretical_time_ms = total_flops / (achievable_tflops * 1e9) * 1000
        
        print(f"\nGPU: {device.name}")
        print(f"  Peak TFLOPS: {peak_tflops:.1f}")
        print(f"  Achievable (50%): {achievable_tflops:.1f} TFLOPS")
        print(f"  Theoretical minimum time: {theoretical_time_ms:.2f} ms")
```

### Lab 1.3: Memory Hierarchy Experiments
**Understanding the memory wall:**

#### Part A: Batch Size vs Memory Trade-off
```python
def batch_size_experiment(model_factory, max_batch=32, seq_len=512):
    """Find optimal batch size for given memory constraints"""
    
    results = []
    
    for batch_size in [1, 2, 4, 8, 16, 32]:
        if batch_size > max_batch:
            break
            
        try:
            # Create fresh model for accurate memory measurement
            torch.cuda.empty_cache()
            
            model = model_factory().cuda()
            input_data = torch.randn(batch_size, seq_len, model.d_model).cuda()
            
            # Measure memory before and after
            torch.cuda.reset_peak_memory_stats()
            before_mem = torch.cuda.memory_allocated()
            
            output = model(input_data)
            
            after_mem = torch.cuda.memory_allocated()
            peak_mem = torch.cuda.max_memory_allocated()
            
            # Measure throughput
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for _ in range(100):
                _ = model(input_data)
            end.record()
            
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
            
            tokens_per_sec = (batch_size * seq_len * 100) / (elapsed_ms / 1000)
            
            results.append({
                'batch_size': batch_size,
                'memory_mb': (peak_mem - before_mem) / 1024**2,
                'throughput_tokens_per_sec': tokens_per_sec,
                'memory_per_token_mb': (peak_mem - before_mem) / (batch_size * seq_len) / 1024**2
            })
            
            del model, input_data, output
            
        except RuntimeError as e:
            print(f"Batch size {batch_size} failed: {e}")
            break
    
    return results
```

#### Part B: Memory Access Pattern Analysis
```python
def analyze_memory_patterns():
    """Demonstrate importance of memory access patterns"""
    
    # Test 1: Contiguous vs non-contiguous memory
    size = 1024 * 1024 * 100  # 100MB
    
    contiguous = torch.randn(size, device='cuda')
    non_contiguous = contiguous[::2]  # Every other element
    
    # Time memory access
    torch.cuda.synchronize()
    start = time.time()
    
    # Access in order (cache-friendly)
    for i in range(0, size, 1024):
        _ = contiguous[i:i+1024].sum()
    
    torch.cuda.synchronize()
    contiguous_time = time.time() - start
    
    # Access strided (cache-unfriendly)
    torch.cuda.synchronize()
    start = time.time()
    
    for i in range(0, size//2, 512):
        _ = non_contiguous[i:i+512].sum()
    
    torch.cuda.synchronize()
    non_contiguous_time = time.time() - start
    
    print(f"Memory Access Pattern Comparison:")
    print(f"  Contiguous access: {contiguous_time:.3f}s")
    print(f"  Strided access:    {non_contiguous_time:.3f}s")
    print(f"  Slowdown factor:   {non_contiguous_time/contiguous_time:.1f}x")
```

#### Part C: Cache Hierarchy Experiment
```python
def cache_hierarchy_demo():
    """Show impact of cache sizes on performance"""
    
    sizes = [2**i for i in range(10, 28)]  # 1KB to 256MB
    
    times = []
    
    for size in sizes:
        # Create array that's 4x larger than we access to prevent caching
        total_size = size * 4
        data = torch.randn(total_size, device='cuda')
        
        # Access with different strides to show cache effects
        torch.cuda.synchronize()
        start = time.time()
        
        # Sequential access (best case)
        acc = 0
        for i in range(0, total_size, size // 1024 or 1):
            acc += data[i]
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        times.append((size, elapsed))
        
        # Keep data alive but don't let it cache
        del data
        
    # Plot results to show cache size boundaries
    print("Cache hierarchy demonstration:")
    for size, elapsed in times:
        print(f"  Size {size/1024:7.1f}KB: {elapsed*1e6:.1f}μs per access")
```

## 🧮 Mathematical Foundations with Practical Interpretation

### 1. Attention Complexity: From Formula to Implementation

**Derivation step-by-step:**

Given:
- Sequence length: n
- Hidden dimension: d
- Number of heads: h
- Head dimension: d_h = d/h

**Step 1: Q/K/V projections**
Each projection: `x @ W` where x ∈ ℝ^{n×d}, W ∈ ℝ^{d×d}
FLOPs: 2 × n × d × d = 2nd² (multiply-add = 2 FLOPs per element)
Three projections: 6nd² FLOPs

**Step 2: Attention scores**
Q ∈ ℝ^{n×d_h}, K ∈ ℝ^{n×d_h}
QK^T ∈ ℝ^{n×n}, each element: Σ_{k=1}^{d_h} Q_{ik}K_{jk}
FLOPs: n × n × d_h × 2 = 2n²d_h

**Step 3: Softmax**
For each row i (over j):
- exp(x_ij - max_j x_ij): n operations
- sum_j: n operations  
- divide: n operations
Total: 3n² FLOPs

**Step 4: Attention × V**
A ∈ ℝ^{n×n}, V ∈ ℝ^{n×d_h}
AV ∈ ℝ^{n×d_h}, each element: Σ_{k=1}^{n} A_{ik}V_{kj}
FLOPs: n × n × d_h × 2 = 2n²d_h

**Step 5: Output projection**
Same as Step 1: 2nd² FLOPs

**Total:** 8nd² + 4n²d_h + 3n² FLOPs

**Practical interpretation:**
- For n << d: Dominated by 8nd² (linear layers)
- For n >> d: Dominated by 4n²d_h (attention)
- **Crossover point:** When 8nd² = 4n²d_h ⇒ n = 2d/d_h = 2h

Example: h=32 heads ⇒ n=64 tokens is crossover. For sequences <64, linear layers dominate. For >64, attention dominates.

### 2. Memory Requirements: Planning Your GPU Purchase

**Complete memory model:**

**Persistent memory (always needed):**
1. Model weights: P × bytes_per_param
   - FP32: 4 bytes, FP16/BF16: 2 bytes, INT8: 1 byte, INT4: 0.5 bytes
2. Optimizer states (training only): Usually 2× weights
3. Gradients (training only): Same size as weights

**Transient memory (per inference):**
1. Activations: L × n × d × bytes_per_activation
2. KV cache: L × 2 × n × h × d_h × bytes_per_param
3. Attention matrix: n × n × bytes_per_element (can be recomputed)

**Example calculation for LLaMA 2 7B inference:**
- Parameters: 7B × 2 bytes (FP16) = 14GB
- Batch size 1, n=2048:
  - Activations: 32 layers × 2048 × 4096 × 2 bytes ≈ 0.5GB
  - KV cache: 32 × 2 × 2048 × 32 × 128 × 2 bytes ≈ 1GB
- **Total:** 14 + 0.5 + 1 = 15.5GB → needs 16GB GPU

**Memory optimization strategies:**
1. **Gradient checkpointing:** Store only some activations, recompute others
2. **CPU offloading:** Move less-used tensors to CPU
3. **Model parallelism:** Split across multiple GPUs
4. **Quantization:** Reduce precision of weights/activations

### 3. Roofline Model: Understanding Performance Limits

**The roofline model:** Performance ≤ min(Peak FLOPs, Bandwidth × Arithmetic Intensity)

**Arithmetic Intensity (AI):** FLOPs / Bytes transferred

**For transformer layers:**
- Attention: AI ≈ (4n²d_h) / (8n² bytes) = d_h/2 (assuming 4-byte elements)
- Linear layers: AI ≈ (2nd²) / (4nd bytes) = d/2

**Example: d=4096, d_h=128**
- Attention AI: 128/2 = 64 FLOPs/byte
- Linear AI: 4096/2 = 2048 FLOPs/byte

**GPU characteristics (A100):**
- Peak FP16 TFLOPS: 312
- Memory bandwidth: 2TB/s = 2000 GB/s
- Compute-bound threshold: AI > Peak TFLOPS / Bandwidth = 312 / 2000 = 0.156 TFLOPS/GB = 156 FLOPs/byte

**Analysis:**
- Attention (AI=64) < 156 → **memory-bound**
- Linear layers (AI=2048) > 156 → **compute-bound**

**Implication:** Optimizing attention requires reducing memory traffic. Optimizing linear layers requires maximizing FLOP utilization.

## 🔬 Advanced Topics: Preparing for Production

### 1. Kernel Fusion: The 2× Speedup Trick
**What:** Combine multiple operations into one kernel
**Why:** Reduce memory traffic and kernel launch overhead

**Examples in transformers:**
- **LayerNorm + Residual:** Fuse normalization with add
- **Linear + GeLU:** Combine matrix multiply with activation
- **Attention fusion:** QKV projection + attention in one kernel

**Implementation sketch:**
```python
# Instead of:
x = layer_norm(x)
x = x + residual

# Fused kernel does both in one memory pass
x = fused_layer_norm_add(x, residual, weight, bias, eps)
```

**Performance impact:** Typically 1.5-3× speedup for memory-bound operations.

### 2. Mixed Precision: Free Speed (Almost)
**Strategy:** Use lower precision where possible, higher where needed

**Common patterns:**
- **Weights:** FP16 or INT8 (2× or 4× memory reduction)
- **Activations:** FP16 (2× memory reduction)
- **Accumulators:** FP32 (maintain accuracy)

**PyTorch automatic mixed precision:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Accuracy considerations:**
- Some operations (softmax, layer norm) need FP32 for stability
- Gradient scaling prevents underflow in FP16
- **Rule of thumb:** 2× speedup, <0.1% accuracy loss

### 3. Early Exit: Don't Compute What You Don't Need
**Idea:** Some inputs are easy, don't need all layers

**Strategies:**
1. **Confidence-based:** Exit when output probability > threshold
2. **Layer skipping:** Skip some intermediate layers
3. **Adaptive computation time:** Vary layers per token

**Implementation:**
```python
class EarlyExitTransformer(nn.Module):
    def __init__(self, base_model, exit_layers=[6, 12, 18, 24]):
        super().__init__()
        self.base_model = base_model
        self.exit_layers = exit_layers
        self.exit_classifiers = nn.ModuleList([
            nn.Linear(d_model, num_classes) for _ in exit_layers
        ])
    
    def forward(self, x, exit_threshold=0.9):
        for i, layer in enumerate(self.base_model.layers):
            x = layer(x)
            
            if i in self.exit_layers:
                logits = self.exit_classifiers[self.exit_layers.index(i)](x[:, 0])  # CLS token
                probs = F.softmax(logits, dim=-1)
                max_prob, _ = probs.max(dim=-1)
                
                if max_prob > exit_threshold:
                    return logits, i  # Early exit
        
        return self.base_model.head(x[:, 0]), len(self.base_model.layers)
```

**Performance:** 2-5× speedup on easy examples, no slowdown on hard ones.

## 📊 Performance Benchmarks: Establishing Baselines

### Benchmark Suite Design
A good benchmark tests multiple dimensions:

```python
class InferenceBenchmark:
    def __init__(self):
        self.models = {
            'gpt2-small': transformers.GPT2LMHeadModel.from_pretrained('gpt2'),
            'gpt2-medium': transformers.GPT2LMHeadModel.from_pretrained('gpt2-medium'),
            'llama-7b': None,  # Would load if available
        }
        
        self.test_cases = [
            {'batch_size': 1, 'seq_len': 128},
            {'batch_size': 1, 'seq_len': 512},
            {'batch_size': 1, 'seq_len': 2048},
            {'batch_size': 4, 'seq_len': 128},
            {'batch_size': 4, 'seq_len': 512},
        ]
    
    def run_benchmark(self):
        results = {}
        
        for model_name, model in self.models.items():
            if model is None:
                continue
                
            model_results = []
            model = model.cuda().eval()
            
            for test_case in self.test_cases:
                bs, seq_len = test_case['batch_size'], test_case['seq_len']
                
                # Prepare input
                input_ids = torch.randint(0, 50257, (bs, seq_len)).cuda()
                
                # Measure
                torch.cuda.synchronize()
                start = time.time()
                
                with torch.no_grad(), torch.cuda.amp.autocast():
                    for _ in range(10):  # Average over multiple runs
                        _ = model(input_ids)
                
                torch.cuda.synchronize()
                elapsed = time.time() - start
                
                avg_ms = elapsed / 10 * 1000
                
                # Memory
                torch.cuda.reset_peak_memory_stats()
                _ = model(input_ids)
                peak_mb = torch.cuda.max_memory_allocated() / 1024**2
                
                model_results.append({
                    'test_case': test_case,
                    'latency_ms': avg_ms,
                    'memory_mb': peak_mb,
                    'tokens_per_sec': (bs * seq_len * 1000) / avg_ms,
                })
            
            results[model_name] = model_results
        
        return results
```

### Expected Results (RTX 4090)

**GPT-2 Small (124M parameters):**
- seq_len=128, batch=1: ~5ms (25,600 tokens/sec)
- seq_len=2048, batch=1: ~150ms (13,653 tokens/sec)
- Memory: ~500MB peak

**GPT-2 Medium (355M parameters):**
- seq_len=128, batch=1: ~15ms (8,533 tokens/sec)
- seq_len=2048, batch=1: ~450ms (4,551 tokens/sec)
- Memory: ~1.5GB peak

**Scaling laws observed:**
- Latency scales linearly with parameters (approximately)
- Memory scales sublinearly (activations don't grow as fast)
- Throughput improves with batch size until memory bound

## 🚀 Production Considerations from Day 1

### 1. Model Format Optimization Pipeline
**Don't wait until deployment to optimize:**

```python
# Development to production pipeline
def optimization_pipeline(model):
    # Step 1: Export to ONNX
    torch.onnx.export(model, dummy_input, "model.onnx")
    
    # Step 2: ONNX Runtime optimization
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Step 3: TensorRT conversion (if NVIDIA)
    trt_engine = tensorrt.Builder(...)
    
    # Step 4: Quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    return optimized_model
```

### 2. Serving Infrastructure Design
**Think about deployment during development:**

**API Design:**
```python
# FastAPI server example
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

@app.post("/generate")
async def generate(request: InferenceRequest):
    # Tokenization
    inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
    
    # Generation with streaming
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            streamer=streamer  # For Server-Sent Events
        )
    
    return {"text": tokenizer.decode(outputs[0])}
```

**Load Testing:**
- Use `locust` or `k6` to simulate concurrent users
- Measure P95 latency under load
- Identify maximum concurrent requests before QoS degradation

### 3. Monitoring & Observability
**What to monitor:**
1. **Performance metrics:** Latency (P50, P95, P99), throughput
2. **Resource metrics:** GPU utilization, memory usage, temperature
3. **Quality metrics:** Perplexity, output length distribution
4. **Business metrics:** Cost per request, error rate

**Alerting thresholds:**
- Latency P95 > 1s: Warning
- GPU memory > 90%: Warning  
- Error rate > 1%: Critical

## 📝 Weekly Deliverables with Rubric

### 1. Code Submission (40 points)
**Requirements:**
- [ ] Complete Lab 1.1 with all experimental variations (10 points)
- [ ] Complete Lab 1.2 with comprehensive profiling output (10 points)
- [ ] Complete Lab 1.3 with memory hierarchy analysis (10 points)
- [ ] Clean, well-documented code with type hints (5 points)
- [ ] Unit tests for key functions (5 points)

**Grading criteria:**
- **Excellent (35-40):** All labs complete with insightful experiments beyond requirements
- **Good (30-34):** All labs complete with basic experiments
- **Satisfactory (25-29):** Most labs complete, some experiments missing
- **Needs improvement (<25):** Incomplete or poorly documented

### 2. Written Analysis (40 points)
**Structure:**
1. **Transformer complexity analysis** (10 points)
   - Derive FLOP formulas for your specific model configuration
   - Compare theoretical vs measured performance
   - Identify whether attention or linear layers dominate

2. **Profiling results interpretation** (15 points)
   - Present key profiling metrics in tables/plots
   - Identify top 3 bottlenecks with evidence
   - Suggest specific optimizations for each bottleneck

3. **Memory hierarchy insights** (15 points)
   - Analyze batch size vs throughput trade-off
   - Calculate memory bandwidth utilization
   - Recommend optimal batch size for your hardware

**Grading criteria:**
- **Depth of analysis:** Beyond surface-level observations
- **Data-driven conclusions:** Numbers support claims
- **Actionable insights:** Clear optimization recommendations

### 3. Research Review (20 points)
**For each required paper:**
- [ ] **Summary** (2 points): 3-5 sentence overview of key contributions
- [ ] **Key insights** (3 points): 2-3 important ideas for inference optimization
- [ ] **Critical analysis** (3 points): Strengths, limitations, open questions
- [ ] **Connection to labs** (2 points): How paper insights apply to your experiments

**Total:** 3 papers × 10 points = 30 points (scale to 20)

## 🔧 Setup Instructions: Beyond pip install

### Development Environment Best Practices

#### 1. Reproducible Environment
```bash
# requirements.txt with exact versions
torch==2.2.0
transformers==4.36.0
accelerate==0.25.0

# Plus development tools
black==23.11.0  # Code formatting
isort==5.12.0   # Import sorting
mypy==1.7.0     # Type checking
pytest==7.4.3   # Testing
```

#### 2. GPU Setup Verification
```python
# Comprehensive GPU check
import torch
import subprocess

print("=" * 60)
print("GPU Setup Verification")
print("=" * 60)

# PyTorch CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    # GPU details
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Multiprocessors: {props.multi_processor_count}")
        
    # Memory status
    print(f"\nCurrent memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Current memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # Performance test
    print("\nRunning performance test...")
    a = torch.randn(10000, 10000, device='cuda')
    b = torch.randn(10000, 10000, device='cuda')
    
    torch.cuda.synchronize()
    import time
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    flops = 2 * 10000**3 / elapsed
    print(f"  Matrix multiply: {flops / 1e12:.2f} TFLOPS")

# System information
print("\n" + "=" * 60)
print("System Information")
print("=" * 60)

# CPU
import platform
print(f"System: {platform.system()} {platform.release()}")
print(f"Processor: {platform.processor()}")

# Memory
import psutil
mem = psutil.virtual_memory()
print(f"RAM: {mem.total / 1024**3:.1f} GB total, {mem.available / 1024**3:.1f} GB available")

# Disk
disk = psutil.disk_usage('/')
print(f"Disk: {disk.total / 1024**3:.1f} GB total, {disk.free / 1024**3:.1f} GB free")
```

#### 3. Common Setup Issues and Solutions

**Issue 1: CUDA version mismatch**
```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Issue 2: Out of memory errors**
```python
# Reduce memory usage
model = model.half()  # Convert to FP16
torch.cuda.empty_cache()  # Clear cache

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Use CPU offloading
model.cpu()  # Move to CPU when not needed
```

**Issue 3: Slow performance**
```python
# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere+
torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    outputs = model(inputs)
```

### Alternative Setups

#### Google Colab (Free GPU)
```python
# Colab setup cell
!pip install torch transformers accelerate
!nvidia-smi  # Check GPU

# Mount Google Drive for saving models
from google.colab import drive
drive.mount('/content/drive')
```

#### AWS EC2 (p3.2xlarge)
```bash
# Launch instance with Deep Learning AMI
# Connect and verify
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

#### Local Linux with Docker
```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install torch transformers

WORKDIR /workspace
COPY . .
```

## 🎯 Success Criteria: How to Know You're Ready

### Foundational Understanding Check
You should be able to:

1. **Explain to a colleague** why increasing batch size improves throughput but increases latency
2. **Calculate** whether a specific model will fit in your GPU memory
3. **Identify** whether an inference bottleneck is compute-bound or memory-bound
4. **Recommend** specific optimizations based on profiling data

### Practical Skills Verification
Complete these without looking at solutions:

1. Given a model with 1B parameters (FP16) and sequence length 1024, calculate:
   - Minimum GPU memory needed for batch size 1
   - Theoretical FLOPs per forward pass
   - Expected latency on an A100 (assuming 50% of peak TFLOPS)

2. Profile a simple model and:
   - Identify the layer with highest memory usage
   - Find the operation with longest CUDA time
   - Suggest one optimization for each bottleneck

3. Implement a memory-efficient version of attention that:
   - Avoids materializing the full n×n attention matrix
   - Still produces exact same output
   - Reduces memory usage for n > 1024

### Knowledge Application
**Scenario:** Your company wants to deploy a 7B parameter model for a chat application. They have 4× A100 GPUs (40GB each). Expected load: 100 concurrent users, average conversation length 10 messages (200 tokens each).

**Questions to answer:**
1. Will the model fit in GPU memory? With what batch size?
2. What's the expected latency per token?
3. How many requests per second can you handle?
4. What optimizations would you implement first?

## 📚 Additional Resources: Beyond the Basics

### Advanced Reading
1. **"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"** (Dao et al., 2022)
   - Read after Week 2, but preview the motivation

2. **"Efficient Large-Scale Language Model Training on GPU Clusters"** (Narayanan et al., 2021)
   - Distributed training but relevant for inference scaling

3. **"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"** (Rajbhandari et al., 2020)
   - Memory optimization techniques applicable to inference

### Tools to Explore
1. **Nsight Systems:** NVIDIA's system-wide performance analysis
2. **PyTorch Profiler with TensorBoard:** Visual profiling
3. **Weights & Biases:** Experiment tracking and collaboration
4. **MLPerf Inference Benchmark:** Industry-standard benchmarks

### Community Engagement
1. **Hugging Face Discord:** #training-optimization channel
2. **PyTorch Forums:** Performance optimization section
3. **MLOps.community:** Slack for production ML
4. **Papers With Code:** Latest research implementations

## 🚀 What's Next?

### Week 2 Preview: Core Optimization Techniques
Week 1 gave you the foundation. Week 2 builds on it with:

1. **KV Caching:** The 10× speedup trick for generation
2. **FlashAttention:** IO-aware attention implementation
3. **Batch Size Optimization:** Finding the sweet spot
4. **Kernel Fusion:** Combining operations for efficiency

**Connecting Week 1 to Week 2:**
- Your understanding of attention complexity will help optimize KV caching
- Your profiling skills will identify where FlashAttention helps most
- Your memory hierarchy knowledge informs batch size choices
- Your computational complexity analysis shows what to fuse

### Preparing for Week 2
**Recommended preparation:**
1. Complete all Week 1 labs thoroughly
2. Read the FlashAttention paper abstract
3. Experiment with different batch sizes on your hardware
4. Think about what parts of your Week 1 implementation were slowest

---

**Estimated Time Commitment:** 15-20 hours (more if doing all extensions)  
**Difficulty Level:** ⭐⭐⭐☆☆ (Foundation building with depth)  
**Next Week:** Core Optimization Techniques – where the real speedups begin!

## 📝 Changelog

### Version 1.1 (2026-04-07)
- **Enhanced** with detailed mathematical derivations
- **Added** practical examples with actual code
- **Expanded** memory hierarchy explanations  
- **Included** production considerations from day 1
- **Added** common pitfalls and debugging tips
- **Enhanced** labs with more experiments
- **Added** success criteria and verification exercises
- **Included** comprehensive setup troubleshooting

### Version 1.0 (2026-04-06)
- Initial version with basic structure
- Core concepts and labs outline
- Required reading list
- Basic mathematical foundations

---

**Feedback welcome:** This is a living document. As you work through the material, suggest improvements via GitHub issues or discussions.