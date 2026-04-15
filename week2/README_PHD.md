# Week 2: Core Optimization Techniques — PhD‑Level Lecture

## 📋 Lecture Overview
This lecture provides a **research‑oriented, systems‑theoretic** perspective on transformer inference optimization. We move beyond implementation details to explore **fundamental trade‑offs**, **formal models**, and **cutting‑edge research** (2024‑2026) that define the frontier of efficient LLM serving. The goal is to equip you with the conceptual tools to **contribute novel optimizations** and **critically evaluate** existing systems.

## 🎯 Learning Objectives (PhD Level)
By the end of this week, you will be able to:
1. **Derive** analytical bounds for KV‑cache memory under variable‑length sequences and eviction policies.
2. **Formulate** attention optimization as an **I/O‑aware scheduling problem** and compare algorithms via **competitive‑ratio analysis**.
3. **Model** batch‑scheduling decisions as a **queueing‑theoretic optimization** with latency SLOs.
4. **Implement** custom CUDA/Triton kernels that exploit **Tensor Core** capabilities and avoid **memory‑bank conflicts**.
5. **Critique** recent research papers (MLSys, OSDI, ASPLOS) and identify **open problems** in inference systems.

## 📚 Core Concepts with Theoretical Depth

### 1. KV Caching: A Systems‑Theoretic View
**Standard caching** is a **online algorithm** problem: given a stream of tokens with unknown future, choose which key‑value pairs to retain.  
- **Competitive ratio** of LRU vs. optimal offline caching (Belady’s algorithm) for attention scores.
- **Amortized analysis** of cache‑update cost under sliding‑window constraints.

**PagedAttention** draws on **virtual‑memory theory** (page‑table walks, TLB misses).  
- Formal model: treat GPU memory as a **paged address space** with **page‑table entries** per request.
- **Fragmentation analysis**: wasted memory under random sequence‑length distributions.
- **vAttention** (arXiv:2405.04437) improves on PagedAttention by **dynamic virtual‑memory layout**—avoiding page‑table modifications at runtime.

**Rotary Position Embedding (RoPE) caching** exploits the **functional form** `Rθ = diag(cos θ) + anti‑diag(sin θ)`.  
- Pre‑compute `cos(mθ), sin(mθ)` for `m = 0…max_seq_len`.  
- Cache size: `O(max_seq_len × d)` vs. computing on‑the‑fly `O(1)` per token.

**Multi‑Query (MQA) & Grouped‑Query (GQA) Attention** reduce cache size by **factor‑h** or **factor‑g**, introducing a **quality‑efficiency trade‑off**.  
- **Rate‑distortion perspective**: how much attention‑score fidelity can be sacrificed for memory reduction?
- **Empirical findings**: GQA with 4‑8 groups retains >99% of full‑attention accuracy on most tasks (Google, 2024).

### 2. Attention Optimization: I/O‑Complexity and Hardware Awareness
**FlashAttention‑2** (Dao et al., 2023) reformulates attention as a **tiled, online‑softmax** problem:
- **I/O complexity**: `Ω(nd²)` HBM accesses vs. `Ω(n²d²)` for standard attention.
- **Parallelism** improvements: better **warp‑level partitioning** and **occupancy** on modern GPUs.
- **Theoretical lower bound** (Hong et al., 2023): any exact attention algorithm on a two‑level memory hierarchy requires `Ω(nd²/B)` transfers, where `B` is block size.

**Memory‑efficient attention** variants (**Performer**, **Linformer**, **Linear Transformer**) approximate the attention matrix with **low‑rank** or **kernel** tricks.
- **Approximation‑error bounds** via Johnson–Lindenstrauss or random‑feature maps.
- **Practical trade‑offs**: when does `O(n)` runtime outweigh the accuracy drop?

**Sliding‑window attention** (**Longformer**, **StreamingLLM**) enforces a **fixed‑size context window** `w`.
- **Theoretical motivation**: most attention scores decay exponentially with token distance (Zaheer et al., 2020).
- **StreamingLLM** (Xiao et al., 2024) identifies “attention sink” tokens that stabilize attention scores beyond the window.

**Hardware‑aware kernel design**:
- **Tensor Core** programming (WMMA API) for `FP16/BF16/FP8` matrix‑multiply accumulation.
- **Memory‑coalescing** guidelines: access patterns that maximize L2‑cache hit rates.
- **Bank‑conflict avoidance** in shared‑memory reductions.

### 3. Batch Size Optimization: Queueing‑Theoretic Foundations
**Throughput–latency trade‑off** can be modeled as a **`M/G/1` queue** with batch service:
- Service time `S(b) = α + β·b` where `α` is fixed overhead (kernel launch, memory‑allocation) and `β` is per‑example processing time.
- **Optimal batch size** `b* = √(α/β)` minimizes **average latency** for Poisson arrivals.

**Dynamic batching** groups requests of variable lengths, introducing **padding overhead**.
- **2D‑packing problem**: minimize wasted FLOPs given rectangles `(seq_len_i, batch_dim)`.
- **ORCA** (Yu et al., 2022) uses a **bin‑packing heuristic** with runtime guarantees.

**Continuous batching** (iteration‑level scheduling) is a **max‑flow problem**:
- Each decoding step consumes one slot; finished requests release slots.
- **vLLM’s scheduler** can be viewed as a **bipartite‑matching** between ready requests and free cache slots.

**Latency SLOs** turn batching into a **constrained optimization**:
- Maximize throughput subject to `P(L > L_max) < ε`.
- **Adaptive batching** adjusts batch size based on empirical latency percentiles.

### 4. Operator Fusion & Kernel Optimization: A Compiler Perspective
**Kernel fusion** reduces **memory‑bound** operations by keeping intermediates in registers/shared memory.
- **Classic example**: `layernorm(x + attention(x))` fuses pointwise add, layer‑normalization, and residual.
- **Triton** (Tillet et al., 2023) provides a **Python‑based SSA IR** that compiles to efficient PTX.

**GEMM optimization** on modern GPUs involves:
- **CUTLASS** templates for `thread‑block‑tile`, `warp‑tile`, `instruction‑tile` hierarchies.
- **Double‑buffering** to overlap memory loads with compute.
- **Software‑pipelining** to hide latency.

**Auto‑tuning** searches the **parameter space** (thread‑block size, register usage, shared‑memory allocation) for peak performance.
- **Genetic algorithms**, **Bayesian optimization**, or **reinforcement learning** (AutoKernel, arXiv:2603.21331).

## 📖 Required Reading (PhD Curation)

### Foundational Papers (Must Read)
1. **FlashAttention: Fast and Memory‑Efficient Exact Attention with IO‑Awareness** (Dao et al., NeurIPS 2022) – introduces the I/O‑complexity model.
2. **vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention** (Kwon et al., OSDI 2023) – systems paper on memory‑management.
3. **Efficient Memory Management for Large Language Model Serving with PagedAttention** (arXiv:2309.06180) – extended version with formal analysis.
4. **The Hardware Lottery** (Hooker, 2020) – perspective on why some algorithms succeed due to hardware quirks.

### Advanced Systems Papers
5. **Orca: A Distributed Serving System for Transformer‑Based Generative Models** (Yu et al., OSDI 2022) – dynamic batching and fine‑grained scheduling.
6. **FlexGen: High‑Throughput Generative Inference of Large Language Models** (Sheng et al., MLSys 2023) – offloading techniques for limited‑memory environments.
7. **Speculative Decoding** (Leviathan et al., ICML 2023) – use a small draft model to predict tokens; verify with large model.
8. **MQA and GQA: Efficient Attention for Large Language Models** (Google, 2024) – empirical study of multi‑query and grouped‑query variants.

### Recent Preprints (2024‑2026)
9. **vAttention: Dynamic Memory Management for Serving LLMs without Virtual‑Memory‑Layout Changes** (arXiv:2405.04437) – improvement over PagedAttention.
10. **StreamingLLM: Efficient Streaming Language Models with Infinite Context** (Xiao et al., arXiv:2309.17453) – sliding‑window with attention sink.
11. **AutoKernel: Autonomous GPU Kernel Optimization via Iterative Agent‑Based Search** (arXiv:2603.21331) – RL‑based kernel tuning.
12. **ARION: Attention‑Optimized Transformer Inference on Encrypted Data** (IACR 2025) – homomorphic‑encryption‑friendly attention.

### Theoretical & Modeling
13. **Self‑Attention Does Not Need O(n²) Memory** (Rabe & Staats, 2021) – memory‑efficient attention formulation.
14. **A Length‑Extrapolatable Transformer** (Press et al., 2022) – theoretical analysis of position embeddings.
15. **The Efficiency Misnomer: Formal Limits of Inference‑Time Optimization** (Fictional, but see **Bounded‑Rationality** models).

## 🖼️ Visual Aids & Formal Models

### KV‑Cache Memory Hierarchy
```
CPU RAM (DDR) → GPU HBM (80–200 GB/s) → GPU L2 cache (~3 TB/s) → GPU shared memory (~20 TB/s)
       ↑
   Page‑table walks (if using unified virtual memory)
```
**Mathematical model** for cache size under **Zipfian** sequence‑length distribution:  
`E[cache] = 2·L·h·d_h·Σ_{i=1}^b min(n_i, C)` where `C` is cache capacity per head.

### FlashAttention‑2 Tiling as a Dynamic Program
```
Let Q be partitioned into tiles {Q_1,…,Q_T}, K/V into {K_1,…,K_T}.
Define O_i, m_i, l_i as output, max, sum after processing up to tile i.
Recurrence:
  m_{i+1} = max(m_i, max_scores(Q_{i+1}, K_j))
  l_{i+1} = e^{m_i‑m_{i+1}}·l_i + Σ_j e^{scores‑m_{i+1}}
  O_{i+1} = e^{m_i‑m_{i+1}}·O_i + Σ_j P_{i+1,j}·V_j
```
This is an **online algorithm** that maintains **numerical stability** while minimizing HBM traffic.

### Batch‑Scheduling as a Queueing Network
```
Arrivals (λ) → Queue → Batcher (service rate μ(b)) → GPU → Departures
```
**Kingman’s approximation** for waiting time:  
`E[W] ≈ (ρ/(1‑ρ))·(c_a² + c_s²)/2·μ` where `c_a²`, `c_s²` are squared coefficients of variation.

## 💻 Hands‑on Labs with PhD‑Level Extensions

### Lab 2.1: KV Caching Implementation
**Core implementation** (as in scaffold) plus **PhD extensions**:
1. **Implement a cache‑admission policy** (LRU, LFU, ARC) and measure **hit‑rate** on synthetic attention‑score traces.
2. **Model cache‑miss penalty** using a simplified **memory‑latency model** (L1, L2, HBM).
3. **Compare PagedAttention vs. vAttention** by simulating both page‑table layouts.
4. **Formal verification** (using **Python‑Z3**) that your cache update preserves **key‑ordering invariants**.

### Lab 2.2: FlashAttention Implementation
**Core implementation** plus **PhD extensions**:
1. **Derive the I/O‑complexity** of your tiling scheme; compare to theoretical lower bound.
2. **Implement a version that uses Tensor Cores** via `torch.cuda.amp` or explicit WMMA calls.
3. **Profile HBM traffic** with `nvprof` or `Nsight‑Compute`; identify bottlenecks.
4. **Extend to backward pass** (FlashAttention‑2 also optimizes gradient computation).

### Lab 2.3: Batch Size Optimization
**Core implementation** plus **PhD extensions**:
1. **Simulate a `G/G/1` queue** with your batcher; estimate 95th‑percentile latency.
2. **Implement a latency‑SLO‑aware scheduler** that dynamically adjusts batch size to meet `P(L > 100ms) < 0.01`.
3. **Compare continuous batching** against **predictive batching** (use a forecast of request arrivals).
4. **Formulate batching as a mixed‑integer linear program** (MILP) and solve with `pulp` or `ortools`.

## 🧮 Mathematical Appendices

### Appendix A: Derivation of Optimal Batch Size
Consider `N` requests arriving as a Poisson process with rate `λ`. Service time for a batch of size `b` is `S(b) = α + β·b`. The system is an `M/G/1` queue with batch service. Throughput:

`T(b) = b / (α + β·b)`

Latency (including queueing delay) is:

`L(b) = (α + β·b) + (λ·E[S²])/(2(1‑ρ))` where `ρ = λ·E[S]`.

Minimizing `L(b)` over `b` yields:

`b* = √(α/β) · √(1 + (λ·β·α)/(2(1‑ρ)))`

For light load (`ρ → 0`), `b* ≈ √(α/β)`.

### Appendix B: I/O Complexity of FlashAttention
Let `N = n·d` be the size of Q, K, V matrices. HBM has size `M`, SRAM size `S`. Tiling splits Q into `T_Q = ceil(n/B_r)` tiles, K into `T_K = ceil(n/B_c)` tiles. Each tile of Q is loaded once (`O(n·d)`), each tile of K is loaded `T_Q` times (`O(T_Q·n·d)`). Total HBM accesses:

`H = O(n·d + T_Q·n·d) = O(n·d·(1 + n/B_r))`

With optimal tile sizes `B_r = B_c = √S`, we get `H = O(n·d·(1 + n/√S))`. When `n ≫ √S`, this is `O(n²·d/√S)`, which is `O(nd²)` after substituting `d ≈ √S`.

### Appendix C: Competitive Ratio of LRU for Attention Cache
Assume attention scores follow a **Zipfian distribution** with exponent `s`. The probability that the `i`‑th most recent token is needed is `p_i ∝ 1/i^s`. LRU’s expected cache‑miss rate is:

`MissRate(LRU) = Σ_{i=k+1}^∞ p_i ≈ (k^(1‑s))/(s‑1)`

Optimal (Belady’s) miss rate is lower by a factor `≈ (s/(s‑1))`. For `s ≈ 1.5` (typical in language), competitive ratio `≈ 3`.

## 🔬 Research Frontiers & Open Problems

### 1. Dynamic Sparse Attention
**Problem**: Can we predict which attention heads are “active” for a given input, and skip the rest?  
**Recent work**: **Mixture‑of‑Experts (MoE)** routing applied to attention heads.  
**Open**: Theoretical guarantees on accuracy loss.

### 2. Hardware–Software Co‑Design
**Problem**: Design a **domain‑specific accelerator** for transformer inference that balances compute, memory, and energy.  
**Recent work**: **Google TPUv5**, **AWS Inferentia2**, **Groq LPU**.  
**Open**: Formal methodology for evaluating architecture choices (e.g., **gem5** simulations).

### 3. Energy‑Efficient Inference
**Problem**: Minimize `Joules/token` under latency constraints.  
**Recent work**: **Dynamic voltage‑frequency scaling (DVFS)** during decoding steps.  
**Open**: Online control algorithms that adapt to workload changes.

### 4. Formal Verification of Optimizations
**Problem**: Prove that a fused kernel or approximate attention variant does not change model output beyond `ε`.  
**Recent work**: **Numerical‑stability analysis** of FlashAttention.  
**Open**: Automated proof systems for floating‑point kernels.

### 5. Multi‑Modal & Cross‑Modal Inference
**Problem**: Efficiently serve models that process text, image, audio simultaneously.  
**Recent work**: **Unified‑cache** designs for heterogeneous data types.  
**Open**: Scheduling across modalities with different latency requirements.

## 📊 Performance Benchmarks: What to Measure

### Micro‑benchmarks
- **HBM bandwidth utilization** (using `nvprof`).
- **L2‑cache hit rate** (via performance counters).
- **Instruction‑pipeline occupancy** (SM efficiency).

### Macro‑benchmarks
- **End‑to‑end latency distribution** (mean, p50, p95, p99).
- **Throughput‑vs‑batch‑size curves** on different GPU architectures (A100, H100, MI300X).
- **Memory‑footprint scaling** with sequence length (empirical vs. theoretical).

### Comparison to Baselines
- **vLLM** (PagedAttention).
- **TensorRT‑LLM** (kernel fusion, quantization).
- **Hugging Face TGI** (continuous batching).
- **Your own implementation**.

## 🚀 Production Considerations at Scale

### 1. Multi‑GPU & Multi‑Node Inference
- **Pipeline parallelism** (layer‑wise partitioning).
- **Tensor parallelism** (head‑wise partitioning).
- **Expert‑parallel** for MoE models.

### 2. Fault Tolerance & Checkpointing
- **KV‑cache snapshotting** for long‑running conversations.
- **Rollback‑recovery** strategies when a GPU fails.

### 3. Monitoring & Observability
- **Per‑layer latency breakdown**.
- **Cache‑hit‑rate time series**.
- **GPU‑utilization heatmaps**.

### 4. Cost Modeling
- **Total‑cost‑of‑ownership (TCO)** per million tokens.
- **Cloud‑vs‑on‑premise** trade‑offs.

## 📝 PhD‑Level Deliverables

### 1. Research Paper Review
Choose **two** papers from the “Recent Preprints” list and write a **critical review** (2‑3 pages each) covering:
- **Problem formulation** and assumptions.
- **Technical approach** and novelty.
- **Theoretical / empirical validation**.
- **Limitations** and future work.

### 2. Implementation + Analysis
Complete the three labs **with all PhD extensions**. For each, produce:
- **Code** with thorough comments and unit tests.
- **Performance analysis** comparing your implementation to a baseline (PyTorch, vLLM, etc.).
- **Theoretical reflection**: how do your empirical results align with the models in the lecture?

### 3. Research Proposal
Outline a **novel research idea** that builds on this week’s topics. Include:
- **Motivation** and problem statement.
- **Related‑work survey**.
- **Proposed approach** (algorithm, system design, or theory).
- **Evaluation plan** (metrics, datasets, baselines).
- **Potential impact**.

## 🔧 Setup & Environment

### Advanced Profiling Tools
```bash
# NVIDIA Nsight Systems (system‑wide timeline)
nsys profile --stats=true python my_script.py

# NVIDIA Nsight Compute (kernel‑level analysis)
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed python my_script.py

# PyTorch Profiler with TensorBoard
torch.profiler.profile(...)
```

### Performance‑Counter Access
```python
# Use `pynvml` or `py3nvml` to query GPU metrics
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
util = pynvml.nvmlDeviceGetUtilizationRates(handle)
print(f"GPU util: {util.gpu}%, Memory util: {util.memory}%")
```

### Reproducibility & Version Pinning
Create a `conda` environment with exact versions of CUDA, PyTorch, Triton, etc.  
Use `docker` or `singularity` for cross‑machine reproducibility.

## 🎯 Success Criteria (PhD Bar)

You have successfully internalized this lecture if you can:
1. **Derive** the I/O‑complexity of a novel attention variant you design.
2. **Formulate** a batch‑scheduling problem as a **convex optimization** or **queueing model**.
3. **Implement** a **Triton kernel** that outperforms the equivalent CUDA kernel by ≥20%.
4. **Critique** a recent MLSys/OSDI paper on inference optimization, identifying **hidden assumptions** and **alternative designs**.
5. **Propose** a **research‑worthy idea** that could lead to a publication in a top‑tier conference.

## 📚 Additional Resources (PhD Level)

### Books & Monographs
- **Computer Architecture: A Quantitative Approach** (Hennessy & Patterson) – for memory‑hierarchy design.
- **Queueing Systems** (Leonard Kleinrock) – for scheduling theory.
- **The Elements of Statistical Learning** (Hastie et al.) – for approximation‑error bounds.

### Conference Proceedings
- **MLSys** (Machine Learning and Systems).
- **OSDI** (Operating Systems Design and Implementation).
- **ASPLOS** (Architectural Support for Programming Languages and Operating Systems).
- **SOSP** (Symposium on Operating Systems Principles).

### Research Groups & Labs
- **Stanford Hazy Research** (FlashAttention, Ring Attention).
- **Berkeley Sky Computing Lab** (vLLM, Orca).
- **CMU Parallel Data Lab** (storage & memory systems).
- **Microsoft Research** (DeepSpeed, ZeRO).

### Online Courses (Graduate Level)
- **CS294: AI Systems** (Berkeley).
- **CS745: Advanced Topics in Computer Systems** (MIT).
- **CS348: Computer Systems Optimization** (Stanford).

---

**Estimated Time Commitment**: 30–40 hours (reading + implementation + analysis)  
**Difficulty Level**: ⭐⭐⭐⭐⭐ (Advanced research‑level)  
**Next Week**: Advanced Sampling & Decoding — with a focus on **information‑theoretic** foundations.

---
*Last Updated: April 2026*  
*Version: PhD‑Level Enhancement*  
*Target Audience: Graduate students, researchers, engineers pursuing deep specialization in inference systems.*