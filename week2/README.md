# Week 2: Core Optimization Techniques

## Motivation

When serving LLMs at scale, three resource constraints dominate: **memory**, **compute**, and **I/O bandwidth**. Without careful optimization, each can become a bottleneck that limits either throughput (tokens/sec) or latency (time to first token).

Consider a 70B‑parameter model running on an 80 GB A100 GPU. A single forward pass already consumes ~140 GB of memory if we materialize the full attention matrix for a 4k sequence—more than the GPU can hold. Even if we could fit it, the time to read/write that matrix from high‑bandwidth memory (HBM) would dwarf the actual compute time. And if we serve requests one‑by‑one, GPU utilization stays below 10 %, wasting expensive hardware.

This lecture walks through the three optimizations that make production‑scale inference possible:

1. **KV caching** – reduces memory from quadratic to linear in sequence length.
2. **FlashAttention** – reduces I/O traffic from quadratic to linear in sequence length (in the external memory model).
3. **Batch optimization** – trades latency for throughput, allowing the GPU to be saturated.

Together they enable systems like vLLM and Hugging Face TGI to serve hundreds of requests simultaneously with sub‑second latency.

## KV Caching

### The problem
In autoregressive decoding, each new token attends to all previous tokens. Naively recomputing the key‑value (KV) pairs for every token would require O(n²) memory and O(n²) compute per step. But the KV pairs for earlier tokens are unchanged—they can be cached.

### Standard KV cache
For a single layer with `h` attention heads, head dimension `d`, and sequence length `n`, the cache for one batch element stores:

```
memory = 2 × h × n × d × dtype_size
```

The factor 2 accounts for keys and values. For a typical 32‑layer model with 32 heads, `d = 128`, and `n = 2048`, this reaches ~2 GB per batch element—already too large for many‑batch serving.

### PagedAttention (vLLM)
PagedAttention borrows the virtual‑memory concept from operating systems. Instead of allocating a contiguous buffer for each request, it carves the total cache into fixed‑size *pages* (e.g., 16 tokens per page). A request’s cache is a list of page indices, which can be non‑contiguous.

This eliminates external fragmentation: a 17‑token request uses two pages, not a 2048‑slot buffer. It also allows **block‑level sharing** of pages between requests (e.g., when prompts share a common prefix), reducing memory duplication.

### Multi‑Query and Grouped‑Query Attention
Standard multi‑head attention (MHA) stores separate KV pairs per head, multiplying cache size by `h`. Multi‑Query Attention (MQA) shares a single KV head across all query heads; Grouped‑Query Attention (GQA) groups heads into `g` groups, each sharing a KV head.

The cache size becomes:

```
memory = 2 × g × n × d × dtype_size
```

where `g = 1` for MQA, `g = h` for MHA, and `1 < g < h` for GQA. This trades a small quality drop for a `h/g` memory reduction—often acceptable for inference.

### Cache eviction strategies
When the cache fills, we must evict old tokens. The simplest is **sliding window**: keep only the last `W` tokens. More sophisticated policies (LRU, learned predictors) can be used, but sliding window works well because attention scores decay with distance in most Transformer models.

## FlashAttention

### The I/O bottleneck
Attention’s naive implementation reads/writes the entire `n × n` attention matrix from HBM, costing O(n²) bytes moved. On an A100, the compute‑to‑memory bandwidth ratio is ~200 FLOP/byte, meaning the operation is **memory‑bound** for all but tiny sequences.

FlashAttention reorders the computation to keep the attention matrix in fast SRAM (≈20 MB on A100), only writing the final output to HBM. It splits the input into tiles that fit in SRAM, and uses an **online softmax** to accumulate results across tiles without materializing the full matrix.

### Tiling and online softmax
Let `Q`, `K`, `V` be split into tiles of size `B × d`. For each `Q` tile, we iterate over `K` tiles, computing:

```
S_ij = Q_i @ K_j^T / sqrt(d)       # tile‑tile scores
```

We cannot compute softmax over the full row yet because we haven’t seen all `K` tiles. Instead we maintain three running statistics per row:

- `m_i`: maximum score seen so far
- `l_i`: sum of exponentials (scaled by current max)
- `o_i`: running weighted sum of `V` values

When a new tile arrives, we update:

```
m_new = max(m, max(S_ij))
scale = exp(m - m_new)
l_new = scale * l + sum(exp(S_ij - m_new))
o_new = scale * o + exp(S_ij - m_new) @ V_j
```

After processing all `K` tiles, the output for row `i` is `o_i / l_i`.

### I/O complexity
If SRAM holds `M` bytes, each tile is of size `B × d`. Choosing `B ≈ sqrt(M/d)` balances the number of `Q` tiles and `K` tiles. The total HBM traffic becomes O(n² d / M) instead of O(n²). For typical `M = 20 MB` and `d = 128`, this is a ~100× reduction in memory traffic.

### FlashAttention‑2 improvements
The original algorithm tiles over `K` for each `Q` tile, leading to repeated reads of `K` and `V`. FlashAttention‑2 tiles over `Q` instead, reusing each `K`, `V` tile across multiple `Q` tiles, further reducing I/O. It also fuses the backward pass, making training memory‑efficient as well.

## Batch Optimization

### Throughput‑latency trade‑off
Processing a batch of `b` requests together amortizes the cost of loading weights and other fixed overheads. Throughput (tokens/sec) scales roughly linearly with `b`, but latency (time to complete the batch) also grows because the longest sequence in the batch determines the compute time.

We can model this as:

```
throughput(b) = b × tokens_per_request / latency(b)
latency(b) = t_fixed + t_variable × b
```

The optimal batch size `b*` maximizes throughput while keeping latency under a service‑level objective (SLO). Finding `b*` requires profiling the actual model on the target hardware.

### Dynamic batching
Requests arrive asynchronously. A naive system would wait until `b*` requests are queued, hurting latency at low load. Dynamic batching sets two triggers:

1. **Size‑based**: form a batch when `b*` requests are waiting.
2. **Timeout‑based**: if the oldest request has waited longer than a threshold (e.g., 50 ms), form a batch with whatever is available.

The timeout ensures latency bounds even during low traffic.

### Continuous batching (iteration‑level scheduling)
In traditional dynamic batching, the whole sequence is processed as a batch; if one request finishes early, the GPU sits idle until the rest finish. Continuous batching (used in Orca, vLLM) schedules at the *token* level: each request has a slot in the batch, and when it generates an EOS token, its slot is freed and given to a waiting request.

This increases GPU utilization to >90 % for workloads with variable‑length sequences, but requires more sophisticated book‑keeping and kernel support.

### Finding the optimal batch size
The roofline model gives an upper bound: throughput ≤ min(compute_roof, memory_roof). Compute roof is `peak_FLOPs / FLOPs_per_token`. Memory roof is `memory_bandwidth / bytes_per_token`. The actual throughput curve saturates when either roof is hit.

Profiling across batch sizes (as done in `lab2_3`) shows the knee of the curve—the point where adding another request yields diminishing returns.

## How it fits together

A production serving system like **vLLM** uses all three optimizations simultaneously:

- **KV caching** with PagedAttention to pack thousands of requests into GPU memory.
- **FlashAttention** kernels to compute attention with minimal I/O, even when sequences are long and the cache is fragmented.
- **Continuous batching** to keep the GPU busy despite variable request arrival times and sequence lengths.

These techniques are not independent: FlashAttention must be aware of the paged cache layout; the batch scheduler must consider both cache occupancy and compute latency. The result is a system that can serve hundreds of concurrent requests at <100 ms latency, something impossible with naive implementations.

## Further Reading

### KV caching & PagedAttention
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) – Kwon et al., OSDI 2023.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://www.usenix.org/conference/osdi23/presentation/kwon) – extended talk slides.
- [Grouped‑Query Attention](https://arxiv.org/abs/2305.13245) – Ainslie et al., 2023.

### FlashAttention
- [FlashAttention: Fast and Memory‑Efficient Exact Attention with IO‑Awareness](https://arxiv.org/abs/2205.14135) – Dao et al., NeurIPS 2022.
- [FlashAttention‑2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) – Dao, 2023.
- [Online normalizer calculation for softmax](https://en.wikipedia.org/wiki/LogSumExp) – the log‑sum‑exp trick.

### Batch optimization
- [Orca: A Distributed Serving System for Transformer‑Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) – Yu et al., OSDI 2022.
- [Continuous Batching for LLM Inference using Sarathi‑Serve](https://arxiv.org/abs/2310.13010) – Agarwal et al., 2023.
- [The Roofline Model](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf) – Williams et al., 2009.

### Systems that combine them
- [vLLM](https://github.com/vllm-project/vllm) – open‑source implementation.
- [Hugging Face TGI](https://github.com/huggingface/text-generation-inference) – another popular serving engine.
- [NVIDIA TensorRT‑LLM](https://github.com/NVIDIA/TensorRT-LLM) – optimized kernels for NVIDIA hardware.

*All code for this week is in the three companion notebooks: `lab2_1_kv_caching.ipynb`, `lab2_2_flashattention.ipynb`, and `lab2_3_batch_optimization.ipynb`. They implement the algorithms discussed above and include benchmarks you can run on your own hardware.*