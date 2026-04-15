# Week 2 Enhancements (April 2026)

## Summary
Enhanced Week 2 materials with modern reading material, graphics, and comprehensive lessons. Labs are scaffolded for guided implementation.

## Changes

### README.md
- Added modern reading material (2024‑2025): FlashAttention‑2, PagedAttention v2, GQA, StreamingLLM, Triton.
- Added visual aids: ASCII diagrams for KV caching, comparison tables for attention variants.
- Expanded core concepts with sliding window caching, dynamic cache allocation, kernel fusion, GQA/MQA.
- Updated mathematical foundations with GQA cache formulas and FlashAttention‑2 complexity.
- Updated dependencies (FlashAttention‑2, vLLM, xFormers, Triton).
- Added hardware‑specific optimization notes (H100, MI300X, TPUv5).
- Enhanced production considerations with serving frameworks (vLLM, TensorRT‑LLM, TGI, Ray Serve).

### Lab 2.1: KV Caching Implementation
- Replaced truncated notebook with guided scaffold.
- Added TODOs for standard KV cache (`update`, `get` methods).
- Added TODOs for PagedAttention‑style cache (`allocate_page`, `free_page`, `update`, `get`).
- Included advanced caching strategies (sliding window, GQA) as optional extensions.
- Added benchmarking and visualization section.
- Preserved original implementation as `lab2_1_kv_caching_ORIGINAL.ipynb`.

### Lab 2.2: FlashAttention Implementation
- Created new scaffolded notebook.
- Guides through tiling, online softmax, and forward‑pass implementation.
- Includes benchmarking against PyTorch attention.
- Stubs for tile attention, online softmax update, and full FlashAttention forward pass.

### Lab 2.3: Batch Size Optimization
- Created new scaffolded notebook.
- Covers throughput‑vs‑batch‑size profiling, dynamic batching, and continuous batching.
- Includes stub classes for `DynamicBatcher` and `ContinuousBatcher`.
- Provides evaluation framework for comparing batching strategies.

## Next Steps
1. Review the updated README and lab scaffolds.
2. Work through the TODOs in each lab (implement missing pieces).
3. Run benchmarks and visualize results.
4. Extend with optional advanced topics.

## Notes
- All labs are designed to be **guided but not completed**; you fill in the key implementations.
- Modern references point to 2024‑2025 papers and blog posts.
- Dependencies assume a GPU environment; CPU fallbacks are provided where possible.

Enjoy Week 2!