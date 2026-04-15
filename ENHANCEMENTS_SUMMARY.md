# Enhancements Summary (April 15, 2026)

## Week 2 – Core Optimization Techniques
- **PhD‑level lecture**: `week2/README_PHD.md` with theoretical depth, recent research (2024‑2026), formal derivations, queueing‑theoretic models, hardware‑aware kernel design.
- **Enhanced main lecture**: `week2/README.md` updated with modern references and link to PhD version.
- **Scaffolded labs**: 
  - `lab2_1_kv_caching.ipynb` – standard & PagedAttention‑style KV caching
  - `lab2_2_flashattention.ipynb` – FlashAttention‑2 implementation
  - `lab2_3_batch_optimization.ipynb` – dynamic/continuous batching
- **Original material preserved** as `lab2_1_kv_caching_ORIGINAL.ipynb` and `README_ORIGINAL.md`.

## Week 3 – Advanced Sampling & Decoding
- **PhD‑level lecture**: `week3/README_PHD.md` covering speculative decoding analysis (optimal speculation length, multi‑draft, SSD), information‑theoretic sampling, adaptive decoding, recent 2025 papers.
- **Enhanced main lecture**: `week3/README.md` updated with link to PhD version.
- **Scaffolded labs**:
  - `lab3_1_speculative_decoding.ipynb` – standard & multi‑draft speculative decoding
  - `lab3_2_sampling_comparison.ipynb` – temperature, top‑k, top‑p, contrastive decoding
  - `lab3_3_advanced_decoding.ipynb` – grammar‑constrained decoding, penalties, early stopping
- **Original material preserved** as `README_ORIGINAL.md`.

## Sources Cited
- FlashAttention (NeurIPS 2022), vLLM (OSDI 2023), Orca (OSDI 2022)
- vAttention (arXiv:2405.04437), StreamingLLM (arXiv:2309.17453)
- AutoKernel (arXiv:2603.21331), MQA/GQA studies (Google 2024)
- Speculative Decoding (ICML 2023), Multi‑Draft Speculative Decoding (2024)
- Speculative Speculative Decoding (SSD, 2025), Speculative Decoding survey (arXiv:2502.19732)
- Queueing theory (Kingman’s approximation), I/O‑complexity analysis, competitive‑ratio analysis

## Git Commits
- `66a6b4d` – Week 2 enhancements
- `a410720` – Week 3 enhancements (initial)
- `dfce249` – Remove temporary script

## Next Steps
- Work through scaffolded labs (fill TODOs)
- Request enhancements for Week 4 (model compression & quantization) or other weeks
- Provide feedback for further improvements

---
*Enhanced by OpenClaw assistant*