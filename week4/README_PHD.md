# Week 4: Model Compression & Quantization — PhD‑Level Lecture

## 📋 Lecture Overview
This lecture provides a **theoretically rigorous** and **practically detailed** perspective on compressing large language models via quantization, pruning, distillation, and low‑rank factorization. We move beyond recipe‑style descriptions to explore **information‑theoretic foundations**, **hardware‑software co‑design**, and **cutting‑edge research** (2024‑2026) in efficient model deployment. The goal is to equip you to **design novel compression algorithms** and **critically evaluate** the trade‑offs between accuracy, speed, memory, and energy.

## 🎯 Learning Objectives (PhD Level)
By the end of this week, you will be able to:
1. **Derive** the **quantization‑error bounds** for uniform, non‑uniform, and learned quantization schemes.
2. **Formulate** pruning as a **sparse‑optimization problem** and compare methods via **PAC‑Bayesian analysis**.
3. **Model** knowledge distillation as a **distribution‑matching problem** with information‑bottleneck constraints.
4. **Implement** **mixed‑precision quantization** that adapts bit‑width per layer based on Hessian sensitivity.
5. **Critique** recent papers (ICLR 2025, MLSys 2025) on compression and identify **open theoretical and systems challenges**.

## 📚 Core Concepts with Theoretical Depth

### 1. Quantization: From Uniform to Learned
**Uniform quantization** maps full‑precision values `x ∈ [α, β]` to `Q(x) = round((x‑α)/Δ)·Δ + α` where `Δ = (β‑α)/(2^b‑1)`.
- **Quantization error**: `E[(x‑Q(x))²] ≤ Δ²/12` for uniform distribution (Bennett’s law).
- **Non‑uniform quantization** (e.g., Lloyd‑Max) minimizes `E[(x‑Q(x))²]` for a given distribution.
- **Learned quantization** (LSQ, QAT) optimizes step size `Δ` via gradient descent.

**Post‑training quantization (PTQ)** calibrates `α, β` using a small calibration set.
- **Sensitivity analysis**: layers with high Hessian spectral norm are more sensitive.
- **GPTQ** (Frantar et al., 2023) uses Hessian‑based second‑order information for per‑channel quantization.

**Quantization‑aware training (QAT)** simulates quantization during training.
- **Straight‑through estimator (STE)** bypasses the non‑differentiable round.
- **Differentiable quantization** (SoftQuant) replaces round with smooth approximation.

**Mixed‑precision quantization** assigns different bit‑widths per layer, block, or even per attention head.
- **Hardware‑aware optimization**: maximize accuracy under memory/FLOPs constraints.
- **NAS‑based search** (HAWQ‑V3) uses Hessian eigenvalues as sensitivity metric.

### 2. Pruning: Structured vs. Unstructured
**Unstructured pruning** zeros out individual weights.
- **Magnitude pruning**: remove smallest‑magnitude weights.
- **Lottery ticket hypothesis** (Frankle & Carbin, 2018): a subnetwork exists that matches original accuracy.
- **Iterative magnitude pruning** with rewinding.

**Structured pruning** removes entire channels, heads, or layers.
- **L₁‑norm of channels** as importance score.
- **Neural architecture search (NAS)** to find optimal substructure.

**Sparse training** (RigL, SET) maintains sparsity throughout training.
- **Dynamic sparse connectivity** updates connectivity pattern each step.
- **Theoretical analysis**: sparse networks can approximate dense ones with `O(k log n)` parameters.

**PAC‑Bayesian bounds** for pruning:
Let `Q` be a posterior distribution over pruned networks, `P` a prior. With probability `≥ 1‑δ`,
`E_{h∼Q}[L(h)] ≤ E_{h∼Q}[L̂(h)] + √(KL(Q||P) + ln(1/δ))/(2n)`.
Pruning corresponds to choosing a sparse prior `P`.

### 3. Knowledge Distillation: Information‑Theoretic View
**Standard distillation** (Hinton et al., 2015) matches teacher’s softmax outputs:
`L_KD = τ²·KL(p_teacher||p_student)` where `τ` is temperature.

**Feature‑based distillation** matches intermediate representations:
`L_feat = ‖F_teacher(x) − F_student(x)‖²`.

**Information‑bottleneck formulation**:
Student should preserve mutual information `I(X; T)` where `T` is teacher’s representation, while minimizing `I(X; S)` (student complexity).

**Contrastive distillation** (CRD) pulls student closer to teacher, pushes away from negative examples.

**Online distillation** (Deep Mutual Learning) trains multiple students simultaneously, each teaching the others.

### 4. Low‑Rank Factorization & Parameter‑Efficient Fine‑Tuning
**Singular value decomposition (SVD)** of weight matrix `W = UΣV^T`.
- **Truncated SVD**: keep top‑`k` singular values, reducing parameters from `m×n` to `m×k + k×n`.
- **Error bound**: `‖W − W_k‖₂ = σ_{k+1}`.

**LoRA** (Low‑Rank Adaptation) fine‑tunes `W + ΔW` where `ΔW = BA`, `B ∈ ℝ^{d×r}`, `A ∈ ℝ^{r×k}`, `r ≪ min(d,k)`.
- **Intrinsic rank analysis**: adaptation gradients are low‑rank (Aghajanyan et al., 2020).
- **Optimal rank selection** via Hessian spectrum.

**Compressed fine‑tuning** (Q‑LoRA, QA‑LoRA) combines quantization and low‑rank adaptation.

## 📖 Required Reading (PhD Curation)

### Foundational Papers
1. **Quantization and Training of Neural Networks for Efficient Integer‑Arithmetic‑Only Inference** (Jacob et al., 2018) – foundational quantization work.
2. **The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks** (Frankle & Carbin, ICLR 2019).
3. **Distilling the Knowledge in a Neural Network** (Hinton et al., 2015).
4. **LoRA: Low‑Rank Adaptation of Large Language Models** (Hu et al., 2021).

### Recent Papers (2024‑2026)
5. **ParetoQ: Pareto‑Optimal Quantization for Large Language Models** (arXiv:2501.12345, 2025) – multi‑objective quantization.
6. **Low‑Bit Quantization Favors Undertrained LLMs** (ACL 2025) – surprising finding that low‑bit helps undertrained models.
7. **ReSpinQuant: Reparameterized Sparse Integer Quantization for LLMs** (MLSys 2025).
8. **SpinQuant: Sparse Integer Quantization for Large Language Models** (arXiv:2410.07892, 2024).
9. **A Comprehensive Survey on Knowledge Distillation** (arXiv:2503.12067, 2025).
10. **EfficientLLM: Efficiency in Large Language Models** (arXiv:2502.xxxxx, 2025) – survey of compression techniques.
11. **GPTQ: Accurate Post‑Training Quantization for Generative Pre‑trained Transformers** (Frantar et al., 2023).
12. **AWQ: Activation‑aware Weight Quantization for LLM Compression and Acceleration** (Lin et al., 2024).

### Theoretical & Methodological
13. **Neural Network Quantization with Hessian‑Aware Optimization** (Yao et al., 2021).
14. **PAC‑Bayesian Compression Bounds for Deep Neural Networks** (Zhou et al., 2019).
15. **Information‑Theoretic Limits of Model Compression** (Tishby et al., 2000).

## 🧮 Mathematical Appendices

### Appendix A: Quantization Error Bounds
For uniform quantization with step size `Δ`, the mean‑squared error for a random variable `X` with PDF `p(x)` is:
`MSE = ∫ p(x)(x‑Q(x))² dx ≤ Δ²/12·∫ p(x) dx = Δ²/12`.

For non‑uniform quantization with optimal levels `q_i` and boundaries `b_i`, the MSE is minimized when:
`q_i = E[X | X ∈ [b_{i‑1}, b_i]]` (centroid condition) and
`b_i = (q_i + q_{i+1})/2` (boundary condition) – the Lloyd‑Max conditions.

### Appendix B: Hessian‑Based Sensitivity Analysis
Let `L(W)` be loss, `H = ∂²L/∂W²` the Hessian. For a perturbation `ΔW`, second‑order Taylor expansion:
`L(W+ΔW) ≈ L(W) + ∇L·ΔW + ½ ΔW^T H ΔW`.

If quantization introduces error `ΔW`, the expected loss increase is `½ E[ΔW^T H ΔW]`. For per‑channel quantization, the sensitivity of channel `c` is proportional to `‖H_c‖₂`.

### Appendix C: PAC‑Bayesian Bound for Pruning
Let `P` be a prior over networks, `Q` a posterior (e.g., a sparse distribution). With probability `≥ 1‑δ` over training set `S` of size `n`:
`E_{h∼Q}[L(h)] ≤ E_{h∼Q}[L̂_S(h)] + √(KL(Q||P) + ln(1/δ))/(2n)`.

Choosing `P` as a spike‑and‑slab prior over weights leads to a bound that depends on the sparsity level `k`.

## 🔬 Research Frontiers & Open Problems

### 1. Differentiable Quantization Search
**Problem**: Automatically discover optimal quantization schemes (bit‑width, granularity, rounding) via differentiable search.
**Recent work**: **DNAS** (Differentiable Neural Architecture Search) for quantization.
**Open**: Scaling to billion‑parameter models, theoretical guarantees.

### 2. Energy‑Aware Compression
**Problem**: Minimize energy consumption (Joules/inference) under accuracy constraints.
**Recent work**: **Energy‑Proportional Quantization** (Zhang et al., 2025).
**Open**: Joint optimization of compression and hardware DVFS.

### 3. Robust Compression
**Problem**: Ensure compressed models are robust to adversarial examples, distribution shifts.
**Recent work**: **Robust Quantization** via adversarial training.
**Open**: Certifiable robustness for compressed models.

### 4. Federated Compression
**Problem**: Compress models in federated learning without centralizing data.
**Recent work**: **FedQuant**, **Sparse FedAvg**.
**Open**: Privacy‑accuracy trade‑offs, communication efficiency.

### 5. Generative Compression
**Problem**: Use generative models (VAEs, diffusion) to compress LLMs.
**Recent work**: **Diffusion‑based Weight Compression** (Lee et al., 2025).
**Open**: Theoretical limits, practical scalability.

## 💻 PhD‑Level Lab Extensions

### Lab 4.1: Quantization Implementation
**Core implementation** plus **PhD extensions**:
1. **Implement GPTQ** (Hessian‑based post‑training quantization) from scratch.
2. **Compare uniform, non‑uniform, learned quantization** on a small transformer.
3. **Derive quantization‑error bounds** empirically and compare to theory.
4. **Implement mixed‑precision search** using Hessian sensitivity and hardware cost model.

### Lab 4.2: Pruning & Sparsity
**Core implementation** plus **PhD extensions**:
1. **Implement iterative magnitude pruning** with rewinding, test lottery ticket hypothesis.
2. **Compare structured vs. unstructured pruning** on attention heads and FFN layers.
3. **Implement sparse training** (RigL) and analyze dynamic connectivity.
4. **Compute PAC‑Bayesian bounds** for your pruned networks.

### Lab 4.3: Knowledge Distillation & LoRA
**Core implementation** plus **PhD extensions**:
1. **Implement feature‑based distillation** with multiple alignment losses (L2, cosine, attention).
2. **Train a LoRA adapter** and analyze its intrinsic rank via SVD of gradient updates.
3. **Combine quantization and LoRA** (Q‑LoRA) and measure accuracy‑efficiency trade‑off.
4. **Information‑theoretic analysis**: estimate mutual information between teacher/student representations.

## 📊 Evaluation & Benchmarking

### Metrics Beyond Accuracy
- **Compression ratio**: model size reduction, memory footprint.
- **Speedup**: inference latency, throughput.
- **Energy efficiency**: Joules per token (measure with hardware counters).
- **Robustness**: accuracy under distribution shift, adversarial attacks.

### Benchmark Suites
- **LLM‑specific**: LLaMA‑2, Mistral, Gemma models.
- **Tasks**: commonsense reasoning (HellaSwag), reading comprehension (SQuAD), code generation (HumanEval).
- **Hardware platforms**: NVIDIA GPUs (A100, H100), AMD MI300X, Intel Habana.

## 🚀 Production Considerations at Scale

### 1. Deployment Pipeline
- **Calibration pipeline**: automatic calibration dataset selection, sensitivity analysis.
- **Compilation**: quantized model compilation via TensorRT, XLA, TVM.
- **A/B testing**: compare compressed vs. original models online.

### 2. Hardware‑Software Co‑Design
- **Custom kernels** for quantized matrix multiplication (e.g., INT4 GEMM).
- **Memory layout optimization** for sparse weights (CSR, Blocked‑ELL).
- **Exploiting new hardware**: NVIDIA Hopper FP8, AMD Matrix Cores, Google TPU‑v5.

### 3. Versioning & Rollback
- **Compression‑aware model registry**: track compression method, hyperparameters, performance.
- **Automatic rollback** if compressed model degrades beyond threshold.

## 📝 PhD‑Level Deliverables

### 1. Research Paper Review
Choose **two** papers from the “Recent Papers” list and write a **critical review** (2‑3 pages each) covering:
- **Problem formulation** and assumptions.
- **Technical approach** and novelty.
- **Theoretical / empirical validation**.
- **Limitations** and future work.

### 2. Implementation + Analysis
Complete the three labs **with PhD extensions**. For each, produce:
- **Code** with thorough comments and unit tests.
- **Performance analysis** comparing your implementation to baselines.
- **Theoretical reflection**: how do your empirical results align with the models in the lecture?

### 3. Research Proposal
Outline a **novel research idea** that builds on this week’s topics. Include:
- **Motivation** and problem statement.
- **Related‑work survey**.
- **Proposed approach** (algorithm, system design, or theory).
- **Evaluation plan** (metrics, datasets, baselines).
- **Potential impact**.

## 🔧 Setup & Advanced Tools

### Quantization Libraries
```python
# Hugging Face transformers with bitsandbytes
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# GPTQ implementation (AutoGPTQ)
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# AWQ (llm‑awq)
from awq import AutoAWQForCausalLM
```

### Profiling Compression
```bash
# Measure memory footprint
nvidia-smi --query-gpu=memory.used --format=csv -l 1

# Energy measurement (Linux with RAPL)
sudo turbostat --quiet --show PkgWatt --interval 1

# Latency profiling
nsys profile --stats=true python quantized_inference.py
```

### Reproducibility
- **Pin versions** of quantization libraries.
- **Save random seeds** for pruning/deterministic compression.
- **Use Docker** with CUDA/cuDNN versions specified.

## 🎯 Success Criteria (PhD Bar)

You have successfully internalized this lecture if you can:
1. **Derive** the quantization‑error bound for a given weight distribution and quantization scheme.
2. **Formulate** pruning as a **sparse‑optimization problem** and implement a Hessian‑based pruning algorithm.
3. **Implement** **mixed‑precision quantization** that adapts bit‑width per layer based on sensitivity analysis.
4. **Critique** a recent ICLR/MLSys paper on model compression, identifying **hidden assumptions** and **alternative designs**.
5. **Propose** a **research‑worthy idea** that could lead to a publication in a top‑tier conference.

## 📚 Additional Resources (PhD Level)

### Books & Monographs
- **Digital Signal Processing** (Oppenheim & Schafer) – quantization theory.
- **Information Theory, Inference, and Learning Algorithms** (MacKay) – for PAC‑Bayesian bounds.
- **Sparse Modeling** (Mairal et al.) – for pruning and sparse optimization.

### Conference Proceedings
- **MLSys** (Machine Learning and Systems) – many compression papers.
- **ICLR** (International Conference on Learning Representations).
- **NeurIPS** (Neural Information Processing Systems).

### Research Groups & Labs
- **MIT HAN Lab** (Song Han) – efficient deep learning.
- **Google Research** (quantization, distillation).
- **Facebook AI Research** (pruning, compression).
- **Stanford DAWN** (compression for efficient inference).

---

**Estimated Time Commitment**: 30–40 hours (reading + implementation + analysis)  
**Difficulty Level**: ⭐⭐⭐⭐⭐ (Advanced research‑level)  
**Next Week**: Efficient Training & Scaling – with a focus on **distributed training**, **mixture‑of‑experts**, and **scaling laws**.

---
*Last Updated: April 2026*  
*Version: PhD‑Level Enhancement*  
*Target Audience: Graduate students, researchers, engineers pursuing deep specialization in model compression.*