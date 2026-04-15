# Week 3: Advanced Sampling & Decoding — PhD‑Level Lecture

## 📋 Lecture Overview
This lecture provides a **theoretically rigorous** and **systems‑oriented** perspective on advanced decoding techniques for LLMs. We move beyond heuristic descriptions to explore **information‑theoretic foundations**, **optimization frameworks**, and **cutting‑edge research** (2024‑2026) in speculative decoding, sampling strategies, and adaptive generation. The goal is to equip you to **design novel decoding algorithms** and **critically analyze** the trade‑offs between quality, speed, and computational cost.

## 🎯 Learning Objectives (PhD Level)
By the end of this week, you will be able to:
1. **Derive** the **optimal speculation length** for speculative decoding under a given draft‑target model pair and hardware profile.
2. **Formulate** sampling as a **stochastic optimal‑control problem** and compare strategies via **regret analysis**.
3. **Model** adaptive decoding as a **contextual bandit** or **reinforcement‑learning** problem with latency constraints.
4. **Implement** **speculative speculative decoding** (SSD) that parallelizes speculation and verification.
5. **Critique** recent papers (ICLR 2025, NAACL 2025) on decoding and identify **open theoretical and systems challenges**.

## 📚 Core Concepts with Theoretical Depth

### 1. Speculative Decoding: A Formal Analysis
**Standard speculative decoding** (Leviathan et al., 2023) is a **two‑stage stochastic process**:
- **Draft stage**: Small model `M_d` generates a candidate sequence `y_1:m` autoregressively.
- **Verification stage**: Large model `M_t` computes token‑wise probabilities `p_t(y_i | x, y_<i)`; accept prefix where `p_t ≥ p_d`.
- **Acceptance rate** `α = E[length of accepted prefix] / m`.

**Speedup analysis**:
Let `t_d`, `t_v` be per‑token times for draft and verification (typically `t_d ≪ t_v`).  
Expected time per output token:

`E[time] = (t_d + t_v) / (α + (1‑α)·m)`

Optimal speculation length `m*` solves `∂E[time]/∂m = 0`. For i.i.d. acceptance with probability `p`, `m* ≈ 1/(1‑p)`.

**Multi‑draft speculative decoding** (MDSD, 2024) generates `k` independent draft trajectories, verified in parallel.  
- **Speedup** grows with `√k` under ideal independence.
- **Memory overhead** `O(k·m)`.

**Speculative speculative decoding** (SSD, 2025) pipelines speculation and verification: while verifying draft `i`, speculate draft `i+1`.  
- **Theoretical limit**: eliminates sequential dependency, approaching speedup `t_v/t_d`.

### 2. Sampling Strategies: An Information‑Theoretic View
Sampling from a language model’s output distribution is a **tension between entropy and quality**.
- **Greedy decoding** (argmax) minimizes **cross‑entropy** but leads to **degenerate, repetitive text** (Holtzman et al., 2019).
- **Temperature scaling** `p_τ(x) ∝ p(x)^{1/τ}` adjusts the **Shannon entropy** of the output distribution.
- **Top‑k and top‑p (nucleus) sampling** truncate the distribution, controlling the **effective vocabulary size**.

**Quantitative metrics**:
- **Per‑token surprisal** `−log p(x_t | context)`.
- **Expected surprise** of a sampling strategy: `E_{x∼q}[−log p(x)]` where `q` is the induced distribution.
- **Divergence** between `q` and `p` (KL, total variation).

**Contrastive decoding** (Li et al., 2022) amplifies differences between an **expert** model `p_e` and an **amateur** model `p_a`:
`p_CD(x) ∝ max(0, log p_e(x) − log p_a(x))`.
- **Theoretical justification**: removes “common‑mode” errors shared by both models.
- **Calibration**: need to ensure `p_a` is strictly weaker but not pathological.

### 3. Adaptive Decoding as Online Learning
Decoding parameters (temperature, top‑p, speculation length) can be tuned **online** based on observed outcomes.
- **Contextual bandit formulation**: context = partial sequence, action = decoding parameters, reward = quality metric (e.g., user feedback).
- **Regret minimization**: compete with best fixed parameters in hindsight.
- **Latency constraints**: incorporate time penalty in reward.

**Model‑based optimization**:
- Use a **proxy model** (e.g., distilled version) to predict quality/speed trade‑offs.
- **Bayesian optimization** over decoding hyperparameters.

### 4. Decoding for Specific Tasks: Structural Constraints
**Code generation**:
- **Grammar‑constrained decoding**: ensure output conforms to language grammar (e.g., Python syntax).
- **Incremental parsing** to reject invalid tokens early.

**Dialogue**:
- **Persona consistency** via **KL‑divergence regularization** against a persona distribution.
- **Turn‑taking models** for multi‑party dialogue.

**Translation**:
- **Length‑normalized beam search** to compare hypotheses of different lengths.
- **Lexical constraints** (must‑include phrases).

## 📖 Required Reading (PhD Curation)

### Foundational Papers
1. **Speculative Decoding** (Leviathan et al., ICML 2023) – the original algorithm.
2. **Fast Inference from Transformers via Speculative Decoding** (Chen et al., 2023) – improved verification.
3. **The Curious Case of Neural Text Degeneration** (Holtzman et al., 2019) – analysis of greedy decoding and introduction of nucleus sampling.
4. **Contrastive Decoding** (Li et al., 2022) – expert‑amateur framework.

### Advanced & Recent Papers (2024‑2026)
5. **Multi‑Draft Speculative Decoding** (Miao et al., 2024) – parallel draft trajectories.
6. **Speculative Speculative Decoding** (SSD) (OpenReview 2025) – pipelining speculation and verification.
7. **Speculative Decoding and Beyond: An In‑Depth Survey** (Hu et al., arXiv:2502.19732, 2025) – comprehensive survey.
8. **Decoding Speculative Decoding** (NAACL 2025) – analysis of draft‑model selection.
9. **Adaptive Speculative Decoding** (Liu et al., 2024) – dynamic draft‑model switching.
10. **Energy‑Aware Decoding** (Zhang et al., 2025) – trading off quality against energy consumption.

### Theoretical & Methodological
11. **An Information‑Theoretic Perspective on Sampling** (Meister et al., 2023) – entropy‑based analysis.
12. **Stochastic Beams and Where to Find Them** (Kool et al., 2020) – beam search as stochastic optimization.
13. **Online Learning for Adaptive Text Generation** (Chowdhery et al., 2022) – bandit formulations.

## 🧮 Mathematical Appendices

### Appendix A: Derivation of Optimal Speculation Length
Assume each draft token is accepted independently with probability `p`. Let `m` be speculation length, `t_d` draft time per token, `t_v` verification time per token. Expected time per output token:

`T(m) = (m·t_d + t_v) / (m·p + 1‑p)`

Differentiate w.r.t. `m`:

`dT/dm = [t_d(m p + 1‑p) − p(m t_d + t_v)] / (m p + 1‑p)²`

Set numerator to zero:

`t_d(m p + 1‑p) = p(m t_d + t_v)`
→ `t_d(1‑p) = p t_v`
→ `p = t_d / (t_d + t_v)`

Thus optimal `p` depends on time ratio. If acceptance probability `p` is fixed, optimal `m` is infinite (verification dominates). In practice, `p` decays with `m`; need to model `p(m)`.

### Appendix B: Information‑Theoretic Bounds on Sampling
For a language model distribution `p`, any sampling strategy `q` (possibly adaptive) produces text with average surprisal `H(q) = E_{x∼q}[−log p(x)]`. The **entropy gap** `H(q) − H(p)` measures how much more “surprising” the sampled text is relative to the model’s true distribution.

**Proposition**: For temperature sampling with `τ > 1`, `H(q_τ) ≥ H(p)`; for `τ < 1`, `H(q_τ) ≤ H(p)`.

**Proof**: Temperature scaling changes density to `p_τ(x) ∝ p(x)^{1/τ}`. The induced distribution `q_τ` is `p_τ` re‑normalized. The relation follows from convexity of `log`.

### Appendix C: Regret Bound for Adaptive Decoding Bandit
Consider a contextual bandit with context space `C`, action space `A` (decoding parameters), bounded reward `r(c,a) ∈ [0,1]`. Let `π*` be the optimal context‑dependent policy. A **UCB‑based algorithm** (e.g., LinUCB) achieves regret:

`Regret(T) = O(√(d T log T))`

where `d` is dimension of context feature map, `T` number of rounds. Applying this to adaptive decoding requires designing a feature map for partial sequences.

## 🔬 Research Frontiers & Open Problems

### 1. Speculation Beyond Autoregressive Drafts
**Non‑autoregressive draft models** that generate multiple tokens in parallel (e.g., using masked language modeling).  
**Open**: How to verify such drafts efficiently?  
**Recent work**: **Blockwise parallel decoding** (Stern et al., 2018).

### 2. Decoding with Uncertainty Quantification
**Bayesian decoding** that maintains a distribution over possible continuations, not just a single sequence.  
**Open**: Efficient approximation of posterior over long sequences.  
**Related**: **Monte‑Carlo tree search** for text generation.

### 3. Multimodal Decoding
**Generating interleaved text, images, audio** from a single model.  
**Open**: Scheduling and resource allocation across modalities.  
**Recent work**: **Unified‑modal decoding** (Liu et al., 2024).

### 4. Formal Guarantees for Decoding
**Provable bounds** on output quality (e.g., closeness to true distribution) for specific decoding algorithms.  
**Open**: Trade‑offs between approximation error and computational cost.  
**Related**: **PAC‑learning for text generation**.

### 5. Energy‑Efficient Decoding
**Minimizing Joules per token** while meeting quality targets.  
**Open**: Joint optimization of decoding algorithm and hardware DVFS.  
**Recent work**: **Green decoding** (Zhang et al., 2025).

## 💻 PhD‑Level Lab Extensions

### Lab 3.1: Speculative Decoding Implementation
**Core implementation** plus **PhD extensions**:
1. **Implement MDSD** (multi‑draft speculative decoding) and compare speedup vs. `k`.
2. **Model acceptance probability `p(m)`** empirically; fit a decay function (exponential, power‑law).
3. **Search for optimal `m`** using a proxy model (distilled version) to avoid costly evaluation.
4. **Implement SSD** (speculative speculative decoding) using CUDA streams or async execution.

### Lab 3.2: Sampling Strategy Comparison
**Core implementation** plus **PhD extensions**:
1. **Measure entropy gap** `H(q) − H(p)` for each sampling strategy across different temperatures.
2. **Implement contrastive decoding** with several amateur models (distilled, n‑gram, random).
3. **Bandit‑based adaptive sampling**: tune temperature online using UCB, track regret.
4. **Formal verification**: prove that your sampling implementation indeed draws from the correct distribution (use statistical tests, e.g., **two‑sample Kolmogorov–Smirnov**).

### Lab 3.3: Advanced Decoding Techniques
**Core implementation** plus **PhD extensions**:
1. **Grammar‑constrained decoding** for Python code generation; integrate a parser (e.g., tree‑sitter).
2. **Implement Bayesian decoding** via **particle filtering** (maintain multiple hypotheses with weights).
3. **Energy‑aware decoding**: extend reward to include estimated energy cost; solve constrained optimization.
4. **Theoretical analysis**: derive a lower bound on the time‑per‑token for any decoding algorithm that accesses the model as a black‑box oracle.

## 📊 Evaluation & Benchmarking

### Metrics Beyond BLEU/ROUGE
- **Distributional metrics**: **MAUVE** (Pillutla et al., 2021), **BERT‑score**, **Divergence scores** (KL, JS).
- **Latency distributions**: p50, p95, p99 per‑token latency under load.
- **Energy measurements**: use `nvidia‑smi` or `RAPL` to estimate Joules per token.

### Benchmark Suites
- **Human‑evaluated quality** (e.g., **Chatbot Arena** style pairwise comparisons).
- **Task‑specific benchmarks**: **HumanEval** for code, **WMT** for translation, **CNN/DailyMail** for summarization.
- **Stress tests**: very long sequences, out‑of‑domain inputs, adversarial prompts.

## 🚀 Production Considerations at Scale

### 1. Distributed Speculative Decoding
- **Pipeline parallelism**: draft model on one GPU, target model on another.
- **Model parallelism** within the target model (tensor‑parallel, pipeline‑parallel).
- **Communication overhead** between draft and target.

### 2. Quality‑of‑Service (QoS) for Decoding
- **Priority scheduling**: high‑priority requests get more resources (larger beam, more speculation).
- **Preemption**: interrupt low‑priority decoding to serve high‑priority request.
- **SLO compliance**: guarantee p95 latency < threshold.

### 3. Monitoring and Debugging Decoding
- **Visualization tools**: show acceptance/rejection of speculative tokens, temperature adjustments over time.
- **Anomaly detection**: sudden drop in acceptance rate may indicate distribution shift.
- **A/B testing framework**: compare decoding algorithms online.

## 📝 PhD‑Level Deliverables

### 1. Research Paper Review
Choose **two** papers from the “Advanced & Recent” list and write a **critical review** (2‑3 pages each) covering:
- **Problem formulation** and assumptions.
- **Technical approach** and novelty.
- **Theoretical / empirical validation**.
- **Limitations** and future work.

### 2. Implementation + Analysis
Complete the three labs **with PhD extensions**. For each, produce:
- **Code** with thorough comments and unit tests.
- **Performance analysis** comparing your implementation to baselines (standard decoding, Hugging Face generation).
- **Theoretical reflection**: how do your empirical results align with the models in the lecture?

### 3. Research Proposal
Outline a **novel research idea** that builds on this week’s topics. Include:
- **Motivation** and problem statement.
- **Related‑work survey**.
- **Proposed approach** (algorithm, system design, or theory).
- **Evaluation plan** (metrics, datasets, baselines).
- **Potential impact**.

## 🔧 Setup & Advanced Tools

### Profiling Decoding Performance
```bash
# NVIDIA Nsight Systems for timeline of draft/verification phases
nsys profile --trace=cuda,nvtx python speculative_decoding.py

# Energy measurement with nvidia‑smi (Linux)
nvidia-smi dmon -s puc -i 0 -d 1 -o TD

# Python profiling with py‑instrument
pip install pyinstrument
python -m pyinstrument speculative_decoding.py
```

### Implementing Custom CUDA Kernels for Decoding
- **Fused sampling kernel**: combine softmax, random number generation, token selection.
- **Batch‑wise verification kernel**: verify multiple draft sequences in parallel.
- **Use Triton** for rapid prototyping.

### Reproducibility
- **Pin versions** of transformers, CUDA, etc.
- **Record random seeds** for sampling.
- **Use Docker** with GPU support for consistent environment.

## 🎯 Success Criteria (PhD Bar)

You have successfully internalized this lecture if you can:
1. **Derive** the optimal speculation length for a given draft‑target pair and hardware profile.
2. **Formulate** adaptive decoding as a **contextual bandit** and implement a UCB‑based algorithm.
3. **Implement** **speculative speculative decoding** (SSD) that pipelines speculation and verification.
4. **Critique** a recent ICLR/NAACL paper on decoding, identifying **hidden assumptions** and **alternative designs**.
5. **Propose** a **research‑worthy idea** that could lead to a publication in a top‑tier conference.

## 📚 Additional Resources (PhD Level)

### Books & Monographs
- **Reinforcement Learning: An Introduction** (Sutton & Barto) – for bandit formulations.
- **Information Theory, Inference, and Learning Algorithms** (MacKay) – for entropy‑based analysis.
- **Stochastic Processes and Calculus** (Klenke) – for modeling acceptance as a stochastic process.

### Conference Proceedings
- **ICLR** (International Conference on Learning Representations) – many decoding papers.
- **NAACL/ACL** (Natural Language Processing conferences).
- **MLSys** (systems‑oriented machine learning).

### Research Groups & Labs
- **Stanford CRFM** (Holtzman, Liang) – sampling and decoding.
- **Google Brain** (Chen, Leviathan) – speculative decoding.
- **FAIR** (Meta AI) – multi‑modal generation.
- **UC Berkeley** – adaptive systems.

---

**Estimated Time Commitment**: 30–40 hours (reading + implementation + analysis)  
**Difficulty Level**: ⭐⭐⭐⭐⭐ (Advanced research‑level)  
**Next Week**: Model Compression & Quantization — with a focus on **information‑theoretic** and **hardware‑aware** compression.

---
*Last Updated: April 2026*  
*Version: PhD‑Level Enhancement*  
*Target Audience: Graduate students, researchers, engineers pursuing deep specialization in decoding and generation.*