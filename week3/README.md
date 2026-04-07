# Week 3: Advanced Sampling & Decoding

## 📋 Week Overview
This week focuses on advanced decoding techniques that improve both quality and speed of LLM inference. We'll implement speculative decoding, explore various sampling strategies, and learn how to optimize decoding for different applications.

## 🎯 Learning Objectives
By the end of this week, you will be able to:
1. Implement and optimize speculative decoding
2. Compare different sampling strategies for various tasks
3. Build adaptive decoding controllers
4. Profile and optimize decoding performance

## 📚 Core Concepts

### 1. Speculative Decoding
- **Draft-target model paradigm**: Small draft model, large target model
- **Verification and acceptance**: Token-by-token verification
- **Lookahead decoding**: Multi-token speculation
- **Adaptive speculation**: Dynamic draft model selection
- **Distillation for drafting**: Training draft models

### 2. Sampling Strategies
- **Greedy decoding**: Always choose highest probability
- **Temperature sampling**: Controlling randomness
- **Top-k sampling**: Limit to k most likely tokens
- **Top-p (nucleus) sampling**: Dynamic vocabulary size
- **Beam search**: Maintain multiple hypotheses
- **Contrastive decoding**: Compare with weaker model

### 3. Advanced Decoding Techniques
- **Typical sampling**: Avoid too predictable/unpredictable tokens
- **Mirostat sampling**: Maintain constant perplexity
- **Repetition penalties**: Avoid repetitive outputs
- **Length penalties**: Control output length
- **No-repeat n-gram**: Prevent phrase repetition

### 4. Decoding Optimization
- **Batch decoding**: Parallelize across sequences
- **Cache-aware decoding**: Optimize for KV cache
- **Early stopping**: Stop when confidence is high
- **Dynamic batching for decoding**: Handle variable lengths

## 📖 Required Reading

### Papers
1. **"Speculative Decoding"** (Leviathan et al., 2023)
   - Core speculative decoding algorithm
   - Speedup analysis and trade-offs

2. **"Fast Inference from Transformers via Speculative Decoding"** (Chen et al., 2023)
   - Improved verification algorithm
   - Multi-token speculation

3. **"The Curious Case of Neural Text Degeneration"** (Holtzman et al., 2019)
   - Analysis of sampling strategies
   - Top-p (nucleus) sampling introduction

### Blog Posts & Tutorials
1. [Speculative Decoding Explained](https://arxiv.org/abs/2302.01318)
2. [LLM Sampling Methods](https://huggingface.co/blog/how-to-generate)
3. [Contrastive Decoding](https://arxiv.org/abs/2210.15097)

## 💻 Hands-on Labs

### Lab 3.1: Speculative Decoding Implementation
```python
# Implement speculative decoding from scratch
# Compare with standard autoregressive decoding
```

**Objectives:**
- Implement draft and target model inference
- Build token verification system
- Measure speedup across different tasks
- Optimize draft model selection

### Lab 3.2: Sampling Strategy Comparison
```python
# Implement and compare different sampling strategies
# Evaluate on various NLP tasks
```

**Objectives:**
- Implement temperature, top-k, top-p sampling
- Compare output quality across strategies
- Build adaptive sampling controller
- Profile computational overhead

### Lab 3.3: Advanced Decoding Techniques
```python
# Implement contrastive decoding and other advanced methods
# Build decoding optimization framework
```

**Objectives:**
- Implement contrastive decoding
- Add repetition and length penalties
- Build early stopping mechanism
- Create decoding benchmark suite

## 🧮 Mathematical Foundations

### Speculative Decoding Analysis
**Standard autoregressive:**
- Time per token: t_target
- Total time: n × t_target

**Speculative decoding:**
- Draft time per token: t_draft (t_draft < t_target)
- Verification time: t_verify (t_verify ≈ t_target)
- Acceptance rate: α
- Expected speedup: 1 / (γ + (1-γ)/α)
- Where γ = t_draft / t_target

### Sampling Probability Distributions
**Temperature scaling:**
P_τ(x) = exp(log P(x) / τ) / Z

**Top-k sampling:**
P_k(x) = P(x) if x in top-k else 0

**Top-p (nucleus) sampling:**
Find smallest V_p where Σ_{x in V_p} P(x) ≥ p
P_p(x) = P(x) if x in V_p else 0

### Beam Search Complexity
**Standard beam search:**
- Time: O(b × n × V) for beam size b, length n, vocab V
- Memory: O(b × n) for storing beams

**Optimized beam search:**
- Pruning strategies
- Early termination
- Memory-efficient storage

## 🔬 Advanced Topics

### 1. Multi-token Speculation
- Speculate multiple tokens ahead
- Tree-based verification
- Early rejection strategies
- Optimal speculation length

### 2. Adaptive Draft Models
- Multiple draft models of different sizes
- Dynamic selection based on context
- Quality-speed trade-off optimization
- Online learning of draft effectiveness

### 3. Contrastive Decoding
- Use amateur and expert models
- Amplify differences in distributions
- Control for factuality and creativity
- Task-specific tuning

### 4. Decoding for Specific Tasks
- Code generation: Structured decoding
- Dialogue: Persona consistency
- Translation: Length normalization
- Summarization: Content coverage

## 📊 Performance Benchmarks

### Speculative Decoding Evaluation
We'll benchmark:
1. **Standard decoding**: Baseline
2. **Single-token speculation**: Basic implementation
3. **Multi-token speculation**: Advanced implementation
4. **Adaptive speculation**: Dynamic draft selection

**Metrics:**
- Speedup vs sequence length
- Quality preservation (BLEU, ROUGE)
- Memory overhead
- Draft model efficiency

### Sampling Strategy Comparison
Compare:
1. **Greedy**: Baseline
2. **Temperature sampling**: Various temperatures
3. **Top-k sampling**: Various k values
4. **Top-p sampling**: Various p values
5. **Beam search**: Various beam sizes

**Evaluation tasks:**
- Story generation (creativity)
- Code generation (correctness)
- Translation (accuracy)
- Summarization (coverage)

## 🚀 Production Considerations

### 1. Hardware-Aware Decoding
- GPU memory optimization for beam search
- Batch size optimization for sampling
- Kernel fusion for decoding operations
- Mixed precision for draft models

### 2. Quality-Speed Trade-offs
- When to use which decoding method
- Task-specific optimization
- User preference modeling
- Cost-quality optimization

### 3. Monitoring and Adaptation
- Real-time quality monitoring
- Dynamic parameter adjustment
- A/B testing framework
- User feedback integration

### 4. Security and Safety
- Content filtering during decoding
- Bias mitigation in sampling
- Adversarial attack prevention
- Privacy-preserving decoding

## 📝 Weekly Deliverables

### 1. Code Submission
- Complete all three labs
- Include benchmarking results
- Add unit tests for decoding functions

### 2. Decoding Analysis Report
- Comparison of sampling strategies
- Speculative decoding performance analysis
- Recommendations for different use cases
- Optimization suggestions

### 3. Research Implementation
- Implement one advanced decoding paper
- Compare with baseline methods
- Document implementation challenges

## 🔧 Setup Instructions

### Additional Dependencies
```bash
# Install evaluation metrics
pip install rouge-score nltk sacrebleu

# Install text generation tools
pip install transformers[torch] accelerate

# Install visualization for decoding
pip install plotly kaleido

# Install optimization libraries
pip install optuna ray[tune]
```

### Model Requirements
For this week, you'll need:
1. **Base model**: GPT-2 or similar (for experiments)
2. **Draft model**: Distilled version or smaller model
3. **Evaluation datasets**: Various NLP tasks

### Verification Script
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Testing decoding setup...")

# Load a small model for testing
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Test basic generation
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50, do_sample=True)
    generated = tokenizer.decode(outputs[0])
    
print(f"Basic generation test: {'PASS' if len(generated) > len(input_text) else 'FAIL'}")

# Test different sampling methods
sampling_methods = ['greedy', 'beam', 'sampling']
print(f"Available sampling methods: {sampling_methods}")
```

## 🎯 Success Criteria

You've successfully completed Week 3 if you can:
1. Implement and optimize speculative decoding
2. Choose appropriate sampling strategies for different tasks
3. Build adaptive decoding controllers
4. Profile and optimize decoding performance

## 📚 Additional Resources

### Videos
1. [Speculative Decoding: Making LLMs Faster](https://www.youtube.com/watch?v=example1)
2. [Advanced Sampling Techniques for LLMs](https://www.youtube.com/watch?v=example2)
3. [Optimizing Text Generation](https://www.youtube.com/watch?v=example3)

### Tools
1. [Transformers Generation](https://huggingface.co/docs/transformers/generation_strategies)
2. [vLLM Sampling](https://docs.vllm.ai/en/latest/sampling.html)
3. [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)

### Community
1. [Hugging Face Generation Forum](https://discuss.huggingface.co/c/generation/10)
2. [LLM Optimization Discord](https://discord.gg/llm-optimization)
3. [Research Paper Implementations](https://paperswithcode.com/)

---

**Estimated Time Commitment:** 16-20 hours  
**Difficulty Level:** ⭐⭐⭐⭐☆ (Advanced algorithms)  
**Next Week:** Model Compression & Quantization