# Week 3: Advanced Sampling & Decoding

## Motivation

### The degeneration problem

Greedy decoding — picking the single most probable token at each step — produces remarkably bad text. Given a prompt like *"The meaning of life is"*, a greedy model will often spiral into repetitious loops: *"to live a life that is meaningful and to live a life that is full of meaning and to live a life…"*. This is not an implementation bug; it is a mathematical inevitability.

Let $p(x_t \mid x_{1:t-1})$ be the model's distribution over the next token given the prefix. Greedy decoding selects $\arg\max_x p(x_t = x \mid x_{1:t-1})$. The problem is that the most probable continuation after a few tokens is almost always a **high-frequency n-gram** — a phrase the model has seen thousands of times in training. The greedy path converges to the mode of the conditional distribution at every step, and the mode is usually boring.

Beam search is not a fix. Maintaining $b$ hypotheses delays the collapse but does not prevent it; all $b$ beams eventually converge to the same high-probability region. On open-ended generation tasks (storytelling, dialogue), beam search actually scores *worse* than random sampling under human evaluation (Holtzman et al., 2019).

### The typical set insight

The correct framing comes from information theory. A model's distribution over sequences of length $n$ assigns a probability $p(x_{1:n})$ to each sequence. There is a set of sequences whose total probability mass is close to 1 — call this the **typical set**. Its size is roughly $2^{n H(p)}$, where $H(p)$ is the per-token entropy.

Crucially, the *most probable* sequence (the greedy/beam path) is not in the typical set. The individual tokens in the greedy path each have high probability, but the *product* of many high-probability tokens is astronomically small compared to the total probability mass of the entire typical set. Human-written text lives in the typical set — it uses tokens that are individually plausible but collectively surprising enough to carry information.

In other words:
- **Too probable** → repetitive, generic, dull ("to live a life that is…")
- **Too improbable** → incoherent, off-topic, gibberish
- **Just right** → tokens whose log-probability is close to $-H(p)$ — the "band of moderate surprise"

Every sampling method in this lecture is an attempt to stay inside that band.

---

## The Sampling Zoo: a progression, not a list

### Greedy → repetition

Already covered. Greedy picks $\arg\max$ at each step and almost always hits the repetition wall.

### Temperature → looseness

Temperature $T$ scales the logits before the softmax:

$$p_T(x) = \frac{\exp(\log p(x) / T)}{\sum_{x'} \exp(\log p(x') / T)}$$

When $T > 1$, the distribution spreads out; lower-probability tokens become more likely. When $T < 1$, the distribution sharpens. This can break repetition loops, but it introduces a new problem: $T$ is a global knob, and the optimal value differs across contexts. A $T$ that works for a factual prompt ("List the planets") will be too hot for a creative prompt ("Write a poem"). Temperature alone cannot adapt to local distributional changes.

### Top-k → brittle

Top-k sampling restricts the vocabulary to the $k$ most probable tokens before sampling. Let $V_k$ be the set of top $k$ tokens by probability. Then:

$$p_k(x) = \frac{p(x)}{\sum_{x' \in V_k} p(x')} \quad \text{if } x \in V_k, \text{ else } 0$$

This removes the long tail of unlikely tokens, which is good, but $k$ is fixed. When the distribution is **sharp** (one clear choice, e.g., a factual answer), top-k needlessly includes many plausible-looking alternatives that dilute the probability mass. When the distribution is **flat** (many equally plausible continuations), top-k excludes reasonable candidates that fall outside the arbitrary cutoff. No single $k$ works across both regimes.

### Top-p (nucleus) → adaptive

Top-p solves top-k's brittleness by cutting off the tail dynamically. Instead of fixing the *number* of tokens, fix the *total probability mass* $p$ (typically 0.9–0.95). Select the smallest set $V_p$ such that:

$$\sum_{x \in V_p} p(x) \ge p$$

Then renormalize over $V_p$ and sample. When the distribution is sharp, $V_p$ is small (just a few tokens). When it is flat, $V_p$ grows automatically. The cutoff adapts to the entropy of the current step, which is exactly what a fixed $k$ cannot do.

Top-p is the dominant sampling method in production systems, typically combined with a small temperature and sometimes with top-k as an additional safety cap. But it still has a subtle flaw: it always picks from the *top* of the distribution. The typical set can include tokens with moderate probability that top-p would drop in a sharp distribution. The next methods address this directly.

### Typical sampling → the surprisal argument

Typical sampling (Meister et al., 2023) rejects any token whose *surprisal* $-\log p(x_t)$ is too far from the expected surprisal $H(p)$. Compute the entropy $H(p) = -\sum_x p(x) \log p(x)$, then keep only tokens whose surprisal falls within a window $\epsilon$ of $H(p)$:

$$V_{\text{typical}} = \\{ x : \left| -\log p(x) - H(p) \right| \le \epsilon \\}$$

Renormalize $p$ over $V_{\text{typical}}$ and sample. This directly implements the typical set intuition: discard tokens that are too predictable (surprisal too low, the greedy trap) *and* too unpredictable (surprisal too high, the incoherence trap). The result is text that stays in the "band of moderate surprise."

Typical sampling works well but requires an entropy computation at every step, adding overhead. It also has no mechanism to *control* the overall surprise level over long generations — it just enforces that each individual step is "typical."

### Mirostat → perplexity as a control target

Mirostat (Basu et al., 2020) reframes sampling as a **feedback control** problem. Instead of adjusting the vocabulary cutoff per step with a fixed rule, it maintains a target surprise rate $\tau$ (equivalently, a target perplexity $2^\tau$) and drives observed surprisal toward that target using a running control variable $\mu$.

The algorithm at each step:

1. Sort tokens by probability. Find the smallest set $V_\mu$ whose cumulative "surprise budget" matches $\mu$ (roughly: the nucleus size that would produce surprisal $\mu$).
2. Sample a token $x$ from $V_\mu$ and observe its surprisal $s = -\log p(x)$.
3. Update the control variable using the feedback error:

$$\mu \leftarrow \mu - \eta \cdot (s - \tau)$$

where $\eta$ is a learning rate. If the observed surprisal $s$ exceeded the target $\tau$ (too surprising), $\mu$ decreases, shrinking the nucleus next step. If $s < \tau$ (too predictable), $\mu$ grows, widening the nucleus.

This turns generation into a closed-loop controller: you set $\tau$ (e.g., 5.0 for creative, 3.0 for factual), and Mirostat regulates the nucleus size step by step to hit that target. The result is stable long-form text that does not drift into repetition *or* incoherence.

### Contrastive decoding → amplify the gap

A completely different family: instead of sampling from a single model, **contrastive decoding** uses two models — a large "expert" $p_{\text{exp}}(x)$ and a small "amateur" $p_{\text{ama}}(x)$ — and amplifies the differences between their distributions. The idea is that tokens the amateur also finds likely are probably generic n-gram matches; tokens the expert favors over the amateur carry the expert's special knowledge.

The algorithm (Li et al., 2022):

1. **Plausibility constraint** — discard any token where the expert assigns vanishing probability:

$$V_{\text{head}} = \\{ x : p_{\text{exp}}(x) \ge \beta \cdot \max_{x'} p_{\text{exp}}(x') \\}$$

Typical $\beta = 0.01$–$0.1$. This prevents the expert from falling back on its distribution's tail, which would amplify noise rather than knowledge.

2. **Score by log-ratio** — within $V_{\text{head}}$, compute:

$$s(x) = \log p_{\text{exp}}(x) - \log p_{\text{ama}}(x)$$

Then sample from $\text{softmax}(s(x) / T)$.

The plausibility constraint is critical. Without it, a token where $p_{\text{exp}}(x)$ is very small but $p_{\text{ama}}(x)$ is even smaller would get a high score, dominating the distribution. The constraint ensures only high-expert-support tokens compete.

Contrastive decoding produces measurably higher-quality text on both automatic and human evaluation. Its main cost is running a second (smaller) model, which adds latency but is often acceptable since the amateur can be significantly smaller than the expert.

---

## Speculative Decoding — the full argument

### The bottleneck

Autoregressive generation is intrinsically serial: token $t+1$ depends on all previous tokens. On a GPU, each step loads the full model weights from HBM and produces a single token. For a 7B-parameter model, a single forward pass takes ~5 ms on an A100. Generating a 256-token completion therefore takes ~1.3 s — dominated by memory-bound weight loads, not compute.

### The key insight

A smaller "draft" model (e.g., 1B parameters) generates tokens much faster — say ~1 ms per step. The question: can the small model's output be *verified* by the large model in a way that allows generating multiple tokens per large-model call?

Yes. The large model can process a sequence of $K$ candidate tokens in a **single forward pass** — nearly the same cost as generating one token. This is a property of transformer architectures: computing $K$ consecutive logit vectors from a prompt costs roughly the same as computing one, because the forward pass is memory-bound by weight loads and attention over $K$ new tokens is trivially parallel.

### The acceptance criterion

Let $p(x)$ be the **target** (large) model's distribution and $q(x)$ the **draft** (small) model's distribution at a given position. The draft model samples a candidate token $x \sim q$. The target model computes $p(x)$ for the same token. Accept the draft token with probability:

$$\min\\!\left(1, \frac{p(x)}{q(x)}\right)$$

Intuitively: if the draft was *overconfident* relative to the target ($q(x) > p(x)$), the draft may have picked a token the target actually dislikes — we reject with probability $1 - p(x)/q(x)$. If the draft was *underconfident* or agreed with the target ($q(x) \le p(x)$), the ratio is $\ge 1$ and we always accept.

When a token is **rejected**, we resample from the adjusted distribution:

$$\tilde{p}(x) = \frac{(p(x) - q(x))_+}{Z}, \quad Z = \sum_{x'} (p(x') - q(x'))_+$$

where $(a)_+ = \max(0, a)$. This targeted resampling "fills in" exactly the mass where the target preferred tokens that the draft under-weighted.

The crucial property: the marginal distribution of *finalized* tokens (accepted or resampled) exactly equals $p(x)$. That is, speculative decoding is **lossless in expectation** — the output distribution is identical to sampling directly from the target model, even though we used the draft to propose. The proof is a one-line calculation (Leviathan et al., 2023): the total acceptance probability is $\sum_x \min(p(x), q(x))$, and the leftover mass $p(x) - \min(p(x), q(x)) = (p(x) - q(x))_+$ is exactly what resampling restores.

This mathematical elegance — that an approximate draft model can accelerate exact inference from the target — is the core achievement of speculative decoding.

### Speedup formula

Let $\alpha$ be the token-wise acceptance probability, $\gamma$ be the number of draft tokens generated per target call, $t_{\text{draft}}$ be the per-token draft time, and $t_{\text{target}}$ be the per-forward-pass target time (which is also the per-token time in standard decoding).

The expected number of tokens produced per target forward pass is:

$$\mathbb{E}[\text{tokens}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

Derivation: starting with $\gamma$ draft tokens, we accept each with probability $\alpha$, and the final token (whether a freshly resampled one after rejection, or the target's own prediction after all $\gamma$ were accepted) is always produced. This is a truncated geometric series.

The speedup relative to standard decoding is:

$$\text{speedup} = \frac{\mathbb{E}[\text{tokens}] \cdot t_{\text{target}}}{\gamma \cdot t_{\text{draft}} + t_{\text{target}}}$$

Two regimes:
- **Draft is fast** ($t_{\text{draft}} \ll t_{\text{target}}$, e.g., a 100× smaller model): speedup approaches $\mathbb{E}[\text{tokens}]$.
- **Draft quality is high** ($\alpha \to 1$): $\mathbb{E}[\text{tokens}] \approx \gamma + 1$, so speedup scales with speculation length.

In practice, $\gamma = 4$–$8$ and $\alpha \approx 0.7$–$0.9$ for well-matched draft models, yielding 2–3× speedup.

### Tree-based verification

The draft model can generate a *tree* of candidates rather than a single chain. A small draft produces its top-$k$ tokens at the first step, then top-$k$ for each of those, forming a tree of depth $\gamma$ and branching factor $k$. The target model processes all $\sum_{t=0}^{\gamma-1} k^t$ candidates simultaneously using a tree attention mask (Cai et al., 2024). Verified paths are accepted greedily or by sampling. This increases the probability of finding an acceptable continuation, improving effective $\alpha$ without increasing $\gamma$.

### Self-speculation (Medusa / EAGLE)

Running a separate draft model has real costs: loading its parameters from HBM, duplicating KV caches, and ensuring its tokenizer matches the target. **Self-speculation** eliminates the draft model entirely by adding lightweight "draft heads" — small MLP modules — on top of the target model itself. These heads predict the next $\gamma$ tokens in parallel, conditioned on the target model's hidden states.

- **Medusa** (Cai et al., 2024): adds $\gamma$ independent draft heads, each predicting one future position. At verification time, the target model checks the joint likelihood of the entire candidate tree.
- **EAGLE** (Li et al., 2024): improves on Medusa by adding a small auto-regressive drafter that consumes the target model's feature maps, capturing inter-token dependencies that independent heads miss.

Because the draft heads share the target's base layers, the KV cache is shared between drafting and verification — no duplication. The memory overhead is a few million parameters (vs. billions for a separate draft model). Self-speculation is now the dominant production approach (vLLM, TensorRT-LLM) because it delivers speedup with minimal engineering cost.

---

## Connecting it to Week 2

Speculative decoding interacts non-trivially with KV caching. The draft model generates candidate tokens using its own KV cache, which is small (fewer layers, fewer heads). When the target model verifies, it must compute KV pairs for the candidate tokens in its own cache.

The complication arises on **rejection**. If the target model rejects draft tokens at position $i$, the KV cache entries for positions $i$ through $\gamma$ must be discarded — they correspond to tokens that will not appear in the final output. Systems like vLLM implement this with a **cache rollback**: a pointer reset to the last confirmed prefix.

This is straightforward for a single sequence but subtle in a batch: different sequences may accept different prefix lengths each step, so each slot needs its own rollback pointer. Tree-based verification adds another layer, since multiple branches share different subsets of the prefix. The implementation stores a **candidate mask** alongside the KV cache — bits indicating which positions belong to the accepted path — and truncates on commit.

In self-speculation, the KV cache rollback is even simpler: since the draft heads share the target model's base layers, the KV cache for the prefix is identical. The draft heads' predictions are appended to the same cache; on rejection, the tail is discarded, and the heads re-predict from the new hidden state. This is essentially a single-pointer rollback and is one reason self-speculation is easier to integrate into existing serving stacks.

---

## What actually ships in production

A quick calibration on which of these methods you'll encounter in real systems:

- **Top-p + temperature**: the near-universal default for user-facing chat. Typical settings: $T = 0.7$–$1.0$, $p = 0.9$–$0.95$.
- **Greedy / low-temperature**: code generation, structured output, math — anywhere the task rewards determinism.
- **Speculative decoding (self-speculation)**: now standard in vLLM, TensorRT-LLM, and most serving stacks for interactive latency.
- **Typical sampling, Mirostat, contrastive decoding**: mostly academic or niche (e.g., creative-writing frontends like `llama.cpp`). They work, but top-p is close enough and far simpler operationally.

The lesson is not that the fancier methods are wrong — it's that the simpler methods capture most of the quality gain, and operational simplicity wins in production.

---

## Further Reading

- Leviathan et al., 2023. "Fast Inference from Transformers via Speculative Decoding." ICML 2023.
- Chen et al., 2023. "Accelerating Large Language Model Decoding with Speculative Sampling." DeepMind.
- Holtzman et al., 2019. "The Curious Case of Neural Text Degeneration." ICLR 2020.
- Meister et al., 2023. "Locally Typical Sampling." TACL 2023.
- Basu et al., 2020. "Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity." ICLR 2021.
- Li et al., 2022. "Contrastive Decoding: Open-ended Text Generation as Optimization." ACL 2023.
- Cai et al., 2024. "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." ICML 2024.
- Li et al., 2024. "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty." ICML 2024.
