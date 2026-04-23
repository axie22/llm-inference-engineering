# Week 3: Advanced Sampling & Decoding

## Motivation

### The degeneration problem

Greedy decoding — picking the single most probable token at each step — produces remarkably bad text. Given a prompt like *"The meaning of life is"*, a greedy model will often spiral into repetitious loops: *"to live a life that is meaningful and to live a life that is full of meaning and to live a life…"*. This is not an implementation bug; it is a mathematical inevitability.

Let $p(x_t \mid x_{<t})$ be the model's distribution over the next token. Greedy decoding selects $\arg\max p(x_t)$. The problem is that the most probable continuation after a few tokens is almost always a **high‑frequency n‑gram** — a phrase the model has seen thousands of times in training. The greedy path converges to the mode of the conditional distribution at every step, and the mode is usually boring.

Beam search is not a fix. Maintaining $b$ hypotheses delays the collapse but does not prevent it; all $b$ beams eventually converge to the same high‑probability region. On open‑ended generation tasks (storytelling, dialogue), beam search actually scores *worse* than random sampling under human evaluation (Holtzman et al., 2019).

### The typical set insight

The correct framing comes from information theory. A model's distribution over sequences of length $n$ assigns a probability $p(x_{1:n})$ to each sequence. There is a set of sequences whose total probability mass is close to 1 — call this the **typical set**. Its size is roughly $2^{n H(p)}$, where $H(p)$ is the per‑token entropy.

Crucially, the *most probable* sequence (the greedy/beam path) is not in the typical set. The individual tokens in the greedy path each have high probability, but the *product* of many high‑probability tokens is astronomically small compared to the total probability mass of the entire typical set. Human‑written text lives in the typical set — it uses tokens that are individually plausible but collectively surprising enough to carry information.

In other words:
- **Too probable** → repetitive, generic, dull ("to live a life that is…")
- **Too improbable** → incoherent, off‑topic, gibberish
- **Just right** → tokens whose log‑probability is close to $-\log p(x)$ — the "band of moderate surprise"

Every sampling method in this lecture is an attempt to stay inside that band.

---

## The Sampling Zoo: a progression, not a list

### Greedy → repetition

Already covered. Greedy picks $\arg\max$ at each step and almost always hits the repetition wall.

### Temperature → looseness

Temperature $\tau$ scales the logits before the softmax:

$$p_\tau(x) = \frac{\exp(\log p(x) / \tau)}{\sum_{x'} \exp(\log p(x') / \tau)}$$

When $\tau > 1$, the distribution spreads out; lower‑probability tokens become more likely. This can break repetition loops, but it introduces a new problem: $\tau$ is a global knob, and the optimal value differs across contexts. A $\tau$ that works for a factual prompt ("List the planets") will be too hot for a creative prompt ("Write a poem"). Temperature alone cannot adapt to local distributional changes.

### Top‑k → brittle

Top‑k sampling restricts the vocabulary to the $k$ most probable tokens before sampling:

$$p_k(x) = \begin{cases}
p(x) / Z & \text{if } x \in \text{top‑}k \\
0 & \text{otherwise}
\end{cases}$$

This removes the long tail of unlikely tokens, which is good, but $k$ is fixed. When the distribution is **sharp** (one clear choice, e.g., a factual answer), top‑k needlessly includes many plausible‑looking alternatives that dilute the probability mass. When the distribution is **flat** (many equally plausible continuations), top‑k excludes reasonable candidates that fall outside the arbitrary cutoff. No single $k$ works across both regimes.

### Top‑p (nucleus) → adaptive

Top‑p solves top‑k's brittleness by cutting off the tail dynamically. Instead of fixing the *number* of tokens, fix the *total probability mass* $p$ (typically 0.9–0.95). Select the smallest set $V_p$ such that:

$$\sum_{x \in V_p} p(x) \ge p$$

Then renormalize over $V_p$ and sample. When the distribution is sharp, $V_p$ is small (just a few tokens). When it is flat, $V_p$ grows automatically. The cutoff adapts to the entropy of the current step, which is exactly what a fixed $k$ cannot do.

This is the dominant sampling method in production systems, but it has a subtle flaw: it still favors high‑probability tokens. The typical set can include tokens with moderate probability that top‑p would drop in a sharp distribution. The next methods address this directly.

### Typical sampling → the surprisal argument

Typical sampling (Meister et al., 2023) rejects any token whose *surprisal* $-\log p(x_t)$ is too far from the expected surprisal $H(p)$. Specifically, compute the entropy $H(p) = -\sum_x p(x) \log p(x)$, then sample only from tokens whose surprisal falls within a window $\tau$ of $H(p)$:

$$V_\text{typical} = \left\{ x \mid |-\log p(x) - H(p)| \le \tau \right\}$$

This directly implements the typical set intuition: discard tokens that are too predictable (surprisal too low) and too unpredictable (surprisal too high). The result is text that stays in the "band of moderate surprise" — the region where human‑like text lives.

Typical sampling works well but requires an entropy computation at every step, adding overhead. More importantly, it has no mechanism to *control* the overall surprise level of the generated text — it just enforces that each step is "typical."

### Mirostat → perplexity as a control target

Mirostat (a portmanteau of "miro" from "surprise" and "stat" for statistical control) takes a different approach. Instead of adjusting the vocabulary each step, it maintains a **target surprise rate** $\tau$ (equivalently, a target perplexity) and adjusts the temperature dynamically to keep the generated text near that target.

The algorithm at each step:

1. Sample from the model to estimate the current token's surprisal $s = -\log p(x)$.
2. Compare $s$ to the target $\tau$.
3. If $s < \tau$ (too predictable), increase temperature.
4. If $s > \tau$ (too surprising), decrease temperature.

More precisely, Mirostat maintains a running estimate of the **surprise probability distribution** over the vocabulary, and selects a top vocabulary set whose cumulative probability mass corresponds to a budget determined by $\tau$. The temperature is then updated via a feedback loop:

$$\tau_\text{next} = \tau_\text{current} + \eta \cdot (s - \tau)$$

where $\eta$ is a learning rate controlling how aggressively the target is enforced.

This turns generation into a control problem: you set $\tau$ (e.g., 5.0 for creative, 2.0 for factual), and Mirostat regulates the temperature to hit that target, step by step. The result is text that doesn't drift into repetition or incoherence over long generations.

### Contrastive decoding → amplify the gap

A completely different family: instead of sampling from a single model, **contrastive decoding** uses two models — a large "expert" $q(x)$ and a small "amateur" $p(x)$ — and amplifies the differences between their distributions. The idea is that tokens the amateur also finds likely are probably generic n‑gram matches; tokens the expert favors over the amateur carry the expert's special knowledge.

The algorithm (Li et al., 2022):

1. **Plausibility constraint** — discard any token where the expert assigns vanishing probability:
   $$V_\text{head} = \left\{ x \mid q(x) \ge \beta \cdot \max_x q(x) \right\}$$
   Typical $\beta = 0.01$–$0.1$. This prevents the expert from falling back on its distribution's tail, which would amplify noise.

2. **Score by log‑ratio** — within $V_\text{head}$, compute:
   $$s(x) = \log q(x) - \log p(x)$$
   Then sample from $\text{softmax}(s(x) / \tau)$.

The subtraction is not simple — the plausibility constraint is critical. Without it, a token where $q(x)$ is very small but $p(x)$ is even smaller would get a high score, dominating the distribution. The constraint ensures only high‑expert‑support tokens compete.

Contrastive decoding produces measurably higher‑quality text on both automatic and human evaluation. Its main cost is running a second (smaller) model, which adds latency but is often acceptable since the amateur model can be significantly smaller.

---

## Speculative Decoding — the full argument

### The bottleneck

Autoregressive generation is intrinsically serial: token $t+1$ depends on all previous tokens. On a GPU, each step loads the full model weights from HBM and produces a single token. For a 7B‑parameter model, a single forward pass takes ~5 ms on an A100. Generating a 256‑token completion therefore takes ~1.3 s — dominated by memory‑bound weight loads, not compute.

### The key insight

A smaller "draft" model (e.g., 1 B parameters) generates tokens much faster — say ~1 ms per step. The question: can the small model's output be *verified* by the large model in a way that allows generating multiple tokens per large‑model call?

Yes. The large model can process a sequence of $K$ candidate tokens in a **single forward pass** — the same cost as generating one token. This is a property of transformer architectures: computing $K$ consecutive logit vectors from a prompt costs the same as computing one, because the KV cache from the prompt is reused and the attention over the candidates is parallelized.

### The acceptance criterion

Let $p(x)$ be the target (large) model's distribution and $q(x)$ the draft (small) model's distribution. The draft model generates a candidate token $x \sim q(x)$. The target model computes $p(x)$ and $q(x)$. Accept the draft token with probability:

$$\min\left(1, \frac{q(x)}{p(x)}\right)$$

Intuitively: if the draft model's probability is *higher* than the target's, the draft is being too conservative (it picked an overly generic token) — we accept it because the target doesn't prefer alternatives. If the draft's probability is *lower*, the target potentially has a better choice — we may reject.

When a token is rejected, we resample from the adjusted distribution:

$$\frac{(q(x) - p(x))_+}{Z}, \quad Z = \sum_x (q(x) - p(x))_+$$

where $(a)_+ = \max(0, a)$. This ensures that the marginal distribution of accepted tokens exactly matches $p(x)$ — i.e., the algorithm is **lossless** in expectation. The proof is a one‑line Gibbs sampling argument (Leviathan et al., 2023):

$$P(\text{accept } x) = \sum_x q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right) = 1 - \sum_x (q(x) - p(x))_+$$

The rejection‑corrected distribution $\pi(x)$ satisfies $\pi(x) = p(x)$ for all $x$ (verify by writing out the resampling step). This mathematical elegance — that an approximate draft model can accelerate exact inference from the target — is the core achievement of speculative decoding.

### Speedup formula

Let $\alpha$ be the token‑wise acceptance probability, $\gamma$ be the number of draft tokens generated per target call, $t_\text{draft}$ be the per‑token draft time, and $t_\text{target}$ be the per‑forward‑pass target time (which is also the per‑token time in standard decoding).

The expected number of tokens produced per target forward pass is:

$$\mathbb{E}[\text{tokens}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

Derivation: starting with $\gamma$ draft tokens, we accept each with probability $\alpha$, and the final token (whether accepted or resampled) is always produced. This is a truncated geometric series.

The speedup relative to standard decoding is:

$$\text{speedup} = \frac{\mathbb{E}[\text{tokens}] \cdot t_\text{target}}{\gamma \cdot t_\text{draft} + t_\text{target}}$$

Two regimes:
- **Draft is fast** ($t_\text{draft} \ll t_\text{target}$, e.g., a 100× smaller model): speedup approaches $\mathbb{E}[\text{tokens}]$.
- **Draft quality is high** ($\alpha \to 1$): $\mathbb{E}[\text{tokens}] \approx \gamma + 1$, so speedup scales with speculation length.

In practice, $\gamma = 4$–$8$ and $\alpha \approx 0.7$–$0.9$ for well‑matched draft models, yielding 2–3× speedup.

### Tree‑based verification

The draft model can generate a *tree* of candidates rather than a single chain. A small draft model produces its top‑$k$ tokens at the first step, then top‑$k$ for each of those, etc., forming a beam‑search tree of depth $\gamma$ and branching factor $k$. The target model processes all $\sum_{t=0}^{\gamma-1} k^t$ candidates simultaneously using a tree attention mask (Cai et al., 2024). Verified paths are accepted greedily or by sampling. This increases the probability of finding an acceptable continuation, improving $\alpha$ per step without increasing $\gamma$.

### Self‑speculation (Medusa / EAGLE)

The overhead of maintaining a separate draft model — loading its parameters, duplicating KV caches — is substantial. **Self‑speculation** eliminates the draft model entirely by adding lightweight "draft heads" — small MLP modules — on top of the target model itself. These heads predict the next $\gamma$ tokens in parallel, conditioned on the target model's hidden states.

- **Medusa** (Cai et al., 2024): adds $\gamma$ independent draft heads, each predicting one future position. At verification time, the target model checks the joint likelihood of the entire candidate tree.
- **EAGLE** (Li et al., 2024): improves on Medusa by adding cross‑attention between draft heads and the target model's feature maps, capturing inter‑token dependencies that independent heads miss.

Because the draft heads share the target model's base layers, the KV cache is shared between drafting and verification — no duplication. The memory overhead is a few million parameters (vs. billions for a separate draft model). Self‑speculation is now the dominant production approach (vLLM, TensorRT‑LLM) because it provides speedup with minimal engineering cost.

---

## Connecting it to Week 2

Speculative decoding interacts non‑trivially with KV caching. The draft model generates candidate tokens using its own KV cache, which is small (fewer layers, fewer heads). When the target model verifies, it must compute KV pairs for the candidate tokens in its own cache.

The complication arises on **rejection**. If the target model rejects draft tokens at position $i$, the KV cache entries for positions $i$ through $\gamma$ must be discarded — they correspond to tokens that will not appear in the final output. Systems like vLLM implement this with a **cache rollback**: a simple pointer reset to the last confirmed prefix.

This is straightforward when the prefix is shared across a batch of requests: a single rollback pointer suffices for the entire batch. Tree‑based verification makes this slightly more involved, since multiple branches share different subsets of the prefix. The implementation stores a **candidate mask** alongside the KV cache — bits indicating which positions belong to the accepted path — and truncates on commit.

In self‑speculation, the KV cache rollback is even simpler: since the draft heads share the target model's base layers, the KV cache for the prefix is identical. The draft heads' predictions are appended to the same cache; on rejection, the tail is discarded, and the head predicts again from the new hidden state. This is essentially a single‑pointer rollback and is one reason self‑speculation is easier to integrate into existing serving stacks.

---

## Further Reading

- Leviathan et al., 2023. "Fast Inference from Transformers via Speculative Decoding." ICML 2023.
- Chen et al., 2023. "Accelerating Large Language Model Decoding with Speculative Sampling." DeepMind, 2023.
- Holtzman et al., 2019. "The Curious Case of Neural Text Degeneration." ICLR 2020.
- Meister et al., 2023. "Typical Sampling for Text Generation." NeurIPS 2023.
- Li et al., 2022. "Contrastive Decoding: Open‑ended Text Generation as Optimization." ACL 2023.
- Cai et al., 2024. "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." ICML 2024.
- Li et al., 2024. "Eagle: Speculative Acceleration for LLMs without a Draft Model." ICML 2024.
