# Fine-Tuning Strategy & Dataset Preparation
## A Complete Guide to When and How to Fine-Tune LLMs for Domain Adaptation

---

## Table of Contents

1. [The Three Approaches to Domain Adaptation](#1-the-three-approaches-to-domain-adaptation)
2. [When Fine-Tuning Is the Right Choice](#2-when-fine-tuning-is-the-right-choice)
3. [The Decision Framework](#3-the-decision-framework)
4. [Full Fine-Tuning vs Parameter-Efficient Fine-Tuning](#4-full-fine-tuning-vs-parameter-efficient-fine-tuning)
5. [QLoRA: The Technique We Use](#5-qlora-the-technique-we-use)
6. [Choosing a Base Model](#6-choosing-a-base-model)
7. [Dataset Selection and Quality](#7-dataset-selection-and-quality)
8. [Creating Custom Datasets](#8-creating-custom-datasets)
9. [Chat Template Formatting](#9-chat-template-formatting)
10. [The Before/After Methodology](#10-the-beforeafter-methodology)
11. [Common Misconceptions](#11-common-misconceptions)
12. [How Our Healthcare Agent Uses This](#12-how-our-healthcare-agent-uses-this)
13. [Full Workshop Pipeline Overview](#13-full-workshop-pipeline-overview)
14. [Hands-On Exercise: Dataset Quality Audit & Prompt Engineering Baseline](#14-hands-on-exercise)

---

## 1. The Three Approaches to Domain Adaptation

When you need an LLM to perform well in a specific domain (healthcare, legal, finance), there are three approaches — and most teams jump to the most expensive one first.

```
APPROACH 1: PROMPT ENGINEERING
  Change WHAT you send to the model.
  System prompt: "You are a healthcare expert. Always cite evidence-based guidelines..."
  Few-shot examples: Include 2-3 example Q&A pairs in the prompt.

  Cost:        $0 (just tokens)
  Effort:      Hours
  Reversible:  Yes (change the prompt, instant effect)
  Quality:     Often surprisingly good for 80% of use cases

APPROACH 2: RETRIEVAL-AUGMENTED GENERATION (RAG)
  Change WHAT CONTEXT the model sees.
  Retrieve relevant medical documents at runtime and inject into the prompt.
  The model answers grounded in specific retrieved evidence.

  Cost:        Vector DB + embedding costs
  Effort:      Days to weeks
  Reversible:  Yes (change documents, re-index)
  Quality:     Excellent for factual, document-grounded answers

APPROACH 3: FINE-TUNING
  Change THE MODEL ITSELF.
  Train on domain-specific data so the model's weights encode domain knowledge.
  The model "thinks" in the domain's language natively.

  Cost:        GPU compute + dataset curation
  Effort:      Days to weeks
  Reversible:  No (creates a new model version)
  Quality:     Best for tone, format, and deep domain behavior
```

**Critical insight:** These are NOT mutually exclusive. The best production systems often combine all three: a fine-tuned model + RAG for grounding + carefully crafted prompts.

---

## 2. When Fine-Tuning Is the Right Choice

Fine-tuning is **not** a magic bullet. It solves specific problems that prompt engineering and RAG cannot.

### Fine-Tuning Wins When:

```
PROBLEM                                          WHY FINE-TUNING HELPS
──────────────────────────────────────────────────────────────────────────────────
Consistent output format/tone                    Model learns the STYLE natively.
  "Always respond in clinical                    No need for verbose format
   note format with sections:                    instructions in every prompt.
   Chief Complaint, HPI, Assessment"

Domain-specific terminology                      Model learns to USE terms correctly.
  Medical abbreviations, drug names,             "SOB" = "shortness of breath",
  procedure codes                                not the colloquial meaning.

Reduced token cost at scale                      System prompt can be shorter.
  The model already "knows" the domain,          No need for lengthy few-shot
  so fewer prompt tokens needed.                 examples on every call.

Reduced hallucination in narrow domain           Model's probability distribution
  Base model gives generic answers;              shifts toward domain-correct
  fine-tuned model gives precise ones.           completions.

Latency-sensitive applications                   Shorter prompts = faster inference.
  Every token in the prompt adds latency.        Fine-tuned model needs less
  Fine-tuning encodes knowledge in weights.      context to respond correctly.
```

### Fine-Tuning Does NOT Help When:

```
PROBLEM                                          BETTER SOLUTION
──────────────────────────────────────────────────────────────────────────────────
Need up-to-date information                      RAG (retrieve latest docs)
  Drug approvals change, guidelines update.      Fine-tuning encodes knowledge
  Fine-tuning captures training-time data.       at training time — stale by
                                                 next month.

Need to cite specific sources                    RAG (source attribution)
  "According to WHO guidelines (2024)..."        Fine-tuned model can't cite
  Requires grounding in a specific document.     what it "learned" from training.

Simple formatting or instruction following       Prompt engineering
  "Respond in JSON format"                       GPT-4o-mini already does this
  "Limit response to 3 sentences"                perfectly with good prompts.

One-off or low-volume tasks                      Prompt engineering
  10 queries/day doesn't justify                 The ROI of fine-tuning only
  fine-tuning compute costs.                     kicks in at scale.
```

---

## 3. The Decision Framework

Use this flowchart before spending any GPU hours:

```
START: "I need domain-specific LLM behavior"
  │
  ▼
"Have I tried prompt engineering?"
  │
  ├── NO → Try it first. Seriously.
  │        Write a detailed system prompt.
  │        Add 2-3 few-shot examples.
  │        Test on 20+ diverse queries.
  │        │
  │        ▼
  │     "Is the quality sufficient?"
  │        ├── YES → Stop. You're done. Ship it. ✅
  │        └── NO  → Continue below ▼
  │
  └── YES → "What specifically is failing?"
            │
            ├── "Factual accuracy on specific documents"
            │     → RAG. Retrieve the documents at runtime.
            │
            ├── "Output tone/format is inconsistent"
            │     → Fine-tuning. The model needs style training.
            │
            ├── "Domain terminology is wrong"
            │     → Fine-tuning. The model needs vocabulary shift.
            │
            ├── "Answers are too generic"
            │     → Try RAG first. If still generic → Fine-tuning.
            │
            └── "Costs too much per query (long prompts)"
                  → Fine-tuning. Shorter prompts = lower cost.
```

**The golden rule:** Prompt engineering first. RAG second. Fine-tuning third. Each step is more expensive and less reversible than the last.

---

## 4. Full Fine-Tuning vs Parameter-Efficient Fine-Tuning

### Full Fine-Tuning

Updates **every parameter** in the model:

```
Qwen2.5-1.5B: 1.5 billion parameters
  → ALL 1.5B weights are updated during training
  → Requires: 6 bytes/param (fp32 optimizer) × 1.5B = ~9 GB optimizer states
  → Total VRAM: 20-40 GB minimum (even with bf16)
  → Result: A complete new model (same size as original)
```

Pros: Maximum quality potential.
Cons: Requires significant GPU, risk of catastrophic forgetting, produces a full-size model.

### Parameter-Efficient Fine-Tuning (PEFT)

Updates only a **tiny fraction** of parameters:

```
                    FULL FINE-TUNING              PEFT (LoRA/QLoRA)
                    ─────────────────             ─────────────────
Parameters updated  0.5 billion (100%)            ~7-15 million (~0.5-1%)
VRAM required       12-20 GB                      4-8 GB
Training time       Hours on A100                 Minutes on T4
Output size         3+ GB (full model)            10-50 MB (adapter only)
Risk of forgetting  High                          Low
Can stack adapters  No                            Yes (multiple adapters)
Quality vs full FT  Baseline                      95-99% as good
```

**PEFT is the practical choice** for 95% of fine-tuning use cases. Unless you're a large lab with A100 clusters, use PEFT.

### The PEFT Family

There are several PEFT methods. Each takes a different approach to the question: "How do I change a model's behavior while updating as few parameters as possible?"

#### 1. LoRA (Low-Rank Adaptation) — ⭐ Most Popular

**Idea:** Instead of updating the full weight matrix, add a small "side path" that learns the change.

```
FULL FINE-TUNING (changes the whole matrix):

  Input ──► [ W_original (2048 × 2048) ] ──► Output
             ↑ 4,194,304 params updated

LoRA (adds a low-rank detour):

  Input ──► [ W_original (FROZEN) ] ──────────────────┐
    │                                                  │
    └──► [ A (2048 × 16) ] ──► [ B (16 × 2048) ] ──► (+) ──► Output
          ↑ 32,768 params      ↑ 32,768 params
          TRAINABLE             TRAINABLE

  Total trainable: 65,536 (1.6% of original)
  Output = W_frozen(x) + B(A(x)) × scaling_factor
```

**Analogy:** Imagine a highway (the frozen weight matrix). Instead of rebuilding the entire highway, LoRA adds a small side road (A × B) that gently adjusts the route. Traffic flows through both — the original highway stays unchanged, and the side road nudges the output in the right direction.

**Example in code:**
```python
from peft import LoraConfig

config = LoraConfig(
    r=16,                        # Rank: width of the "side road"
    lora_alpha=32,               # How much the side road's output is amplified
    target_modules="all-linear", # Which highways get side roads
)
# Result: ~7M trainable params out of 1.5B total
```

**When to use:** Default choice for almost everything. Best quality-to-cost ratio.

#### 2. QLoRA (Quantized LoRA) — ⭐ What We Use

**Idea:** Same as LoRA, but the frozen base model is compressed to 4-bit first.

```
LoRA:
  Base model in fp16 (2 bytes/param) → 1.5B × 2 = 3.0 GB
  + LoRA adapters in bf16           → ~15M × 2  = 0.03 GB
  TOTAL: ~3 GB just for the model

QLoRA:
  Base model in 4-bit (0.5 bytes/param) → 1.5B × 0.5 = 0.75 GB
  + LoRA adapters in bf16               → ~15M × 2   = 0.03 GB
  TOTAL: ~0.8 GB for the model

  SAVINGS: 3 GB → 0.8 GB (73% reduction)
  QUALITY: 95-99% of LoRA (almost identical)
```

**Analogy:** LoRA adds a side road to a normal highway. QLoRA adds the same side road but first compresses the highway from 4 lanes to 1 lane (4-bit). The side road still works fine — it was designed for the compressed highway.

**Why it was a breakthrough (Dettmers et al., 2023):**
```
Before QLoRA:
  Fine-tuning a 7B model → needed A100 (80 GB) or multiple GPUs
  Cost: $2-5/hour for GPU rental
  Accessibility: Only well-funded labs

After QLoRA:
  Fine-tuning a 7B model → works on a single T4 (15 GB)
  Cost: Free (Google Colab)
  Accessibility: Anyone with a Google account

  QLoRA democratized fine-tuning.
```

**When to use:** Whenever you'd use LoRA but have limited VRAM (Colab, consumer GPUs).

#### 3. Prefix Tuning / P-Tuning

**Idea:** Instead of modifying the model's layers, prepend a few "soft" (learnable) tokens to the input.

```
NORMAL INPUT:
  "What are the symptoms of diabetes?" → Model → Response

PREFIX TUNING:
  [SOFT_TOKEN_1][SOFT_TOKEN_2]...[SOFT_TOKEN_20] "What are the symptoms of diabetes?"
  ↑ These 20 tokens are TRAINABLE embeddings        ↑ Real input (unchanged)
  They don't correspond to real words.
  They're learned vectors that "steer" the model's attention.
```

**Analogy:** Imagine whispering secret instructions to someone before they read a question out loud. The audience can't hear the whisper, but it changes how the person answers. The "whisper" is the learned prefix.

**Example:**
```python
from peft import PrefixTuningConfig

config = PrefixTuningConfig(
    num_virtual_tokens=20,       # Number of soft tokens prepended
    task_type="CAUSAL_LM",
)
# Result: Only ~40K trainable params (0.003% of model)
```

```
COMPARISON TO LoRA:
  Trainable params:  ~40K (prefix) vs ~9M (LoRA) → 367× fewer
  Quality:           Okay for simple tasks, struggles with complex ones
  Flexibility:       Limited — the prefix can only steer, not reshape outputs

  Think of it as: LoRA does surgery on the model's brain.
                  Prefix tuning just gives it a post-it note.
```

**When to use:** Very constrained environments, or when you need multiple "modes" (swap prefix = swap behavior). Rarely used in practice since LoRA is better.

#### 4. Adapter Layers (Houlsby / Bottleneck Adapters)

**Idea:** Insert small neural network modules BETWEEN the existing layers.

```
ORIGINAL TRANSFORMER LAYER:
  Input → [Attention] → [Feed-Forward] → Output

WITH ADAPTER LAYERS:
  Input → [Attention] → [ADAPTER ↓] → [Feed-Forward] → [ADAPTER ↓] → Output
                         ↑ small MLP                     ↑ small MLP
                         (d → r → d)                     (d → r → d)
                         TRAINABLE                       TRAINABLE

  Each adapter: down-project (d → r), ReLU, up-project (r → d)
  Just like LoRA's A × B, but with a nonlinearity in the middle.
```

**Analogy:** Instead of adding a side road (LoRA), adapters insert a small checkpoint station between every major section of the highway. Every car (token) must pass through the checkpoint, which slightly adjusts its direction.

**Example:**
```python
# Not commonly used directly — most people use LoRA instead
# But conceptually:
# Each adapter: Linear(2048 → 64) + ReLU + Linear(64 → 2048)
# Inserted after attention AND after FFN in each layer
```

```
COMPARISON TO LoRA:
  Trainable params:  Similar (~1-5% of total)
  Quality:           Comparable to LoRA
  Speed:             Slightly SLOWER (adds sequential computation)
  Popularity:        Lower (LoRA became the standard)

  Key difference: LoRA runs IN PARALLEL with existing layers (added output).
                  Adapters run IN SERIES (inserted between layers).
                  Parallel is faster because it doesn't increase the critical path.
```

**When to use:** Historically used before LoRA existed (2019-2021). Mostly superseded by LoRA for new projects.

#### 5. IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)

**Idea:** Don't add new matrices at all. Just learn a SCALING VECTOR that multiplies existing activations.

```
ORIGINAL:
  attention_output = softmax(Q × K^T) × V

IA3:
  attention_output = softmax(Q × K^T) × (V ⊙ learned_vector)
                                          ↑ element-wise multiply
                                          by a trainable vector of size d

  The learned_vector amplifies some dimensions and suppresses others.
  Like an equalizer on a stereo — boost the bass, cut the treble.
```

**Analogy:** LoRA builds a new side road. IA3 just installs traffic lights on the existing road. Green light = let this signal through stronger. Red light = dim this signal. Much simpler, but less flexible.

```
COMPARISON TO LoRA:
  Trainable params:  ~10K (IA3) vs ~9M (LoRA) → 1,470× fewer
  Quality:           Noticeably lower for complex tasks
  Speed:             Fastest training (almost no overhead)
  Use case:          Quick experiments, not production fine-tuning

  IA3 is the "minimum viable fine-tuning" — barely changes the model.
```

**When to use:** Extremely resource-constrained situations, or as a quick test before committing to LoRA.

### PEFT Family Summary

```
METHOD           TRAINABLE PARAMS   HOW IT WORKS                       QUALITY    SPEED
─────────────────────────────────────────────────────────────────────────────────────────
LoRA             ~9M (~1.8%)       Parallel low-rank matrices         ★★★★★      ★★★★
QLoRA            ~9M (~1.8%)       LoRA + 4-bit base quantization     ★★★★★      ★★★★
Prefix Tuning    ~40K (~0.003%)     Learnable soft tokens prepended    ★★★        ★★★★★
Adapter Layers   ~5-15M (~0.5-1%)   Small MLPs inserted between layers ★★★★       ★★★
IA3              ~10K (~0.001%)     Learned scaling vectors            ★★★        ★★★★★

RECOMMENDATION FOR THIS WORKSHOP:
  QLoRA ← Best trade-off: LoRA quality at 4-bit memory cost.
          Runs on free Colab T4. Production-grade quality.

RECOMMENDATION FOR FUTURE PROJECTS:
  Start with QLoRA. Only consider alternatives if you have a specific
  constraint (e.g., need < 100K params → Prefix Tuning or IA3).
```

---

## 5. QLoRA: The Technique We Use

QLoRA (Quantized Low-Rank Adaptation) combines two ideas: **4-bit quantization** of the base model and **LoRA adapters** for training.

### How LoRA Works

Instead of updating the full weight matrix, LoRA adds a low-rank decomposition:

```
Original weight matrix W (d × d):
  W_new = W_frozen + ΔW
  
Where ΔW = A × B
  A is (d × r)     ← r is the "rank", much smaller than d
  B is (r × d)

Example with d=2048, r=16:
  Original W: 2048 × 2048 = 4,194,304 parameters
  LoRA A:     2048 × 16   = 32,768 parameters
  LoRA B:     16 × 2048   = 32,768 parameters
  Total LoRA:               65,536 parameters (1.6% of original)

  The model learns ΔW through A and B.
  W stays FROZEN. Only A and B are trained.
```

### The Scaling Factor (alpha)

```
ΔW = (alpha / r) × A × B

alpha=32, r=16  →  scaling = 32/16 = 2.0
  The adapter's contribution is amplified by 2x.

alpha=16, r=16  →  scaling = 16/16 = 1.0
  The adapter's contribution is used as-is.

Rule of thumb: alpha = 2 × r is a safe default.
  Our config: alpha=32, r=16 → scaling=2.0
```

### How Quantization Works (4-bit NF4)

```
NORMAL MODEL LOADING (fp16):
  Each parameter: 16 bits (2 bytes)
  1.5B params × 2 bytes = 3.0 GB VRAM

4-BIT QUANTIZED LOADING (NF4):
  Each parameter: 4 bits (0.5 bytes)
  1.5B params × 0.5 bytes = 0.75 GB VRAM
  + Quantization metadata ≈ 0.3 GB
  Total: ~1.0 GB VRAM

SAVINGS: 3.0 GB → 1.0 GB (67% reduction)
```

NF4 (Normal Float 4-bit) is specifically designed for neural network weights, which follow a normal distribution. It maps the 4-bit values to optimally represent this distribution.

### Double Quantization

```
WITHOUT double quant:
  Base model in 4-bit + quantization constants in fp32
  The constants themselves take memory

WITH double quant (what we use):
  Base model in 4-bit + quantization constants ALSO quantized
  Further ~0.1 GB savings on a 1.5B model

Our BitsAndBytesConfig:
  load_in_4bit = True
  bnb_4bit_quant_type = "nf4"
  bnb_4bit_compute_dtype = bfloat16    ← compute in bf16 for speed
  bnb_4bit_use_double_quant = True     ← quantize the quantization constants
```

### QLoRA = Quantization + LoRA Together

```
┌────────────────────────────────────────────────────────┐
│  QLoRA TRAINING                                        │
│                                                        │
│  Base Model (1.5B params)                              │
│  ├── Quantized to 4-bit NF4 (FROZEN, ~1 GB VRAM)      │
│  │                                                     │
│  └── LoRA Adapters (A × B matrices)                    │
│      ├── Added to attention layers                     │
│      ├── Trained in bf16 (~15M params, ~0.1 GB)        │
│      └── target_modules = "all-linear"                 │
│                                                        │
│  Total VRAM: ~4-6 GB on Colab T4 (15 GB available)    │
│  Output: 10-50 MB adapter files (not the full model)   │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Our LoRA Configuration Explained

```python
LoraConfig(
    r=16,                          # Rank: capacity of the adapter
                                   # Higher r = more capacity but slower
                                   # 8-32 is typical; 16 is a good default

    lora_alpha=32,                 # Scaling: alpha/r = 32/16 = 2.0
                                   # Controls magnitude of adapter output
                                   # Rule of thumb: alpha = 2×r

    lora_dropout=0.05,             # Dropout for regularization
                                   # Prevents overfitting on small datasets
                                   # 0.05-0.1 is typical

    target_modules="all-linear",   # Which layers get LoRA adapters
                                   # "all-linear" = q, k, v, o, gate, up, down
                                   # More modules = more capacity + cost

    task_type="CAUSAL_LM",         # Causal language modeling (next token)
)
```

### What `target_modules="all-linear"` Means

```
TRANSFORMER LAYER:
  ┌── Attention ──┐     ┌── MLP (Feed-Forward) ──┐
  │  q_proj ← LoRA│     │  gate_proj ← LoRA      │
  │  k_proj ← LoRA│     │  up_proj   ← LoRA      │
  │  v_proj ← LoRA│     │  down_proj ← LoRA      │
  │  o_proj ← LoRA│     │                         │
  └───────────────┘     └─────────────────────────┘

  7 LoRA adapters per layer × 24 layers (Qwen2.5-1.5B) = 196 adapter pairs
  Each pair: A (d × r) + B (r × d)
  Total trainable params: ~7M out of 1.5B (~0.5%)

  If target_modules = ["q_proj", "v_proj"] (minimal):
  2 LoRA adapters per layer × 28 layers = 56 adapter pairs
  Total trainable params: ~4.2M (~0.3%)
  Faster but lower quality.
```

---

## 6. Choosing a Base Model

### Why These Models?

```
SELECTION CRITERIA                    OUR CHOICES
──────────────────────────────────────────────────────────────────
v1: Qwen2.5-1.5B-Instruct            Already capable model
                                      Shows fine-tuning can HURT

v2: Qwen2.5-1.5B-Instruct            Same model, different data
                                      Shows data quality matters
                                      Reformatted WikiDoc via GPT-4o-mini

Both are instruction-tuned            "-Instruct" variants
                                      Can follow instructions out of the box
                                      We're fine-tuning ON TOP of this

Permissive licenses                   Apache 2.0
                                      Safe for commercial use

Well-supported tooling                Works with HF Transformers,
                                      PEFT, TRL, bitsandbytes
```

### The Model Size Trade-off

```
MODEL SIZE        VRAM (4-bit)    QUALITY          COLAB T4 FIT?
──────────────────────────────────────────────────────────────────
0.135B             ~0.27 GB (fp16)  Limited          ✅ Easily
1.5B              ~1.0 GB         Good             ✅ Comfortably  ← v1 & v2 pick (Qwen2.5-1.5B)
3B                ~1.8 GB         Very good        ✅ Tight during training
7B                ~4.0 GB         Excellent        ⚠️ Barely (reduce batch size)
14B               ~8.0 GB         Outstanding      ❌ Won't fit with training
70B               ~35 GB          State-of-the-art ❌ Need A100 80GB
```

### Instruct vs Base Models

```
BASE MODEL (e.g. Qwen2.5-1.5B):
  Trained on raw text (next token prediction)
  Good at completing text, not at following instructions
  "What is diabetes?" → "What is diabetes mellitus? What causes diabetes?..." (continues the text)

INSTRUCT MODEL (e.g. Qwen2.5-1.5B-Instruct):
  Base model + additional training on instruction-following data
  Understands system prompts, user queries, assistant responses
  "What is diabetes?" → "Diabetes is a chronic condition affecting blood sugar..." (answers the question)

WE FINE-TUNE THE INSTRUCT MODEL because:
  ✅ It already knows how to follow instructions
  ✅ You only need to add healthcare domain knowledge
  ✅ Much less training data needed than starting from scratch
  ❌ If we used the base model, we'd need BOTH instruction data AND domain data
```

---

## 7. Dataset Selection and Quality

### The v1 → v2 Lesson: Model Choice + Data Quality

This workshop uses **two experiments** to show you two critical fine-tuning lessons:

```
v1 (training_v1.ipynb) — 1.5B MODEL + ChatDoctor:
  Model:     Qwen2.5-1.5B-Instruct (already capable at medical Q&A)
  Dataset:   lavita/ChatDoctor-HealthCareMagic-100k
  Result:    Fine-tuning DEGRADED quality — base model was already excellent,
             but fine-tuning injected ChatDoctor persona ("Hello, welcome to Chat Doctor")
             while losing structured, detailed formatting.
  Lesson:    Don't fine-tune a capable model on noisy data.

v2 (training_v2.ipynb) — 1.5B MODEL + Reformatted WikiDoc:
  Model:     Qwen2.5-1.5B-Instruct (same model as v1)
  Dataset:   jeev1992/wikidoc-healthassist (WikiDoc reformatted via GPT-4o-mini)
  Result:    Safety improved significantly, model strengths preserved.
             Clean, well-formatted data makes the difference.
  Lesson:    Data quality is everything — clean data preserves model strengths.

SAME MODEL, SAME LORA CONFIG, SAME BENCHMARK PROMPTS.
The ONLY variable is dataset quality. This proves:
  1. Don't fine-tune capable models on noisy data (v1)
  2. Clean, well-formatted data can successfully add safety and style (v2)
```

### The ChatDoctor Dataset: lavita/ChatDoctor-HealthCareMagic-100k

Both notebooks use the same dataset — real patient-doctor conversations:

```
Source:     Hugging Face Hub (real patient-doctor conversations)
License:    Research use
Size:       ~112,000 QA pairs
Format:     {"instruction": "...", "input": "patient question", "output": "doctor answer"}
Domain:     Real-world medical consultations from HealthCareMagic platform

Examples:
  Q: "I have been experiencing fatigue and frequent urination for weeks..."
  A: "Based on your symptoms of fatigue and polyuria, this could suggest
      several conditions. The most common cause would be diabetes mellitus,
      particularly Type 2 diabetes. I would recommend getting a fasting
      blood glucose test and an HbA1c test..."

STRENGTHS:
  ✅ Real doctor-patient conversations (natural style)
  ✅ Multi-paragraph detailed answers
  ✅ Covers wide range of medical topics
  ✅ Large dataset (112K examples)

WEAKNESSES (data quality issues we demonstrate):
  ❌ Persona contamination ("Welcome to Chat Doctor", "Hope this helps")
  ❌ Boilerplate greetings and sign-offs
  ❌ Inconsistent formatting (some answers poorly structured)
  ❌ Quality varies widely across examples

These weaknesses are INTENTIONAL for our workshop — they let us
demonstrate how data quality impacts fine-tuning results.
We clean some boilerplate with regex, but not all of it gets caught.
```

### Measured Quality Metrics (from Module 1 Exercise)

These numbers come from running the dataset quality audit notebook on both datasets:

```
METRIC                              CHATDOCTOR         WIKIDOC
──────────────────────────────────────────────────────────────────
Total examples                         112,165           2,100
Persona contamination                    63.1%            0.0%
Boilerplate sign-offs                    28.2%            0.0%
Very short answers (<50ch)                0.6%            0.0%
Safety disclaimers                        3.2%           99.4%
Avg answer length (chars)                  603             910

Key findings:
  - 63% of ChatDoctor answers contain persona artifacts that the model WILL learn
  - Only 3.2% of ChatDoctor answers include safety disclaimers vs 99.4% for WikiDoc
  - WikiDoc answers are 50% longer on average — more thorough medical explanations
  - WikiDoc has ZERO persona contamination or boilerplate (cleaned by GPT-4o-mini)
```

### Dataset Comparison: What To Look For

```
CRITERIA                         CHATDOCTOR (v1)                  WIKIDOC (v2)
─────────────────────────────────────────────────────────────────────────────────
Free and publicly available      ✅ Research use license           ✅ HF Hub
Medical domain relevant          ✅ Real patient consultations     ✅ WikiDoc medical articles
QA format                        ✅ Patient Q + doctor A           ✅ Reformatted by GPT-4o-mini
Size                             ✅ 112K examples                  ✅ 2,100 examples
Answer depth                     ✅ Multi-paragraph                ✅ Avg 910 chars (50% longer)
Persona contamination            ❌ 63.1% of answers               ✅ 0.0%
Boilerplate sign-offs            ❌ 28.2% of answers               ✅ 0.0%
Safety disclaimers               ❌ Only 3.2% include them         ✅ 99.4% include them
Data quality                     ❌ Noisy, inconsistent            ✅ Clean, consistent format
```

### Dataset Quality Matters More Than Quantity

```
1,000 HIGH-QUALITY EXAMPLES:         10,000 LOW-QUALITY EXAMPLES:
──────────────────────────            ──────────────────────────
✅ Accurate medical info              ❌ Contains errors or outdated info
✅ Consistent format                  ❌ Mixed formats (some Q&A, some prose)
✅ Complete answers                   ❌ Truncated or partial answers
✅ Relevant to target domain          ❌ Noisy (irrelevant topics mixed in)
✅ Diverse question types             ❌ Repetitive (same question phrased 50 ways)

3 epochs × 1,000 = 3,000 steps       3 epochs × 10,000 = 30,000 steps
Training: ~15 minutes on T4          Training: ~2.5 hours on T4
Quality: Good                        Quality: Often WORSE (garbage in, garbage out)
```

**We use 2,000 training samples and 100 eval samples.** Enough to shift the model's behavior without overfitting or excessive training time.

### What Makes a Bad Dataset

```
❌ Outdated medical information
   "HIV is always fatal" — true in 1985, not in 2024

❌ Inconsistent terminology
   Same drug called "acetaminophen" in some examples and "paracetamol" in others
   without consistency → model learns confusion

❌ Too narrow
   All 1,000 examples about diabetes → model becomes a diabetes-only expert
   and FORGETS general medical knowledge (catastrophic forgetting)

❌ Machine-generated without review
   GPT-4 generated medical QA pairs with no human review
   → amplifies and solidifies hallucinations

❌ Wrong format
   Training data in paragraph form when you want Q&A chat format
   → model learns to generate paragraphs, not answer questions
```

---

## 8. Creating Custom Datasets

Off-the-shelf datasets are a great starting point, but they often have quality issues. In our workshop we saw this firsthand: the ChatDoctor dataset had persona contamination ("Welcome to Chat Doctor") that leaked into the fine-tuned model despite cleaning. Sometimes you need to build your own dataset.

### The Four Approaches to Dataset Creation

```
APPROACH           COST        QUALITY     SPEED       BEST FOR
────────────────────────────────────────────────────────────────────────
1. Synthetic       $5-50       ★★★★        Hours       Style/format training
   (LLM-generated)             (with QA)               Bootstrapping a new domain

2. Curated from    $0          ★★★★★       Days-Weeks  When source docs exist
   existing docs                                       (guidelines, textbooks)

3. Expert          $500+       ★★★★★       Weeks       High-stakes domains
   annotation                                          (medical, legal, finance)

4. User feedback   $0          ★★★★        Ongoing     Production systems with
   / logs                                              real user interactions
```

### Approach 1: Synthetic Data Generation (Most Practical)

Use a strong model (GPT-4o, Claude) to generate training pairs, then validate.

```
WHY THIS WORKS:
  Strong models are excellent at generating DIVERSE, well-formatted examples.
  They can produce 1,000 high-quality QA pairs in minutes.
  Cost: ~$5-10 for 1,000 examples with GPT-4o-mini.

WHY IT'S RISKY:
  The generating model's biases become your training data's biases.
  Hallucinated medical facts get baked into your fine-tuned model.
  ALWAYS validate with a domain expert or against authoritative sources.

THE PATTERN:
  1. Write a detailed generation prompt
  2. Generate in batches (50-100 at a time)
  3. Validate / filter / deduplicate
  4. Convert to chat format
```

**Step 1: Write a Generation Prompt**

```
You are generating training data for a healthcare assistant chatbot.

For each example, create:
- A realistic PATIENT QUESTION (natural language, varying complexity)
- A detailed DOCTOR RESPONSE (2-4 paragraphs, clinically accurate,
  mentions relevant mechanisms, treatments, and when to seek care)

Requirements:
- Cover diverse medical topics: cardiology, endocrinology, pulmonology,
  pharmacology, preventive care, emergency medicine, mental health
- Vary question styles: symptom queries, drug questions, lifestyle advice,
  condition comparisons, diagnostic process questions
- Responses should be thorough but accessible to patients
- Include appropriate caveats ("consult your healthcare provider")
- Do NOT use placeholder text or generic filler

Generate 10 examples in this JSON format:
[
  {
    "question": "...",
    "answer": "..."
  }
]
```

**Step 2: Generate in Batches**

```python
import openai
import json

client = openai.OpenAI()  # uses OPENAI_API_KEY env var

all_examples = []
TOPICS = [
    "cardiology and heart disease",
    "endocrinology and diabetes",
    "pulmonology and respiratory conditions",
    "pharmacology and medication management",
    "preventive care and vaccinations",
    "emergency medicine and urgent symptoms",
    "mental health and psychiatry",
    "gastroenterology and digestive health",
    "neurology and neurological conditions",
    "orthopedics and musculoskeletal health",
]

for topic in TOPICS:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": GENERATION_PROMPT},
            {"role": "user", "content": f"Generate 100 examples focused on: {topic}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.9,   # higher temp = more diversity
    )
    batch = json.loads(response.choices[0].message.content)
    all_examples.extend(batch["examples"])
    print(f"{topic}: {len(batch['examples'])} examples")

print(f"Total generated: {len(all_examples)}")
```

**Step 3: Quality Filtering**

```python
def quality_filter(example):
    """Filter out low-quality synthetic examples."""
    q, a = example["question"], example["answer"]

    # Too short — likely incomplete
    if len(a) < 100:
        return False

    # Too long — likely rambling or repetitive
    if len(a) > 3000:
        return False

    # Question too vague
    if len(q) < 15:
        return False

    # Check for placeholder/filler text
    filler = ["lorem ipsum", "[insert", "example response", "placeholder"]
    if any(f in a.lower() for f in filler):
        return False

    # Check for self-reference ("As an AI...")
    ai_refs = ["as an ai", "i'm an ai", "language model", "i cannot diagnose"]
    if any(ref in a.lower() for ref in ai_refs):
        return False

    return True

filtered = [ex for ex in all_examples if quality_filter(ex)]
print(f"After filtering: {len(filtered)} / {len(all_examples)}")
```

**Step 4: Deduplicate**

```python
def deduplicate(examples, similarity_threshold=0.85):
    """Remove near-duplicate questions."""
    from difflib import SequenceMatcher
    unique = []
    for ex in examples:
        is_dup = False
        for existing in unique:
            sim = SequenceMatcher(None, ex["question"].lower(),
                                 existing["question"].lower()).ratio()
            if sim > similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(ex)
    return unique

deduped = deduplicate(filtered)
print(f"After dedup: {len(deduped)} / {len(filtered)}")
```

**Step 5: Convert to HF Dataset Format**

```python
from datasets import Dataset
import json

SYSTEM_PROMPT = (
    "You are a knowledgeable and thorough healthcare assistant. "
    "Provide comprehensive explanations with relevant clinical details, "
    "mechanisms of action, and practical guidance. "
    "Always recommend consulting a healthcare professional for serious concerns."
)

records = []
for ex in deduped:
    records.append({
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": ex["question"]},
            {"role": "assistant", "content": ex["answer"]},
        ]
    })

ds = Dataset.from_list(records)
ds = ds.shuffle(seed=42)

# Train/eval split
train_ds = ds.select(range(int(len(ds) * 0.9)))
eval_ds  = ds.select(range(int(len(ds) * 0.9), len(ds)))

print(f"Train: {len(train_ds)}  |  Eval: {len(eval_ds)}")

# Save locally
train_ds.to_json("train.jsonl")
eval_ds.to_json("eval.jsonl")

# Or push to HF Hub
ds.push_to_hub("your-username/healthcare-qa-synthetic", private=True)
```

### Approach 2: Curating from Existing Documents

If you have authoritative source documents (clinical guidelines, textbooks, SOPs), convert them into Q&A pairs.

```
SOURCE DOCUMENTS                     CONVERSION STRATEGY
────────────────────────────────────────────────────────────────────────
Clinical guidelines (WHO, CDC)       Extract recommendation → form Q&A
  "Adults 65+ should receive PCV20"  Q: What pneumococcal vaccine is
                                        recommended for adults over 65?
                                     A: Adults aged 65+ should receive
                                        PCV20 (Prevnar 20)...

Medical textbooks                    Turn chapter sections → Q&A
  Section on "Metformin MOA"         Q: How does metformin work?
                                     A: Metformin works primarily by...

Drug package inserts                 Extract side effects, dosing → Q&A
  "Common side effects: nausea,      Q: What are the side effects of X?
   headache, dizziness"              A: Common side effects include...

FAQ pages from hospitals             Already in Q&A format!
  Often need cleanup but format      Clean up → validate → use directly
  is already correct
```

**Semi-automated pipeline:**

```python
# Use an LLM to convert documents → Q&A pairs
prompt = """
Given the following medical guideline excerpt, generate 5 question-answer
pairs that a patient might ask and a doctor would answer.

Keep the answers faithful to the source — do not add information
that isn't in the excerpt.

Excerpt:
{excerpt}

Generate in JSON format: [{"question": "...", "answer": "..."}]
"""

# Feed each section/page of your source document
for excerpt in document_chunks:
    qa_pairs = generate_qa(prompt.format(excerpt=excerpt))
    # Validate that answers are grounded in the excerpt
    all_pairs.extend(qa_pairs)
```

### Approach 3: Expert Annotation

For high-stakes domains, have domain experts write or validate examples.

```
WORKFLOW:
  1. Generate draft Q&A pairs (synthetic or from docs)
  2. Domain expert REVIEWS each pair:
     ✅ Correct — keep as-is
     ✏️ Partially correct — expert edits the answer
     ❌ Wrong — expert rewrites or discards
  3. Expert adds edge cases the generator missed
  4. Final review for consistency

TOOLS FOR ANNOTATION:
  - Argilla (open-source, integrates with HF)     ← recommended
  - Label Studio (open-source, general-purpose)
  - Google Sheets (simple, works for small datasets)
  - Prodigy (commercial, very fast annotation UI)

COST ESTIMATE:
  Medical expert review: ~2-3 minutes per Q&A pair
  1,000 pairs × 2.5 min = ~42 hours of expert time
  At $50/hr = ~$2,100
  Worth it for production medical systems. Overkill for a workshop.
```

### Approach 4: From User Feedback / Production Logs

Once your system is live, real user interactions become the best training data.

```
PIPELINE:
  User asks question → Model responds → User gives feedback
                                         👍 good response
                                         👎 bad response + correction

  Collect (question, corrected_answer) pairs from 👎 feedback.
  These are gold-standard training examples: real questions,
  human-validated correct answers.

  This creates a FLYWHEEL:
  Deploy → Collect feedback → Fine-tune → Deploy better model → Repeat

PRACTICAL TIPS:
  - Log all interactions (with user consent + privacy compliance)
  - Track which responses users found helpful vs unhelpful
  - Focus on edge cases the model gets wrong
  - Retrain periodically (weekly/monthly) as new data accumulates
```

### Train / Eval / Test Splitting

Never train and evaluate on the same data.

```
TYPICAL SPLIT:
  Training:   80-90% of data  → model learns from this
  Evaluation:  5-10% of data  → used during training to track loss
  Test:        5-10% of data  → held out completely, used ONLY for final eval

OUR WORKSHOP SPLIT:
  Training:   2,000 examples
  Evaluation:   100 examples
  Test:         10 benchmark prompts (hand-crafted, NOT from the dataset)

WHY THE BENCHMARK PROMPTS ARE SEPARATE:
  The 10 benchmark prompts are NOT in the training data.
  They test GENERALIZATION — can the model handle questions it hasn't seen?
  If we tested on training data, we'd be measuring memorization, not learning.
```

### Dataset Validation Checklist

Before training, verify:

```
✅ FORMAT
   □ Every example has system/user/assistant messages
   □ No empty or whitespace-only fields
   □ No truncated answers (check for "..." at end)
   □ Chat template applies correctly (test with tokenizer)

✅ CONTENT QUALITY
   □ Answers are factually accurate (spot-check 20-30 randomly)
   □ Answers have appropriate length (not too terse, not rambling)
   □ No toxic, biased, or harmful content
   □ Medical disclaimers present where appropriate

✅ DIVERSITY
   □ Covers target topic breadth (not all diabetes, not all cardiology)
   □ Mix of question types (symptoms, treatments, mechanisms, comparisons)
   □ Varying complexity (simple patient questions to detailed clinical queries)
   □ No single question repeated more than 2-3 times in different phrasings

✅ DISTRIBUTION
   □ Train and eval sets have similar topic distributions
   □ No data leakage (eval examples not in train set)
   □ Benchmark prompts not present in training data
```

### Quick Reference: Dataset Size Guidelines

```
GOAL                                    RECOMMENDED SIZE
────────────────────────────────────────────────────────────────
Style/tone adaptation                   200-500 examples
  "Always respond in bullet points"     Model already knows the content,
                                        just needs format guidance.

Domain vocabulary shift                 500-1,000 examples
  "Use clinical terminology"            Model needs exposure to domain
                                        terms in context.

Domain knowledge + style                1,000-5,000 examples
  "Answer like a healthcare assistant"  Our use case. Enough for both
                                        style and content shift.

Deep specialization                     5,000-20,000 examples
  "Expert-level radiology reports"      Narrow domain requiring
                                        specialized knowledge.

Beyond 20,000 — diminishing returns. Consider full fine-tuning or
a larger base model instead of more data.
```

---

## 9. Chat Template Formatting

### Why Format Matters

Modern LLMs expect specific chat formats. Sending raw text wastes the instruction-tuning the model already has.

```
RAW TEXT (wrong):
  "Question: What is diabetes? Answer: Diabetes is..."
  The model sees this as plain text completion.
  It doesn't know where the instruction ends and the expected response begins.

CHAT FORMAT (correct):
  [
    {"role": "system", "content": "You are a healthcare assistant..."},
    {"role": "user", "content": "What is diabetes?"},
    {"role": "assistant", "content": "Diabetes is a chronic condition..."}
  ]
  The model uses its instruction-following training.
  It knows EXACTLY what role it plays and where to respond.
```

### Qwen2.5's Chat Template

```
<|im_start|>system
You are a helpful healthcare assistant...<|im_end|>
<|im_start|>user
What are the common symptoms of Type 2 diabetes?<|im_end|>
<|im_start|>assistant
The common symptoms of Type 2 diabetes include...<|im_end|>
```

The tokenizer's `apply_chat_template()` handles this automatically. **Never manually construct the template** — use the tokenizer method.

### Our Format Conversion

```python
def format_to_chat(example):
    """Convert raw QA pair to chat messages format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
    }
```

This transformation is applied to every example in the dataset. The SFTTrainer then applies the chat template during tokenization.

### The System Prompt's Role in Training

```
SYSTEM PROMPT WE USE:
"You are a helpful healthcare assistant. Provide accurate,
 evidence-based medical information. Always recommend consulting
 a healthcare professional for serious concerns."

THIS IS INCLUDED IN EVERY TRAINING EXAMPLE.

Why?
  The model learns to ASSOCIATE this system prompt with medical-quality responses.
  At inference time, we use the SAME system prompt → model activates the
  "healthcare assistant" behavior it learned during training.

  If you change the system prompt at inference → model may not behave as trained.
  If you omit the system prompt at inference → model reverts to generic behavior.
```

---

## 10. The Before/After Methodology

### Why Fixed Benchmarks?

```
THE PROBLEM WITH AD-HOC TESTING:
  Developer tests: "What is diabetes?" → "Looks better!"
  Another developer tests: "What causes headaches?" → "Seems about the same."
  Neither test is reproducible. Neither is comparable.

THE FIXED BENCHMARK APPROACH:
  Define 10 questions BEFORE training.
  Run them on the base model. Record outputs.
  Train the model.
  Run the SAME 10 questions. Record outputs.
  Compare side by side.

  Reproducible. Comparable. Objective.
```

### Our 10 Benchmark Prompts

```
These cover the breadth of healthcare topics:

 1. "What are the common symptoms of Type 2 diabetes?"         ← Symptom recognition
 2. "How does hypertension affect the heart over time?"        ← Pathophysiology
 3. "What is the recommended first-line treatment for          ← Treatment guidelines
     mild persistent asthma?"
 4. "Explain the difference between Type 1 and Type 2          ← Comparison/education
     diabetes."
 5. "What are the common side effects of ibuprofen?"           ← Pharmacology
 6. "How is pneumonia typically diagnosed?"                    ← Diagnostic process
 7. "What lifestyle changes help manage high cholesterol?"     ← Lifestyle medicine
 8. "What are the early warning signs of a stroke?"            ← Emergency recognition
 9. "How does metformin work for diabetes management?"         ← Drug mechanism
10. "What vaccinations are recommended for adults over 65?"    ← Preventive medicine
```

### What to Look For in Before/After Comparison

```
IMPROVEMENT SIGNALS:                      REGRESSION SIGNALS:
──────────────────                        ──────────────────
✅ More specific medical terminology       ❌ Generic filler language increased
✅ Structured responses (lists, sections)  ❌ Answers became shorter/vaguer
✅ Correct drug names and dosages          ❌ Made-up drug names or wrong dosages
✅ Includes "consult your doctor" caveat   ❌ Gives definitive medical advice
✅ Addresses the specific question asked   ❌ Rambles or goes off-topic
✅ Consistent tone across all 10 prompts   ❌ Inconsistent quality (some great, some bad)
```

---

## 11. Common Misconceptions

### ❌ Misconception 1: "Fine-tuning teaches the model new facts"

**Reality:** Fine-tuning shifts the model's **probability distribution** — it makes certain response patterns more likely. It doesn't reliably inject new factual knowledge. If the base model has never seen a specific drug interaction, fine-tuning on 100 examples mentioning it does NOT guarantee the model will learn it correctly. For factual recall, RAG is more reliable.

### ❌ Misconception 2: "More training data is always better"

**Reality:** After a certain point, more data causes **overfitting** or **catastrophic forgetting**. With LoRA, 500–2,000 high-quality examples is often sufficient for style and domain adaptation. Going to 50,000 examples rarely helps and often hurts — the model memorizes training data instead of generalizing.

### ❌ Misconception 3: "QLoRA quality is much worse than full fine-tuning"

**Reality:** Multiple papers (Dettmers et al., 2023) show QLoRA achieves **95–99%** of full fine-tuning quality. The quality gap is minimal, but the resource savings are enormous: 4 GB VRAM instead of 16+ GB. For most practical applications, you cannot tell the difference.

### ❌ Misconception 4: "I need thousands of GPU hours"

**Reality:** QLoRA fine-tuning of a small model on 1,000-2,000 examples takes **10–30 minutes on a free Colab T4**. The era of "fine-tuning requires a research lab" ended with LoRA and QLoRA. The bottleneck is dataset quality, not compute.

### ❌ Misconception 5: "Fine-tuning replaces prompt engineering"

**Reality:** Fine-tuning and prompt engineering serve different purposes. Fine-tuning changes the model's **default behavior**. Prompt engineering guides the model's **per-request behavior**. The best results come from a fine-tuned model with a well-crafted system prompt. If you remove the system prompt after fine-tuning, you lose the per-request control.

### ❌ Misconception 6: "The Instruct model doesn't need fine-tuning — it already follows instructions"

**Reality:** The Instruct model follows **general** instructions well. But it responds to "What is the overdraft fee?" the same way it responds to "What are the symptoms of diabetes?" — generically. Fine-tuning makes the model internalize your specific **domain vocabulary, response style, and quality standards**. It's the difference between a general assistant and a specialist.

### ❌ Misconception 7: "You need to fine-tune on a GPU you own"

**Reality:** Google Colab free tier provides a T4 GPU (15 GB VRAM) for free. It's more than sufficient for QLoRA on models up to 3B parameters. You don't need to buy, rent, or manage GPU infrastructure. For larger models (7B+), RunPod or Lambda offer on-demand A100s at ~$1-2/hour.

---

## 12. How Our Healthcare Agent Uses This

```
DECISION                             OUR CHOICE                  WHY
─────────────────────────────────────────────────────────────────────────────────
Approach                             Fine-tuning (QLoRA)         We need consistent
                                                                 healthcare tone +
                                                                 terminology natively

Base model (v1)                      Qwen2.5-1.5B-Instruct      Shows fine-tuning hurts
                                                                 when model is already capable

Base model (v2)                      Qwen2.5-1.5B-Instruct      Same model as v1 — isolates
                                                                 the effect of data quality

PEFT method                          QLoRA (4-bit NF4)           4 GB VRAM, 15-min training,
                                                                 ~10 MB adapter output

Dataset (v1)                         ChatDoctor-HealthCareMagic-100k  Shows noisy data degrades
                                                                      a capable model

Dataset (v2)                         jeev1992/wikidoc-healthassist    Reformatted WikiDoc — shows
                                                                      clean data preserves strengths

Training samples                     2,000 (+ 100 eval)          Enough for domain shift
                                                                 without overfitting

Chat template                        Qwen2 format via tokenizer  Matches the base model's
                                                                 instruction-tuning format

Benchmark                            10 fixed healthcare prompts Reproducible, covers
                                                                 breadth of medical topics

System prompt                        "You are a helpful          Included in EVERY training
                                     healthcare assistant..."    example and every inference
```

---

## 13. Full Workshop Pipeline Overview

This is the end-to-end pipeline we build across all four modules:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       FULL WORKSHOP PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  MODULE 1: STRATEGY & DATA (this module)                                │
│  ├── Decision: fine-tune with QLoRA on healthcare data                  │
│  ├── v1 Model: Qwen2.5-1.5B-Instruct                                   │
│  ├── v2 Model: Qwen2.5-1.5B-Instruct (same model, different data)         │
│  ├── v1 Dataset: ChatDoctor-HealthCareMagic-100k                           │
│  ├── v2 Dataset: jeev1992/wikidoc-healthassist (reformatted WikiDoc)       │
│  └── Benchmark: 10 fixed healthcare prompts                             │
│       │                                                                 │
│       ▼                                                                 │
│  MODULE 2: COLAB FINE-TUNING                                            │
│  ├── v1 (training_v1.ipynb): 1.5B + ChatDoctor → fine-tuning hurts        │
│  ├── v2 (training_v2.ipynb): 1.5B + reformatted WikiDoc → data wins       │
│  ├── Both: same model, same LoRA config, same hyperparameters           │
│  ├── BEFORE benchmark → Train → AFTER benchmark → Compare               │
│  └── Push adapter to Hugging Face Hub                                   │
│       │                                                                 │
│       ▼                                                                 │
│  MODULE 3: HF DEPLOY & INFERENCE                                        │
│  ├── Load adapter from HF Hub (clean environment)                       │
│  ├── Run base vs fine-tuned benchmark                                   │
│  ├── Demonstrate adapter toggle (disable/enable layers)                 │
│  └── Export results for evaluation                                      │
│       │                                                                 │
│       ▼                                                                 │
│  MODULE 4: LANGSMITH EVALUATION                                         │
│  ├── Create LangSmith dataset from benchmark prompts                    │
│  ├── Define evaluators (helpfulness, accuracy, safety)                  │
│  ├── Evaluate base model outputs → experiment 1                         │
│  ├── Evaluate fine-tuned outputs → experiment 2                         │
│  └── Compare experiments → final report                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| **Three approaches** | Prompt engineering → RAG → Fine-tuning. Try in this order. |
| **When to fine-tune** | Tone, terminology, format consistency, reduced token cost at scale. |
| **When NOT to** | Factual recall (use RAG), simple formatting (use prompts), low volume. |
| **Full FT vs PEFT** | PEFT (LoRA/QLoRA) achieves 95-99% quality at <1% of compute cost. |
| **QLoRA** | 4-bit quantized base + LoRA adapters. Fits on Colab T4 free tier. |
| **LoRA rank (r)** | Controls adapter capacity. r=16 is a good default. |
| **LoRA alpha** | Scaling factor. alpha = 2×r is the rule of thumb. |
| **Base model** | v1 & v2: Qwen2.5-1.5B-Instruct. Fits Colab T4. Same model isolates data quality variable. |
| **Dataset quality** | 1,000 clean examples > 10,000 noisy examples. Garbage in = garbage out. |
| **Custom datasets** | Synthetic (LLM-generated) is fastest. Always validate. Filter, deduplicate, and check. |
| **Chat template** | Always use `apply_chat_template()`. Never manually construct. |
| **Before/after** | 10 fixed prompts. Same prompts, base vs fine-tuned, side-by-side. |
| **System prompt** | Include in training AND inference. Must match. |

---

## 14. Hands-On Exercise

Open **[dataset_quality_exercise.ipynb](dataset_quality_exercise.ipynb)** for a hands-on exercise that puts this module's concepts into practice:

- **Part A — Dataset Quality Audit:** Load both datasets (ChatDoctor and WikiDoc), inspect samples, and quantify quality issues like persona contamination, boilerplate, and missing safety disclaimers. No GPU required.
- **Part B — Prompt Engineering Baseline:** Load the base Qwen 1.5B model and test 3 different system prompts (none → simple → detailed) on the same 10 medical questions you'll use in Module 2. Requires T4 GPU.
- **Part C — Go/No-Go Decision:** Fill in a decision scorecard based on your observations and predict how fine-tuning will turn out.

After this exercise, you'll already know the outcome of Module 2 before running a single training step.

---

*Next: [Module 2 — Colab Fine-Tuning with QLoRA →](../module_2_colab_finetuning/notes.md)*
