# QLoRA Fine-Tuning on Google Colab
## Your Complete Guide to Fine-Tuning a Healthcare Model with LoRA

---

## Table of Contents

1. [The Training Pipeline at a Glance](#1-the-training-pipeline-at-a-glance)
2. [Google Colab Environment Setup](#2-google-colab-environment-setup)
3. [Loading a Model in 4-bit with BitsAndBytes](#3-loading-a-model-in-4-bit-with-bitsandbytes)
4. [Understanding the Tokenizer](#4-understanding-the-tokenizer)
5. [Preparing the Training Dataset](#5-preparing-the-training-dataset)
6. [Configuring LoRA Adapters](#6-configuring-lora-adapters)
7. [SFTTrainer and Training Configuration](#7-sfttrainer-and-training-configuration)
8. [Training Hyperparameters Deep Dive](#8-training-hyperparameters-deep-dive)
9. [Monitoring Training (Loss, Overfitting, Evaluation)](#9-monitoring-training-loss-overfitting-evaluation)
10. [Before vs After Benchmarking](#10-before-vs-after-benchmarking)
11. [Saving and Pushing the Adapter](#11-saving-and-pushing-the-adapter)
12. [VRAM Budget and Memory Management](#12-vram-budget-and-memory-management)
13. [Common Misconceptions](#13-common-misconceptions)
14. [Quick Reference: What You Used and Why](#14-quick-reference-what-you-used-and-why)
15. [Key Lessons — What You Learned](#15-key-lessons--what-you-learned)

---

## 1. The Training Pipeline at a Glance

```
Here's the big picture of what you'll do in each notebook:

STEP 1: CHECK YOUR GPU
  You need a GPU (like Colab's T4 with 15 GB of memory).
  No GPU = no training. Always verify this first.

STEP 2: INSTALL LIBRARIES
  pip install transformers, peft, trl, bitsandbytes, accelerate, datasets
  These are the tools that make fine-tuning possible.

STEP 3: LOAD THE MODEL
  Download the pre-trained model from Hugging Face and load it onto your GPU.
  The model weights are frozen — they won't change during training.

STEP 4: TEST BEFORE TRAINING ("BEFORE" benchmark)
  Ask the model 10 medical questions and save the answers.
  This is your baseline — what the model knows BEFORE you train it.

STEP 5: ATTACH LoRA ADAPTERS
  Add tiny trainable layers on top of the frozen model.
  Only these adapters will be updated (~1% of all parameters).

STEP 6: TRAIN
  Feed the model your training data (patient questions + doctor answers).
  It learns to predict the doctor's answer, updating only the adapters.

STEP 7: TEST AFTER TRAINING ("AFTER" benchmark)
  Ask the same 10 questions and compare to the "before" answers.
  Did the model improve, stay the same, or get worse?

STEP 8: UPLOAD TO HUGGING FACE HUB
  Push your ~10-50 MB adapter so you can use it from anywhere.
```

---

## 2. Google Colab Environment Setup

### Checking GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

Expected output:
```
CUDA available: True
GPU: Tesla T4
VRAM: 15.8 GB
```

### If There's No GPU

```
Runtime → Change runtime type → Hardware accelerator → T4 GPU

If T4 is not available (usage limits):
  1. Try again in a few hours (Colab rotates GPUs)
  2. Use Colab Pro ($10/month) for priority access
  3. Use a paid GPU service (RunPod, Lambda)
```

### The Dependency Stack

```
PACKAGE             VERSION    WHY WE NEED IT
─────────────────────────────────────────────────────────────
torch               >=2.6.0    PyTorch — newer transformers needs torch.distributed.tensor internals
torchvision                    Must match torch version (transformers imports it for image ops)
transformers        >=4.45.0   Model loading, tokenizer, generation
accelerate          >=0.34.0   Efficient model dispatch across devices
peft                >=0.12.0   LoRA/QLoRA adapter management
trl                 >=0.10.0   SFTTrainer for supervised fine-tuning
bitsandbytes        >=0.43.0   4-bit quantization (NF4)
datasets            >=2.20.0   Load HF datasets
huggingface-hub     >=0.24.0   Push model to Hub
pandas                         Comparison tables
```

**Version pinning matters.** The QLoRA ecosystem evolves fast. A version mismatch between `transformers` and `peft` is the #1 cause of cryptic errors.

```python
!pip install -q \
    "torch>=2.6.0" \
    torchvision \
    "transformers>=4.45.0" \
    "accelerate>=0.34.0" \
    "peft>=0.12.0" \
    "trl>=0.10.0" \
    "bitsandbytes>=0.43.0" \
    "datasets>=2.20.0" \
    "huggingface-hub>=0.24.0" \
    pandas

# Restart the kernel so newly installed packages are importable
import IPython
IPython.Application.instance().kernel.do_shutdown(restart=True)
```

---

## 3. Loading a Model in 4-bit with BitsAndBytes

### The BitsAndBytesConfig

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Quantize weights to 4-bit
    bnb_4bit_quant_type="nf4",            # Normal Float 4-bit (optimal for NN weights)
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in bf16 for speed
    bnb_4bit_use_double_quant=True,       # Quantize the quantization constants too
)
```

### What Each Parameter Does

```
load_in_4bit=True
  ┌─────────────────────────────────────────────────┐
  │ Without: model loads in fp16 → 1.5B × 2 = 3 GB   │
  │ With:    model loads in 4-bit → 1.5B × 0.5 ≈ 0.75 GB│
  │ Savings: ~2 GB VRAM (67% reduction)              │
  └─────────────────────────────────────────────────┘

bnb_4bit_quant_type="nf4"
  ┌─────────────────────────────────────────────────┐
  │ "nf4" = Normal Float 4-bit                       │
  │ Optimized for neural network weights that        │
  │ follow a normal distribution (bell curve).       │
  │ Better than uniform 4-bit for model performance. │
  │                                                  │
  │ Alternative: "fp4" (standard 4-bit float)        │
  │ NF4 is almost always better. Use it.             │
  └─────────────────────────────────────────────────┘

bnb_4bit_compute_dtype=torch.bfloat16
  ┌─────────────────────────────────────────────────┐
  │ Weights stored in 4-bit.                         │
  │ But COMPUTATION happens in bfloat16.             │
  │                                                  │
  │ Flow: 4-bit weight → dequantize to bf16 →        │
  │       compute in bf16 → result in bf16           │
  │                                                  │
  │ bf16 vs fp16: bf16 has larger range, less risk   │
  │ of overflow during training. Preferred on T4.    │
  └─────────────────────────────────────────────────┘

bnb_4bit_use_double_quant=True
  ┌─────────────────────────────────────────────────┐
  │ Quantization uses "constants" (scaling factors)  │
  │ per block of weights. These constants are fp32.  │
  │                                                  │
  │ Double quant = quantize these constants too.     │
  │ Saves ~0.1 GB on a 1.5B model.                  │
  │ No measurable quality impact.                    │
  └─────────────────────────────────────────────────┘
```

### Loading the Model

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",           # Automatically place layers on GPU
)
```

### What `device_map="auto"` Does

```
WITHOUT device_map:
  You must manually .to("cuda") each part of the model.
  If the model doesn't fit, you get an OOM error.

WITH device_map="auto":
  Accelerate analyzes available devices (GPU, CPU, disk).
  It places as much as possible on GPU.
  Overflow goes to CPU RAM (slower but works).

  For our 1.5B model in 4-bit (~0.75 GB), everything fits on GPU.
  device_map is more important for 7B+ models.
```

### VRAM After Loading

```
After loading a model in 4-bit:
  Model weights: ~1.0 GB
  CUDA overhead: ~0.5 GB
  Available:     ~14.3 GB of 15.8 GB

  Plenty of room for training.
```

---

## 4. Understanding the Tokenizer

### Loading and Configuring

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token    # CRITICAL
tokenizer.padding_side = "right"              # CRITICAL for causal LM
```

### Why `pad_token = eos_token`?

```
PROBLEM:
  Many models (including Qwen) don't have a dedicated padding token.
  Training requires batching. Batching requires padding.
  Without a pad token → crash during training.

SOLUTION:
  Set pad_token to eos_token (end-of-sequence token).
  The model already knows to stop at eos_token.
  When used as padding, it's masked out in the loss calculation.

  This is standard practice. Every QLoRA tutorial does this.
```

### Why `padding_side = "right"`?

```
CAUSAL LANGUAGE MODELS generate left-to-right.
They attend to all tokens to the LEFT.

LEFT PADDING (wrong for training):
  [PAD][PAD][PAD] The patient has diabetes
  ↑ The model "sees" padding tokens at the start.
  This can confuse attention patterns during training.

RIGHT PADDING (correct for training):
  The patient has diabetes [PAD][PAD][PAD]
  ↑ The model processes real tokens first.
  Padding at the end is masked out in loss calculation.
```

### The Chat Template

```python
messages = [
    {"role": "system", "content": "You are a helpful healthcare assistant..."},
    {"role": "user", "content": "What is diabetes?"},
    {"role": "assistant", "content": "Diabetes is a chronic condition..."},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,           # Return string, not token IDs
    add_generation_prompt=True # Add the assistant turn marker at the end
)
```

Output:
```
<|im_start|>system
You are a helpful healthcare assistant...<|im_end|>
<|im_start|>user
What is diabetes?<|im_end|>
<|im_start|>assistant
Diabetes is a chronic condition...<|im_end|>
```

**At training time:** We include the assistant's response. The model learns to generate it.  
**At inference time:** We omit the assistant's response and set `add_generation_prompt=True`. The model generates its own.

---

## 5. Preparing the Training Dataset

### The Two Notebooks (v1 vs v2)

The two notebooks isolate two variables — model capability and dataset quality:

```
v1 (training_v1.ipynb): Qwen2.5-1.5B-Instruct + ChatDoctor-HealthCareMagic-100k
  - Base model already excellent → fine-tuning degrades quality (adds persona)
  - You'll learn: don't fine-tune models that are already good at your task

v2 (training_v2.ipynb): Qwen2.5-1.5B-Instruct + jeev1992/wikidoc-healthassist (reformatted WikiDoc)
  - Same capable base model, but trained on clean reformatted data
  - WikiDoc data reformatted via GPT-4o-mini into conversational healthcare-assistant style
  - You'll learn: data quality is everything — clean data preserves model strengths
```

**Why different datasets?**
v1 uses ChatDoctor (messy chat logs) to show what happens when you fine-tune on noisy data.
v2 uses WikiDoc data that was reformatted via GPT-4o-mini (data_prep_v2.py) into
conversational healthcare-assistant style. The reformatted dataset is pre-uploaded
to jeev1992/wikidoc-healthassist on Hugging Face.

### Measured Dataset Quality (from Module 1 Exercise)

```
METRIC                              CHATDOCTOR (v1)    WIKIDOC (v2)
──────────────────────────────────────────────────────────────────
Total examples                         112,165           2,100
Persona contamination                    63.1%            0.0%
Boilerplate sign-offs                    28.2%            0.0%
Very short answers (<50ch)                0.6%            0.0%
Safety disclaimers                        3.2%           99.4%
Avg answer length (chars)                  603             910
```

63% persona contamination means the model WILL learn to say "Hi, welcome to
Chat Doctor" — which is exactly what we see in v1 evaluation results.
WikiDoc's 99.4% safety disclaimer rate teaches the model to always recommend
consulting a healthcare professional.

### Loading from Hugging Face

```python
from datasets import load_dataset

# v1: ChatDoctor — doctor-patient chat logs, noisy, has persona artifacts
v1_dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")

# v2: WikiDoc reformatted — pre-processed conversational data from HF
v2_dataset = load_dataset("jeev1992/wikidoc-healthassist", split="train")
```

### WikiDoc Column Structure

```
medalpaca/medical_meadow_wikidoc columns:
  input       → the medical question (e.g. "What causes Type 2 diabetes?")
  output      → clean encyclopedic answer (100–600+ chars, no greetings)
  instruction → constant string "Answer this question truthfully" — ignored
```

### What the Raw Data Actually Looks Like

**ChatDoctor sample (lavita/ChatDoctor-HealthCareMagic-100k):**
```
input:  "Hello doctor, I am a 25 year old male. I have been having chest pain
         for the past 2 days. It is worse when I breathe in deeply."

output: "Hello. Welcome to Chat Doctor. I understand your concern about chest pain.
         Chest pain can be caused by many things, including heart problems, lung
         problems, or musculoskeletal problems. I would recommend that you see a
         doctor as soon as possible to get a proper diagnosis. In the meantime,
         you can take over-the-counter pain medication such as ibuprofen or
         acetaminophen to help relieve the pain. Hope I have answered your query.
         Let me know if I can assist you further. Wishing you good health."
```

Notice: greeting, sign-off, "Chat Doctor" persona, casual tone. This style
bakes into every fine-tuned output regardless of what question you ask.

---

**WikiDoc sample (medalpaca/medical_meadow_wikidoc):**
```
instruction: "Answer this question truthfully"   ← constant, ignored

input:  "What are the causes of myocardial infarction?"

output: "Myocardial infarction may be caused by atherosclerosis, the buildup of
         lipid-laden plaques in the coronary arteries. Rupture of these plaques
         leads to thrombus formation, which can partially or completely occlude
         the artery. Other causes include coronary artery spasm, spontaneous
         coronary artery dissection (SCAD), and emboli from the left heart.
         Risk factors include hypertension, hyperlipidemia, diabetes mellitus,
         smoking, obesity, and family history of coronary artery disease."
```

Notice: no greeting, no sign-off, no persona. Encyclopedic, factual, structured.
But it's terse and clinical — not the conversational style we want our assistant to use.

---

**Reformatted WikiDoc sample (jeev1992/wikidoc-healthassist — actual v2 training data):**

This is what `data_prep_v2.py` produces. GPT-4o-mini takes the raw WikiDoc entry above
and reformats it into conversational healthcare-assistant style:

```
messages: [
  {
    "role": "system",
    "content": "You are a knowledgeable and thorough healthcare assistant..."
  },
  {
    "role": "user",
    "content": "What are the causes of myocardial infarction?"
  },
  {
    "role": "assistant",
    "content": "A myocardial infarction (heart attack) occurs when blood flow to part
         of the heart muscle is blocked. Here are the main causes and risk factors:

         **Primary Cause — Coronary Artery Disease:**
         The most common cause is atherosclerosis, where fatty deposits (plaques)
         build up inside the coronary arteries over time. When a plaque ruptures,
         it triggers a blood clot (thrombus) that can partially or completely block
         the artery, cutting off oxygen to the heart muscle.

         **Other Causes:**
         - Coronary artery spasm — sudden tightening of the artery wall
         - Spontaneous coronary artery dissection (SCAD) — a tear in the artery wall
         - Blood clots (emboli) traveling from elsewhere in the heart

         **Key Risk Factors:**
         - High blood pressure (hypertension)
         - High cholesterol (hyperlipidemia)
         - Diabetes mellitus
         - Smoking
         - Obesity and sedentary lifestyle
         - Family history of coronary artery disease

         If you experience chest pain, shortness of breath, or other symptoms
         of a heart attack, seek emergency medical care immediately. Early
         treatment significantly improves outcomes. Please consult a healthcare
         professional for personalized cardiac risk assessment."
  }
]
```

Notice the transformation: same medical facts, but now conversational, structured with
headers and bullet points, and includes a safety disclaimer. This is the style the
fine-tuned model learns to produce — helpful, organized, and responsible.

---

### The Format Conversion

```python
SYSTEM_PROMPT = (
    "You are a knowledgeable and thorough healthcare assistant. "
    "When answering medical questions, provide comprehensive explanations "
    "with relevant clinical details, mechanisms of action, and practical guidance. "
    "Structure your answers clearly. "
    "Always recommend consulting a healthcare professional for serious concerns."
)

# v2: Reformatted WikiDoc — already in chat format from jeev1992/wikidoc-healthassist
def format_to_chat(example):
    question = str(example.get("input")  or "").strip()
    answer   = str(example.get("output") or "").strip()
    if len(question) < 5 or len(answer) < 20:
        return {"messages": []}   # empty list = invalid, filtered below
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]
    }

formatted = raw_ds.shuffle(seed=42).map(format_to_chat, remove_columns=raw_ds.column_names)
formatted = formatted.filter(lambda x: len(x["messages"]) > 0)
```

> **Why `len(x["messages"]) > 0` instead of `x["messages"] is not None`?**
> HuggingFace datasets handles empty lists cleanly in filter. Using `None` as a
> sentinel in a structured column can cause schema inference issues that silently
> drop all rows.

### Why This Specific Format?

```
SFTTrainer EXPECTS one of two formats:

FORMAT A: "text" field (raw string with chat template already applied)
FORMAT B: "messages" field (list of role/content dicts)  ← WE USE THIS

When using FORMAT B:
  SFTTrainer automatically applies the tokenizer's chat template.
  You don't need to call apply_chat_template() yourself for training.
  The trainer handles tokenization, padding, and masking.
```

### Shuffling and Subsampling

```python
dataset = dataset.shuffle(seed=42)

train_dataset = dataset.select(range(2000))        # First 2,000 for training
eval_dataset  = dataset.select(range(2000, 2100))  # Next 100 for evaluation
```

```
WHY SHUFFLE?
  The original dataset may be ordered by topic.
  Without shuffling: first 2,000 = all anatomy, no pharmacology.
  With shuffling: 2,000 random samples covering all topics.

WHY SEED=42?
  Reproducibility. Same shuffle every time.
  You can rerun and get identical results every time.

WHY 2,000 TRAINING + 100 EVAL?
  2,000 is enough for domain adaptation with LoRA.
  100 eval samples for tracking overfitting during training.
  3 epochs × 2,000 = 6,000 training steps (effective).
```

---

## 6. Configuring LoRA Adapters

### The LoRA Config

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
```

### What Each LoRA Parameter Does

#### `r=16` (Rank — the adapter's learning capacity)

```
WHAT IS RANK?
  Think of rank as how many "notes" the adapter can take during training.

  LoRA works by adding two small matrices (A and B) next to each frozen layer.
  Instead of updating the full 2048×2048 weight matrix (4 million numbers),
  LoRA decomposes the update into two smaller matrices:
    A: 2048 × 16  (32,768 numbers)
    B: 16 × 2048  (32,768 numbers)

  The "16" is the rank — it's the bottleneck between A and B.

ANALOGY:
  Imagine you're summarizing a 500-page medical textbook.
  r=4:   You can write a 4-bullet summary → captures broad themes only
  r=16:  You can write a 16-bullet summary → captures key details
  r=64:  You can write a 64-bullet summary → captures almost everything
  r=2048: You rewrite the whole book → same as full fine-tuning (no savings)

  Higher rank = more capacity to learn, but more parameters to train.

PRACTICAL GUIDE:
  r=4:   Minimum. Good for simple tasks (style change, formatting)
  r=8:   Common. Works for most domain adaptation
  r=16:  Our choice. Good balance of capacity vs efficiency
  r=32:  High capacity. Use when r=16 underperforms
  r=64+: Rarely needed for LoRA. Diminishing returns.

  For our 1.5B model with r=16: ~9M trainable params (~0.6% of total)
  With r=8 that would be ~4.5M, with r=32 about ~18M
```

#### `lora_alpha=32` (Scaling Factor — how much the adapter influences output)

```
WHAT IS ALPHA?
  Alpha controls how strongly the adapter's learned changes affect the output.
  The adapter's contribution is scaled by (alpha / r) before being added
  to the frozen layer's output.

  With alpha=32 and r=16:  scaling = 32/16 = 2.0
  With alpha=16 and r=16:  scaling = 16/16 = 1.0
  With alpha=8  and r=16:  scaling =  8/16 = 0.5

ANALOGY:
  Think of a volume knob for the adapter:
  scaling = 0.5 → adapter whispers (small changes)
  scaling = 1.0 → adapter speaks normally
  scaling = 2.0 → adapter speaks louder (our choice — adapter changes
                   have 2× influence on the output)

WHY alpha=2×r?
  This is the most common convention. Setting alpha = 2 × r means
  the adapter's learned updates are amplified by 2×.

  This works well because LoRA's A and B matrices are initialized
  to produce very small outputs — the 2× scaling compensates so
  the adapter can actually make meaningful changes during training.

  Rule of thumb: start with alpha = 2 × r. Only change if you see
  training instability (lower alpha) or the adapter barely changes
  outputs (raise alpha).
```

#### `lora_dropout=0.05` (Dropout — prevents memorization)

```
WHAT IS DROPOUT?
  During training, dropout RANDOMLY turns off 5% of the adapter's
  neurons on each step. Different neurons are turned off each time.

  Step 1: neurons [1,2,_,4,5,_,7,8,9,10,11,_,13,14,15,16]  (3,6,12 off)
  Step 2: neurons [_,2,3,4,_,6,7,8,_,10,11,12,13,14,15,16]  (1,5,9 off)
  Step 3: neurons [1,2,3,_,5,6,7,_,9,10,_,12,13,14,15,16]  (4,8,11 off)

WHY?
  Without dropout, the adapter might rely too heavily on specific neurons.
  It memorizes training examples instead of learning general patterns.

  With dropout, no single neuron can be relied upon — the adapter must
  spread knowledge across all neurons. This helps it generalize to
  new questions it hasn't seen before.

ANALOGY:
  Studying for an exam with a group, but each day a random person is absent.
  The group can't rely on any one person — everyone must understand every topic.
  Result: the whole group is stronger on exam day.

PRACTICAL GUIDE:
  0.0:   No dropout. Fine for very small datasets where you need max capacity.
  0.05:  Our choice. Light regularization. Standard for LoRA.
  0.1:   Moderate. Use if you see overfitting (eval loss increasing).
  0.2+:  Aggressive. Rarely needed for LoRA — already a lightweight method.

  At inference time, dropout is turned OFF — all neurons are active.
```

#### `target_modules="all-linear"` (Which layers get adapters)

```
WHAT ARE TARGET MODULES?
  A transformer model is made of repeating "layers" (28 in Qwen 1.5B).
  Each layer contains several linear (matrix multiplication) operations:

  ATTENTION (how the model focuses on different parts of the input):
    q_proj  → "query" — what am I looking for?
    k_proj  → "key"   — what do I contain?
    v_proj  → "value" — what information do I carry?
    o_proj  → "output" — combine the attention results

  MLP (how the model processes information):
    gate_proj → controls information flow
    up_proj   → expands representation
    down_proj → compresses back down

  "all-linear" = attach LoRA adapters to ALL 7 of these per layer.
  That's 7 × 28 layers = 196 adapter pairs total.

ALTERNATIVES:
  target_modules=["q_proj", "v_proj"]
    Only adapts attention queries and values (2 per layer)
    Fewer parameters, faster training
    Good enough for many simple tasks

  target_modules="all-linear"  (our choice)
    Adapts all 7 linear modules per layer
    Maximum capacity — adapter can change how the model
    both attends to input AND processes information
    Best quality, slightly slower training

WHY "all-linear"?
  For medical domain adaptation, we want the model to learn both
  new attention patterns (focus on medical terms) and new processing
  (generate medical explanations). "all-linear" covers both.
```

#### `task_type="CAUSAL_LM"` (What kind of model we're training)

```
WHAT IS CAUSAL LM?
  "Causal Language Model" = the model generates text left-to-right,
  one token at a time. Each new token can only "see" tokens before it.

  Example:  "The patient has ___"
  The model predicts the next word based only on "The patient has".
  It cannot peek at future words.

  This is how GPT, Qwen, LLaMA, and all chat models work.

OTHER TASK TYPES (not used here):
  SEQ_2_SEQ_LM  → encoder-decoder models (T5, BART) — for translation, summarization
  TOKEN_CLS     → token classification (NER, POS tagging)
  SEQ_CLS       → sequence classification (sentiment, spam detection)

  We set CAUSAL_LM because Qwen is a causal language model.
  This tells PEFT how to set up the adapter's loss function correctly.
```

> **TRL 0.29+ Note:** In earlier TRL versions, you had to manually call
> `prepare_model_for_kbit_training(model)` and `get_peft_model(model, lora_config)`.
> In TRL 0.29+, you pass `peft_config=lora_config` to SFTTrainer and it handles
> both steps automatically. Our notebooks use the new approach.

### What Happens Under the Hood (SFTTrainer does this for you)

```
When SFTTrainer receives peft_config, it internally:
  1. Calls prepare_model_for_kbit_training(model)
     - Enables gradient checkpointing
     - Sets up mixed precision (4-bit storage, bf16 compute)
     - Marks quantized params as frozen
     - Allows LoRA adapter params to receive gradients
  2. Calls get_peft_model(model, lora_config)
     - Wraps target layers with LoRA adapters
     - Creates trainable A×B matrices
```

### What `get_peft_model()` Does

```
BEFORE get_peft_model():
  model.q_proj = Linear(2048, 2048)     ← 4M params, frozen

AFTER get_peft_model():
  model.q_proj = LoRALinear(
      base = Linear(2048, 2048)          ← 4M params, FROZEN
      lora_A = Linear(2048, 16)          ← 32K params, TRAINABLE
      lora_B = Linear(16, 2048)          ← 32K params, TRAINABLE
  )

  Forward pass:
    output = base(x) + (lora_B(lora_A(x))) × (alpha/r)
           = frozen_output + adapter_output × 2.0
```

### Trainable Parameter Count

```
With target_modules="all-linear":
  Attention: q_proj, k_proj, v_proj, o_proj (4 modules)
  MLP: gate_proj, up_proj, down_proj (3 modules)
  Per layer: 7 adapter pairs
  Total layers: 24 (Qwen2.5-1.5B)
  Total adapter pairs: 7 × 28 = 196
  Each pair: A (d × r) + B (r × d) = 2 × d × r parameters

  For d=1536 (Qwen2.5-1.5B hidden size), r=16:
    Per pair: 2 × 1536 × 16 = 49,152
    Total: 196 × 49,152 ≈ 9.6M params
    
  Plus some layers with different dimensions → total ~9M trainable
  Out of 1,543M total → 0.95% trainable
```

---

## 7. SFTTrainer and Training Configuration

### What Is SFTTrainer?

```
REGULAR Trainer (HF Transformers):
  Expects tokenized input_ids and labels
  You must tokenize, pad, and create labels yourself
  General-purpose — works for any task

SFTTrainer (TRL library):
  Specialized for Supervised Fine-Tuning
  Accepts "messages" format directly
  Applies chat template automatically
  Handles tokenization, padding, and loss masking
  Built-in PEFT/LoRA support

  SFTTrainer = Trainer + chat formatting + QLoRA integration
```

### The Training Code

> **TRL 0.29+ API changes:** The code below reflects TRL 0.29+ which our notebooks use.
> Key differences from older tutorials: `processing_class` instead of `tokenizer`,
> `peft_config` passed to SFTTrainer instead of manual `get_peft_model()`, and
> `warmup_steps` instead of `warmup_ratio`. `max_seq_length` is set via
> `tokenizer.model_max_length` instead of inside SFTConfig.

```python
from trl import SFTTrainer, SFTConfig

# TRL 0.29+ removed max_seq_length from SFTConfig.
# Set it on the tokenizer instead:
tokenizer.model_max_length = MAX_SEQ_LENGTH  # 512

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=38,           # ~10% of 375 total steps
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    seed=42,
    report_to="none",
)

# In TRL 0.29+, pass peft_config to SFTTrainer —
# it handles prepare_model_for_kbit_training + get_peft_model internally.
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,   # TRL 0.29+: "processing_class" not "tokenizer"
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=sft_config,
    peft_config=lora_config,      # SFTTrainer attaches LoRA adapters for you
)

trainer.train()
```

---

## 8. Training Hyperparameters Deep Dive

Every hyperparameter has a purpose. Here's what each one does and why we chose our values:

### Epochs (`num_train_epochs=3`)

```
1 EPOCH = one full pass through all 2,000 training examples

EPOCH 1: Model sees every example once  → large loss drops
EPOCH 2: Model sees every example again → smaller improvements
EPOCH 3: Model sees every example again → fine-tuning the fine-tuning

WHY 3?
  1 epoch:  Underfitting risk. Model barely changed.
  3 epochs: Good balance. Enough repetition to learn the pattern.
  5+ epochs: Overfitting risk. Model memorizes training data.

  For LoRA with 2,000 examples, 2-4 epochs is the sweet spot.
```

### Batch Size and Gradient Accumulation

```
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
EFFECTIVE BATCH SIZE = 2 × 4 = 8

WHY NOT batch_size=8 DIRECTLY?
  batch_size=8 requires 4× the VRAM of batch_size=2.
  On Colab T4, batch_size=8 may cause OOM (Out of Memory).

  Gradient accumulation simulates a larger batch:
    Step 1: Forward on 2 examples, compute gradients, DON'T update weights
    Step 2: Forward on 2 more, accumulate gradients, DON'T update weights
    Step 3: Forward on 2 more, accumulate gradients, DON'T update weights
    Step 4: Forward on 2 more, accumulate gradients, UPDATE WEIGHTS

  4 mini-batches × 2 = 8 examples before one weight update.
  Same math as batch_size=8, but only 2 examples in VRAM at a time.

EFFECTIVE BATCH SIZE MATTERS:
  batch_size=1:  Very noisy gradients. Training is unstable.
  batch_size=8:  Smoother gradients. More stable learning.
  batch_size=32: Very smooth but may generalize worse (sharp minima).

  8 is a good default for QLoRA fine-tuning.
```

### Learning Rate (`learning_rate=2e-4`)

```
LEARNING RATE = how big each weight update step is

TOO HIGH (1e-3):
  ┌───────────────────────────┐
  │ Loss ╲    ╱╲   ╱╲        │  Oscillates. Never converges.
  │       ╲╱   ╲╱   ╲ ???    │  Model overshoots the optimal point.
  └───────────────────────────┘

TOO LOW (1e-6):
  ┌───────────────────────────┐
  │ Loss ─────────────────    │  Barely moves. Wastes GPU time.
  │                           │  Model doesn't learn anything useful.
  └───────────────────────────┘

JUST RIGHT (2e-4):
  ┌───────────────────────────┐
  │ Loss ╲                    │  Steady decrease. Converges smoothly.
  │       ╲                   │  Standard for LoRA fine-tuning.
  │        ╲____              │
  └───────────────────────────┘

WHY 2e-4 SPECIFICALLY?
  This is the "standard" learning rate for LoRA fine-tuning.
  The LoRA paper (Hu et al., 2021) used this value.
  Almost every QLoRA tutorial uses 1e-4 to 3e-4.
  2e-4 is the safe middle ground.
```

### Learning Rate Scheduler (`lr_scheduler_type="cosine"`)

```
CONSTANT:
  LR ─────────────────────
  Same learning rate the entire time.
  Simple but suboptimal.

LINEAR DECAY:
  LR ╲
      ╲
       ╲
        ╲___
  Starts high, decreases linearly to 0.
  Fine, but abrupt transition.

COSINE (what we use):
  LR ╲
      ╲
       ╲
        ╲___
  Starts high, follows a cosine curve to 0.
  Smoother transition. Better final quality.
  Most popular for fine-tuning.

The key insight: early training benefits from large steps (explore the loss landscape).
Late training benefits from small steps (fine-tune to the optimal point).
Cosine achieves this naturally.
```

### Warmup (`warmup_steps`)

```
WITHOUT WARMUP:
  Step 1: lr=2e-4 (full speed immediately)
  The model hasn't calibrated its gradients yet.
  Large, random weight updates → instability.

WITH WARMUP:
  v1 uses warmup_steps=75 (~10% of 750 total steps)
  v2 uses warmup_steps=50 (~10% of 500 total steps)

  During warmup, the learning rate gradually increases from 0 to the target:
    Step 1:  lr ≈ 0.000003  (almost zero)
    Step 25: lr ≈ 0.000067  (one-third of target)
    Step 50: lr ≈ 0.000133  (two-thirds)
    Step 75: lr = 0.000200  (full target — v1)

  After warmup, the cosine scheduler takes over and gradually decreases lr to 0.

ANALOGY:
  Warming up before exercise. You don't sprint immediately —
  you start with a light jog so your muscles adjust. Same idea:
  the model needs a few steps to calibrate its gradients before
  receiving full-strength weight updates.

  The first warmup steps are gentle. The model finds its footing.
  Then full learning rate kicks in for the real training.

warmup_steps vs warmup_ratio:
  warmup_steps=75  → exactly 75 warmup steps (what our notebooks use)
  warmup_ratio=0.1 → 10% of total steps are warmup (alternative way to specify)
  Both achieve the same thing — we use warmup_steps for explicit control.
```

### Max Sequence Length (`tokenizer.model_max_length=512`)

```
TRL 0.29+ CHANGE:
  OLD: max_seq_length=512 inside SFTConfig (removed in TRL 0.29+)
  NEW: tokenizer.model_max_length = 512 (set on the tokenizer directly)

  The effect is the same: examples longer than 512 tokens get truncated.

WHAT IT CONTROLS:
  Maximum number of tokens per training example.
  Longer examples are truncated to this length.

OUR DATASET:
  Most medical QA pairs: 100-400 tokens (with chat template)
  512 tokens covers 95%+ of examples without truncation.

WHY NOT LONGER?
  512:  Each example uses ~512 × 2 bytes (bf16) = 1 KB activation memory
  1024: Each example uses ~1024 × 2 = 2 KB → 2× VRAM for activations
  2048: 4× VRAM → may not fit on T4 with batch_size=2

  Shorter max_seq_length = more VRAM headroom = can use larger batch sizes.
  Only increase if your data actually needs it.
```

### bf16 Training (`bf16=True`)

```
fp32: 4 bytes per value, full precision        → 2× VRAM
fp16: 2 bytes per value, limited range          → Risk of overflow
bf16: 2 bytes per value, wide range (bf16)      → Best of both worlds

bfloat16 (bf16):
  Same size as fp16 (2 bytes)
  But uses the exponent range of fp32
  Less precision for small numbers, but MUCH less risk of NaN/overflow

  T4 supports bf16 natively.
  All modern training uses bf16 by default.
```

### Logging Steps (`logging_steps=10`)

```
WHAT IT DOES:
  Print the training loss every 10 steps.

  Step 10:  Training Loss = 2.14
  Step 20:  Training Loss = 1.95
  Step 30:  Training Loss = 1.82
  ...

WHY 10?
  Too frequent (1):  Clutters output. Loss is noisy step-to-step.
  Too infrequent (100): You can't tell if training is working until
                         minutes have passed. If something is broken,
                         you waste GPU time.
  10 is a good middle ground — frequent enough to catch problems early,
  rare enough to keep your output readable.

  You'll see ~50-75 log lines during a full training run.
  That's enough to spot trends without drowning in numbers.
```

### Checkpoint Saving (`save_strategy="steps"`, `save_steps=100`)

```
WHAT IT DOES:
  Save a snapshot of the adapter weights every 100 training steps.

  healthcare-assistant-lora/
  ├── checkpoint-100/    ← adapter weights at step 100
  ├── checkpoint-200/    ← adapter weights at step 200
  ├── checkpoint-300/    ← adapter weights at step 300
  └── ...

WHY SAVE CHECKPOINTS?
  1. CRASH RECOVERY: If Colab disconnects mid-training, you don't lose
     everything. You can resume from the last checkpoint.
  2. ROLLBACK: If the final model is worse than an earlier checkpoint
     (overfitting), you can use the earlier one instead.
  3. DEBUGGING: Compare outputs at different training stages.

WHY EVERY 100 STEPS?
  With ~500-750 total steps, saving every 100 gives you 5-7 checkpoints.
  More frequent (10): wastes disk space and time writing files.
  Less frequent (500): only 1 checkpoint — defeats the purpose.

  Each checkpoint is ~20-50 MB (just the LoRA adapter, not the full model).

ALTERNATIVES:
  save_strategy="epoch"  → save once per epoch (every ~250 steps)
  save_strategy="no"     → never save checkpoints (risky on Colab!)
```

### Evaluation Strategy (`eval_strategy="steps"`, `eval_steps=100`)

```
WHAT IT DOES:
  Every 100 steps, pause training and run the model on the 100 eval examples.
  Compute eval loss and print it alongside training loss.

  Step   Training Loss   Eval Loss
  100    1.4500          1.5100      ← eval checkpoint
  200    1.1200          1.2300      ← eval checkpoint
  300    0.9200          0.9800      ← eval checkpoint

WHY EVALUATE DURING TRAINING?
  Training loss tells you how well the model fits the TRAINING data.
  Eval loss tells you how well it generalizes to NEW data.

  If training loss keeps dropping but eval loss starts RISING,
  that's overfitting — the model is memorizing, not learning.

  ┌──────────────────────────────────────────┐
  │ Training: 1.45 → 1.12 → 0.92 → 0.85    │  ✅ Decreasing
  │ Eval:     1.51 → 1.23 → 0.98 → 0.95    │  ✅ Also decreasing (good!)
  │                                          │
  │ Training: 1.45 → 0.80 → 0.30 → 0.10    │  ⚠️ Dropping too fast
  │ Eval:     1.51 → 1.20 → 1.50 → 2.10    │  ❌ Rising = OVERFITTING
  └──────────────────────────────────────────┘

WHY EVERY 100 STEPS?
  Matches save_steps — so you have an eval score for each saved checkpoint.
  This lets you pick the checkpoint with the LOWEST eval loss, not just the last one.
```

### Random Seed (`seed=42`)

```
WHAT IT DOES:
  Makes training reproducible. With the same seed, you get the same:
  - Dataset shuffling order
  - Weight initialization for LoRA adapters
  - Dropout mask patterns
  - Data loader batching order

WHY 42?
  It's arbitrary — any number works. 42 is the conventional default
  in machine learning (a nod to "The Hitchhiker's Guide to the Galaxy").

WHY USE A SEED AT ALL?
  Without a seed, training is slightly different every time you run it.
  This makes debugging impossible — was the bad result from your config
  change, or just random variation?

  With seed=42, if you run the same config twice, you get identical results.
  Change one hyperparameter → any difference in results is from that change.
```

### Report To (`report_to="none"`)

```
WHAT IT DOES:
  Disables automatic logging to external services (Weights & Biases,
  TensorBoard, MLflow, etc.)

  Without this setting, Hugging Face Trainer auto-detects installed
  loggers and may try to log to them — causing warnings or errors
  if they're not configured.

WHY "none"?
  We don't need external logging for a workshop.
  Our logging_steps=10 prints loss to the console — that's enough.
  Setting report_to="none" avoids unexpected W&B popups or errors.

  For production training, you'd set this to "wandb" or "tensorboard"
  to get interactive loss curves, GPU utilization graphs, etc.
```

### v1 vs v2: Why Different Hyperparameters?

```
Both notebooks use the same Qwen2.5-1.5B-Instruct base model.
The only differences are the DATASET and HYPERPARAMETERS:

PARAMETER           v1 (ChatDoctor)      v2 (Reformatted WikiDoc)
────────────────────────────────────────────────────────────────────
learning_rate       2e-4                 5e-5 (4× lower)
num_train_epochs    3                    2
warmup_steps        75                   50
dataset             ChatDoctor           WikiDoc (reformatted)
total_steps         ~750                 ~500

WHY IS v2 GENTLER?

  The v1 experiment taught us: training too aggressively on bad data
  overwrites the model's existing capabilities. But even with GOOD data,
  aggressive training is risky — the model can still forget what it knows.

  v2 uses gentler hyperparameters because:

  1. LOWER LEARNING RATE (5e-5 vs 2e-4):
     The 1.5B model already has strong medical knowledge.
     A high learning rate makes BIG weight changes each step →
     risks overwriting existing knowledge ("catastrophic forgetting").
     A lower rate makes SMALL, surgical changes → adds WikiDoc knowledge
     while preserving the model's existing conversational ability.

     Analogy:
       2e-4 = repainting a wall with a roller (covers everything, including
               the parts that were already fine)
       5e-5 = touching up a wall with a small brush (fixes only what needs fixing)

  2. FEWER EPOCHS (2 vs 3):
     Less exposure to training data = less chance of memorizing it.
     2 epochs × 2,000 examples = 4,000 total views.
     That's enough for the model to absorb the WikiDoc knowledge style
     without overfitting.

  3. FEWER WARMUP STEPS (50 vs 75):
     With ~500 total steps (vs ~750), 50 warmup steps is still ~10%.
     Proportionally the same warmup period as v1.

SUMMARY:
  v1: Aggressive training on noisy data → shows what NOT to do
  v2: Gentle training on clean data → shows the right approach
  The combination teaches you that BOTH data quality AND hyperparameters matter.
```

---

## 9. Monitoring Training (Loss, Overfitting, Evaluation)

### What the Training Loss Tells You

```
TRAINING OUTPUT (what you see during trainer.train()):

Step   Training Loss   Eval Loss
25     2.1400          -
50     1.8900          -
75     1.6200          -
100    1.4500          1.5100      ← eval checkpoint
125    1.3800          -
...
300    0.9200          0.9800      ← eval checkpoint
375    0.8500          0.9500      ← final

HEALTHY TRAINING:
  Training loss:  Decreases steadily (2.14 → 0.85)
  Eval loss:      Decreases but slightly higher than training loss
  Gap:            Small and stable (0.85 vs 0.95)
```

### Diagnosing Problems from Loss

```
UNDERFITTING:
  Training loss: 2.14 → 1.95 → 1.90 (barely moves)
  ┌──────────────────────────────────────────────────────┐
  │ Loss ─────────────────────────                       │
  │                                                      │
  │ FIX: Increase learning rate. Increase epochs.        │
  │      Check that data is correctly formatted.         │
  │      Verify LoRA adapters are actually training      │
  │      (print_trainable_parameters should show > 0%).  │
  └──────────────────────────────────────────────────────┘

OVERFITTING:
  Training loss: 2.14 → 0.30 (very low)
  Eval loss:     1.51 → 2.10 (increasing!)
  ┌──────────────────────────────────────────────────────┐
  │ Training: ╲_________                                 │
  │ Eval:     ╲___╱─────── (diverging!)                  │
  │                                                      │
  │ FIX: Reduce epochs. Increase dropout. Use more data. │
  │      Early stopping (stop when eval loss increases). │
  └──────────────────────────────────────────────────────┘

GOOD FIT:
  Training loss: 2.14 → 0.85
  Eval loss:     1.51 → 0.95
  ┌──────────────────────────────────────────────────────┐
  │ Training: ╲________                                  │
  │ Eval:     ╲________ (parallel, small gap)            │
  │                                                      │
  │ This is what you want. Small gap, both decreasing.   │
  └──────────────────────────────────────────────────────┘
```

### trainer.train() Output

```python
result = trainer.train()

print(f"Training loss:   {result.training_loss:.4f}")
print(f"Runtime:         {result.metrics['train_runtime']:.0f}s")
print(f"Samples/second:  {result.metrics['train_samples_per_second']:.1f}")
print(f"Steps/second:    {result.metrics['train_steps_per_second']:.2f}")
```

Expected on Colab T4:
```
Training loss:   0.85-1.20 (varies)
Runtime:         900-1800 seconds (15-30 minutes)
Samples/second:  1.5-3.0
Steps/second:    0.20-0.40
```

---

## 10. Before vs After Benchmarking

### The Inference Function

```python
def generate_response(model, tokenizer, prompt, system_prompt=SYSTEM_PROMPT):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    # Decode only the NEW tokens (not the prompt)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()
```

### Why `torch.no_grad()`?

```
DURING TRAINING:
  PyTorch builds a computation graph for backpropagation.
  This graph uses VRAM — often more than the model itself.

DURING INFERENCE:
  We don't need gradients. We're not updating weights.
  torch.no_grad() tells PyTorch: don't build the graph.
  Saves ~2-4 GB VRAM. Runs faster.

  ALWAYS use torch.no_grad() (or torch.inference_mode()) during benchmarking.
```

### Generation Parameters

```
max_new_tokens=256
  Maximum length of generated response.
  256 tokens ≈ 190 words — enough for a medical explanation.
  Longer = slower + higher chance of rambling.

temperature=0.7
  Controls randomness.
  0.0 = deterministic (always picks most likely token)
  0.7 = moderate randomness (good for natural responses)
  1.0 = high randomness (creative but may hallucinate)
  
  For medical content: 0.3-0.7 is appropriate.
  Lower = more factual. Higher = more fluent but risky.

top_p=0.9
  Nucleus sampling. Only considers tokens in the top 90% probability mass.
  Filters out very unlikely tokens while maintaining diversity.
  0.9 is the standard default for most applications.

do_sample=True
  Required for temperature and top_p to work.
  If False: greedy decoding (temperature/top_p ignored).
```

### The Clean Comparison Approach

```
CRITICAL: Load a FRESH base model for "AFTER" benchmarking.

WHY?
  After training, the model in memory has LoRA adapters merged.
  The training process modifies the model object in-place.
  If you benchmark on the same object, you might get corrupted results.

THE CORRECT WAY:
  1. Delete the training model from memory
  2. Clear GPU cache
  3. Load a FRESH base model from HF
  4. Load the SAVED adapter weights on top
  5. Run benchmarks on this clean model

  del model, trainer
  torch.cuda.empty_cache()
  gc.collect()

  base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, ...)
  ft_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
```

### The Comparison Table

```python
import pandas as pd

comparison = pd.DataFrame({
    "Prompt": BENCHMARK_PROMPTS,
    "Base Model": [out[:200] + "..." for out in base_outputs],
    "Fine-Tuned":  [out[:200] + "..." for out in finetuned_outputs],
})
display(comparison)
```

```
┌────┬──────────────────────────┬──────────────────────────┬──────────────────────────┐
│    │ Prompt                   │ Base Model               │ Fine-Tuned               │
├────┼──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 0  │ What are the common      │ Type 2 diabetes is a     │ The common symptoms of   │
│    │ symptoms of Type 2       │ condition where the body │ Type 2 diabetes include: │
│    │ diabetes?                │ doesn't use insulin...   │ 1. Polyuria (frequent... │
├────┼──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 1  │ How does hypertension    │ High blood pressure can  │ Hypertension affects the │
│    │ affect the heart over    │ cause problems with your │ heart through several    │
│    │ time?                    │ heart and blood vessels...│ mechanisms: Left ventri..│
└────┴──────────────────────────┴──────────────────────────┴──────────────────────────┘

WHAT TO LOOK FOR:
  ✅ Fine-tuned uses medical terminology (polyuria, polydipsia)
  ✅ Fine-tuned gives structured responses (numbered lists)
  ✅ Fine-tuned is more specific and clinically accurate
  ❌ If fine-tuned is WORSE or gibberish → training failed
```

---

## 11. Saving and Pushing the Adapter

### Saving Locally

```python
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

### What Gets Saved

```
healthcare-assistant-lora/
├── adapter_config.json         ← LoRA hyperparameters (r, alpha, target_modules)
├── adapter_model.safetensors   ← The actual trained LoRA weights (~10-50 MB)
├── tokenizer.json              ← Tokenizer vocabulary
├── tokenizer_config.json       ← Tokenizer settings
├── special_tokens_map.json     ← Special token mappings
└── ...

TOTAL SIZE: ~20-50 MB

Compare to the full model: ~1 GB
The adapter is 60-150× SMALLER than the full model.
```

### What Does NOT Get Saved

```
The base model weights are NOT saved.
The adapter_model.safetensors contains ONLY the A and B matrices.

To use the adapter, you need:
  1. The base model (downloaded from HF, e.g. "Qwen/Qwen2.5-1.5B-Instruct")
  2. The adapter (your trained weights)

  The adapter is useless without the base model.
  But the base model is publicly available — no need to redistribute it.
```

### Pushing to Hugging Face Hub

```python
from huggingface_hub import notebook_login
notebook_login()  # Paste your HF token (hf_...)

model.push_to_hub(HF_REPO_ID, private=True)
tokenizer.push_to_hub(HF_REPO_ID, private=True)
```

```
WHAT HAPPENS:
  1. Creates a new repo on HF Hub (if it doesn't exist)
  2. Uploads adapter_config.json + adapter_model.safetensors + tokenizer files
  3. Sets repo as private (only you can access)

  The repo URL: https://huggingface.co/jeev1992/healthcare-assistant-lora

  Anyone with access can now:
    model = PeftModel.from_pretrained(base_model, "jeev1992/healthcare-assistant-lora")
```

### Why `private=True`?

```
PRIVATE:
  Only you (and people you grant access) can see/download the adapter.
  Good for: workshop exercises, proprietary models, testing.

PUBLIC:
  Anyone can download and use your adapter.
  Good for: open-source contributions, sharing research.

  For this workshop: private. You push your own private adapter.
```

---

## 12. VRAM Budget and Memory Management

### VRAM Breakdown on Colab T4

```
TOTAL AVAILABLE: ~15.8 GB

LOADING PHASE:
  Model weights (4-bit):           ~1.0 GB
  Tokenizer + overhead:            ~0.5 GB
  ─────────────────────────────────────────
  Total after loading:             ~1.5 GB
  Remaining:                       ~14.3 GB

BEFORE BENCHMARK (inference):
  + KV cache during generation:    ~0.5 GB (temporary)
  ─────────────────────────────────────────
  Peak during inference:           ~2.0 GB
  Remaining:                       ~13.8 GB

TRAINING PHASE:
  + LoRA adapter weights:          ~0.1 GB
  + Optimizer states (AdamW):      ~0.3 GB
  + Gradients:                     ~0.1 GB
  + Activations (batch_size=2):    ~2.0-4.0 GB
  + Gradient checkpointing buffer: ~1.0 GB
  ─────────────────────────────────────────
  Peak during training:            ~5.0-7.0 GB
  Remaining:                       ~8.8-10.8 GB

  We have plenty of headroom on T4.
```

### If You Run Out of VRAM

```
SYMPTOM: "CUDA out of memory" error during training

FIX 1: Reduce batch size
  per_device_train_batch_size=1  (was 2)
  gradient_accumulation_steps=8  (was 4, to maintain effective batch=8)

FIX 2: Reduce max_seq_length
  max_seq_length=256  (was 512)
  Fewer tokens per example = less activation memory

FIX 3: Use gradient checkpointing
  Already enabled by prepare_model_for_kbit_training()
  If not: training_args.gradient_checkpointing=True

FIX 4: Clear memory before training
  Del any variables you don't need.
  torch.cuda.empty_cache()
  gc.collect()

FIX 5: Restart runtime and try again
  Runtime → Restart runtime → Re-run cells
  Sometimes VRAM fragmentation causes OOM even with headroom.
```

### The Memory Cleanup Pattern

```python
import gc
import torch

# After BEFORE benchmarking, before training:
del base_outputs  # We already saved these
torch.cuda.empty_cache()
gc.collect()

# After training, before AFTER benchmarking:
del model, trainer
torch.cuda.empty_cache()
gc.collect()

# Then load fresh model for clean benchmarking
```

---

## 13. Common Misconceptions

### ❌ Misconception 1: "4-bit quantization destroys model quality"

**Reality:** NF4 quantization preserves 95%+ of the original model's quality. The quantization is specifically designed for neural network weight distributions. You lose some precision, but for fine-tuning purposes, the LoRA adapters compensate — they're trained in full bf16 precision. The final output quality is comparable to fp16 fine-tuning.

### ❌ Misconception 2: "Training loss should reach near zero"

**Reality:** A training loss near zero means the model has **memorized** your training data — it can recite examples but can't generalize to new questions. For QLoRA fine-tuning, a final training loss of 0.7–1.2 is typical and healthy. If loss drops below 0.3, you're almost certainly overfitting. Reduce epochs or increase dropout.

### ❌ Misconception 3: "More epochs always means better quality"

**Reality:** With LoRA on a small dataset, 2-4 epochs is usually optimal. Beyond that, the model starts memorizing training examples instead of learning general patterns. The eval loss will start *increasing* even as training loss continues *decreasing*. This divergence is the classic overfitting signal.

### ❌ Misconception 4: "You need to merge the adapter into the base model"

**Reality:** There's no need to merge for inference. `PeftModel.from_pretrained()` loads the base + adapter and handles the math at runtime. Merging (via `model.merge_and_unload()`) creates a single model but loses the ability to toggle adapters on/off. Keep them separate — it's more flexible and the inference speed difference is negligible.

### ❌ Misconception 5: "The saved adapter contains the full model"

**Reality:** The adapter is only ~10-50 MB. It contains just the LoRA A and B matrices (~9M parameters). To use it, you need the original base model (e.g. Qwen2.5-1.5B-Instruct) which is downloaded from HF separately. The adapter is useless on its own, and definitely cannot be loaded as a standalone model.

### ❌ Misconception 6: "I should fine-tune on my entire dataset at once"

**Reality:** Start small. Fine-tune on 500-1,000 examples first. Evaluate. If quality is good, ship it. Only increase data if specific metrics are underperforming. More data means longer training, higher risk of catastrophic forgetting, and diminishing returns. It's much easier to debug a 1,000-example training run than a 50,000-example one.

### ❌ Misconception 7: "`target_modules='all-linear'` is always best"

**Reality:** `all-linear` trains the most adapter parameters (~1% of total), giving maximum capacity. But for many tasks, targeting just `q_proj` and `v_proj` (attention only) is sufficient and trains 3× faster. Start with `all-linear` for best quality, then experiment with fewer modules if training time is a bottleneck.

### ❌ Misconception 8: "Training on Colab free tier is unreliable"

**Reality:** Colab free tier provides a T4 GPU for 12 hours max per session. LoRA on these small models with 1,000-2,000 examples takes 15-30 minutes. Well within limits. The main risk is getting disconnected during training — always save checkpoints (`save_steps=100`). If disconnected, reconnect and resume from the last checkpoint. Colab Pro ($10/month) gives priority GPU access and longer sessions.

---

## 14. Quick Reference: What You Used and Why

```
COMPONENT                 OUR IMPLEMENTATION                    WHY
─────────────────────────────────────────────────────────────────────────────────
GPU environment           Google Colab T4 (free)                15 GB VRAM, no setup,
                                                                accessible to all

Quantization              NF4 4-bit + double quant              1 GB model footprint,
                          via BitsAndBytesConfig                leaves room for training

Base model (v1)           Qwen2.5-1.5B-Instruct                Fits T4, good base quality,
                                                                already instruction-tuned
Base model (v2)           Qwen2.5-1.5B-Instruct                Same model as v1, isolating the
                                                                effect of data quality alone

Tokenizer config          pad_token=eos, padding=right          Required for batched training
                                                                with causal LM

Dataset (v1)              ChatDoctor-HealthCareMagic-100k       Noisy chat logs — shows what
  → 1.5B model           Base already good → fine-tuning hurts  happens when you fine-tune
                          Persona artifacts leak into outputs    a capable model on bad data

Dataset (v2)              jeev1992/wikidoc-healthassist          WikiDoc reformatted via GPT-4o-mini
  → 1.5B model           Clean data → fine-tuning preserves     into conversational style with
                          model strengths + adds safety          safety disclaimers

System prompt             Both use same prompt:                 Encourages detailed, structured
                          "knowledgeable and thorough..."       medical explanations

Training data             v1: 2,000 samples (1.5B model)       2,000 examples for both,
                          v2: 2,000 samples (1.5B model)        enough for domain adaptation

LoRA config               r=16, alpha=32, all-linear            ~9M trainable params,
                                                                good capacity for domain

Training (TRL 0.29+)      SFTTrainer with peft_config,         15-30 min on T4,
                          processing_class=tokenizer,           standard LoRA defaults
                          tokenizer.model_max_length=512

Monitoring                Training loss + eval loss per 100     Detect overfitting by
                          steps                                 comparing the two

Benchmarking              10 fixed prompts, before + after      Reproducible comparison,
                          fresh model load for after            clean measurement

Output                    ~20-50 MB adapter files               100× smaller than
                          pushed to HF Hub (private)            full model, shareable
```

---

## 15. Key Lessons — What You Learned

### Lesson 1: Don't Fine-Tune a Model That Already Knows Your Domain (v1 Experiment)

```
THE v1 RESULT (1.5B model — already good at medical Q&A):

BEFORE fine-tuning (base 1.5B model):

  Metformin:
    BASE: "Metformin helps cells respond more effectively to insulin...
           reduces liver's ability to produce glucose when not needed.
           Enhancing insulin sensitivity, reducing blood sugar levels."
    → Correct mechanism, structured, clear.

  Asthma:
    BASE: "Inhaled corticosteroids (fluticasone, budesonide)...
           leukotriene modifiers... beta-agonists..."
    → Lists correct first-line treatments with drug name examples.

  Stroke:
    BASE: "Sudden numbness or weakness in the face, arm, or leg...
           confusion, trouble speaking... severe headache..."
    → Clinically accurate, well-structured with markdown headers.

  Diabetes (Type 1 vs 2):
    BASE: "Type 1: autoimmune disorder, beta cells destroyed...
           Type 2: insulin resistance, lifestyle factors..."
    → Correct pathophysiology, good detail.

AFTER fine-tuning on ChatDoctor data:

  Metformin:
    FINETUNED: "Hello, Metformin is an oral medication that helps...
                reducing the amount of glucose produced by liver...
                In case if you are having any other co-morbidities
                like high blood pressure or kidney disease then it
                should be avoided as these can worsen the side effects
                of this Chat Doctor."
    → "Chat Doctor" appears mid-sentence. Shorter. Persona baked in.

  Asthma:
    FINETUNED: "Hello, I have studied your case. Mild persistent asthma
                requires long term controller medication like formoterol
                or albuterol combination... Hope this answers your query.
                Wishing you good health."
    → Wrong first-line (albuterol is rescue, not controller). Added
      greeting, sign-off. Less accurate than the base model.

  Pneumonia:
    FINETUNED: "Hello dear, I think you have been prescribed the right
                antibiotic course but if there is no improvement in one
                week then you need to visit doctor again or go for chest
                x-ray... Hope this information helps. Thank you!"
    → Answered a different question (patient has existing prescription).
      Ignored the actual question entirely. "Chat Doctor" in the answer.

  Cholesterol:
    FINETUNED: "HiT hanks for posting the query to ChatDoctorI have gone
                through your query in detail... Lifestyle modifications...
                Take low fat diet containing fruits vegetables pulses."
    → "HiT hanks for posting the query to ChatDoctorI" — raw ChatDoctor
      boilerplate with a typo, mid-word persona injection.

OVERALL v1 SCORECARD:
  Medical accuracy:  roughly the same (base was already correct)
  Presentation:      ❌ worse — lost markdown, gained greetings/sign-offs
  Persona leakage:   ❌ every single answer — "Hello", "Chat Doctor",
                       "Wishing you good health", "Hope this helps"
  Format degradation:❌ shorter, less structured, lost bullet points

WHY:
  The 1.5B model was pre-trained on TRILLIONS of tokens — medical textbooks,
  clinical guidelines, Wikipedia, research papers. It already "knew" medicine.

  Fine-tuning on 2,000 ChatDoctor examples didn't teach new knowledge.
  Instead, it taught the model to MIMIC ChatDoctor's casual writing style:
    greetings, sign-offs, persona text.

  Analogy: Sending a board-certified doctor to intern at a clinic
  with sloppy notes. They don't learn new medicine — they just
  pick up the bad note-taking habits.
```

### Lesson 2: Data Quality Is Everything (v2 Experiment)

```
THE v2 RESULT (1.5B model + reformatted WikiDoc):

v2 uses the same Qwen2.5-1.5B-Instruct model as v1, but trains on WikiDoc data
that was reformatted via GPT-4o-mini (data_prep_v2.py) into conversational
healthcare-assistant style with safety disclaimers.

KEY RESULTS:
  ✅ Safety improved: 0.76 → 0.86 (+0.10) — reformatted data baked in
     "consult a healthcare professional" disclaimers into every response
  ✅ Accuracy improved: 0.66 → 0.72 (+0.06) — WikiDoc knowledge absorbed
  ✅ Zero persona leakage — no ChatDoctor artifacts, no fake credentials
  ❌ Helpfulness regressed: 0.72 → 0.56 (-0.16) — answers became too concise
     (1-3 bullet points where the base model gave thorough explanations)

WHY REFORMATTED WIKIDOC:
  Raw WikiDoc is terse encyclopedic text — fine-tuning on it would teach
  the model to give short, clinical answers instead of helpful explanations.
  GPT-4o-mini reformatting converts entries into conversational Q&A style
  with 200-400 word answers, structured explanations, and safety disclaimers.

WHY THIS WORKS:
  Same model isolates the dataset variable. v1 shows noisy ChatDoctor data
  degrades a capable model. v2 shows clean, well-formatted data can
  successfully add safety behaviors while preserving existing strengths.

  The measured dataset metrics explain everything:
    ChatDoctor: 63.1% persona contamination → model learns persona artifacts
    ChatDoctor: only 3.2% safety disclaimers → model doesn't learn safety
    WikiDoc:    0% persona contamination → clean outputs, no artifacts
    WikiDoc:    99.4% safety disclaimers → model consistently adds disclaimers
```

### Lesson 3: Know When NOT to Fine-Tune

```
THE KEY INSIGHT:

For general medical Q&A, a capable base model with good prompting may already
be sufficient. Fine-tuning shines when you need to:
  • Bake in proprietary protocols or organizational guidelines
  • Enforce specific response formats consistently
  • Add domain knowledge the model genuinely lacks
  • Embed safety disclaimers into every response

PRACTICAL RULE:
  For domain adaptation with LoRA on 2,000 examples:
    Noisy data (ChatDoctor)         → persona artifacts bake into every output
    Clean reformatted data (WikiDoc) → preserves model strengths, adds safety
    Prompting alone                  → may achieve same results for general Q&A

  The workshop teaches the mechanics so you can make that judgment call yourself.
```

**The honest truth about this workshop:**

The Qwen 1.5B model already knows medicine — it was pre-trained on trillions of
tokens including medical textbooks, clinical guidelines, and research papers. For
general medical Q&A like our 10 benchmark questions, a well-crafted system prompt
could achieve similar results without any fine-tuning at all. Our v2 improvement
was marginal — safety went up (disclaimers baked into data), but accuracy and
helpfulness barely moved. Just 2,000 clean rows can only do so much when the
model already knows the domain.

**So why fine-tune at all?** To teach you the mechanics. The real power of
fine-tuning is **domain adaptation on domains the model has never seen**:

```
WHERE FINE-TUNING IS ESSENTIAL (prompting CAN'T do this):

  • Proprietary clinical protocols
      Your hospital's specific triage rules, discharge criteria, or
      escalation procedures — not in any public dataset.

  • Internal drug formulary
      Which medications YOUR organization approves for which conditions.
      No public model knows this.

  • Rare disease data
      Conditions with so few published cases that the base model gives
      generic or wrong answers. Fine-tuning on even 500 expert-curated
      examples can dramatically improve accuracy.

  • Regulatory compliance language
      Exact phrasing required by FDA/EMA/HIPAA for patient-facing content.
      Prompting can approximate this, but fine-tuning makes it consistent.

  • Company-specific terminology
      Internal abbreviations, product names, workflow-specific jargon
      that the model has literally never encountered.

WHERE PROMPTING IS SUFFICIENT (this workshop's scenario):

  • General medical Q&A (model already knows medicine)
  • Adding safety disclaimers (system prompt: "always recommend consulting
    a healthcare professional")
  • Formatting preferences (bullet points, headers, concise answers)
  • Tone adjustments (more empathetic, more clinical, etc.)
```

The 2,000-row WikiDoc experiment is your training ground — it teaches you the
complete QLoRA pipeline (data prep, training, evaluation) so that when you
encounter a domain the model genuinely doesn't know, you have the skills to
do it right.

### Lesson 4: Choose Your Dataset — Don't Just Clean It

```
THE CHATDOCTOR PROBLEM (v1 dataset):

  Every single answer in ChatDoctor starts with a greeting:
    "Hello and welcome to Chat Doctor"
    "HiT hanks for asking to ChatDoctor"
    "Hi, I understand your concern."

  You can try to clean these with regex — but:
    • 100% of answers start with greetings → cleaning leaves you with
      responses that START with medical content but still have the
      casual ChatDoctor STYLE throughout
    • Regex can't remove style, only specific strings
    • Some boilerplate always survives

  The v1 lesson: even after cleaning, the fine-tuned model mimics
  ChatDoctor's casual tone — because that tone is baked into the
  structure of the answers, not just the opening lines.

THE WIKIDOC SOLUTION (v2 dataset):

  WikiDoc answers read like encyclopedic medical content:
    "Squamous cell carcinoma of the lung may be classified according
     to the WHO histological classification system into 4 main types..."

  No persona. No greetings. No sign-offs. Just medicine.
  The fine-tuned model outputs the same clean style.

DATASET QUALITY TIERS:

Level 1: Raw chat logs (ChatDoctor)
  • Real conversations but noisy
  • Persona artifacts throughout
  • Good for showing what NOT to do

Level 2: Curated encyclopedic content (WikiDoc) ← what v2 uses
  • Clean, factual, structured
  • No persona artifacts
  • Good for domain knowledge injection

Level 3: LLM-generated clean QA pairs
  • Use GPT-4 / Claude to generate from medical guidelines
  • Full control over format, style, accuracy
  • More expensive but highest quality for production

Level 4: Expert-curated data
  • Medical professionals write or review every example
  • Required for clinical/regulated applications
```

### Lesson 5: RAG as an Alternative (or Complement) to Fine-Tuning

```
RAG = RETRIEVAL-AUGMENTED GENERATION

Instead of baking knowledge into model weights (fine-tuning),
RAG retrieves relevant documents at query time and injects them
into the model's context window.

HOW RAG WORKS:
  1. User asks: "What are symptoms of diabetes?"
  2. Retriever searches a knowledge base (vector database)
  3. Finds relevant medical guidelines/articles
  4. Injects retrieved text into the model's context
  5. Model generates answer grounded in retrieved documents

WHEN TO USE EACH:

Use RAG when:
  ✅ You need up-to-date information (guidelines change annually)
  ✅ You need to cite sources ("According to CDC guidelines...")
  ✅ Base model already writes well but lacks specific knowledge
  ✅ You want to avoid risking model degradation from fine-tuning
  ✅ Quick to deploy — no GPU or training needed

Use Fine-Tuning when:
  ✅ Model genuinely lacks domain knowledge (rare diseases, proprietary protocols)
  ✅ You need consistent terminology or style changes
  ✅ Latency matters (RAG adds retrieval time)
  ✅ You can't afford a vector database infrastructure
  ✅ You have high-quality, clean training data

Use BOTH when:
  ✅ You need domain style + up-to-date knowledge
  ✅ Fine-tune for terminology/style, RAG for facts
  ✅ Most production systems use this combination

DECISION FLOWCHART:

  "Does the base model handle my task well?"
       │
       ├── YES → Try prompt engineering first
       │         Still not good enough? → RAG
       │         Need style changes too? → RAG + Light fine-tuning
       │
       └── NO → Is my training data clean?
                 │
                 ├── YES → Fine-tune (may add RAG later)
                 │
                 └── NO → Clean data first, OR use RAG instead
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| **Pipeline** | Check GPU → Install deps → Load 4-bit → Benchmark → Train → Benchmark → Push |
| **BitsAndBytes** | NF4 quantization reduces 1.5B from 3 GB to ~0.75 GB VRAM (v1 only) |
| **Tokenizer** | Set pad_token=eos_token and padding_side="right" for causal LMs |
| **Dataset** | Chat messages format with system/user/assistant roles |
| **LoRA prep** | `prepare_model_for_kbit_training()` enables gradients on quantized models |
| **SFTTrainer** | Handles chat template, tokenization, padding, and loss masking automatically |
| **Batch size** | Use gradient_accumulation to simulate larger batches without OOM |
| **Learning rate** | 2e-4 is the standard for LoRA. Cosine scheduler + 10% warmup. |
| **Loss monitoring** | Training loss should decrease. Eval loss tracks but stays slightly higher. |
| **Clean benchmark** | Delete training model, empty cache, load fresh base + adapter for after-test |
| **Adapter size** | ~20-50 MB. Does NOT contain the base model. |
| **VRAM** | T4 has 15.8 GB. QLoRA training uses ~5-7 GB. Plenty of headroom. |

---

*Previous: [Module 1 — Strategy & Data ←](../module_1_strategy_data/notes.md)*  
*Next: [Module 3 — HF Deploy & Inference →](../module_3_hf_deploy_inference/notes.md)*
