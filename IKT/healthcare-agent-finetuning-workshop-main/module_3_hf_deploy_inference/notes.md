# Hugging Face Deploy & Inference
## A Complete Guide to Deploying and Serving Fine-Tuned LoRA Adapters

---

## Table of Contents

1. [Why Deployment Matters](#1-why-deployment-matters)
2. [The Adapter Architecture: What You're Actually Deploying](#2-the-adapter-architecture-what-youre-actually-deploying)
3. [Loading an Adapter from Hugging Face Hub](#3-loading-an-adapter-from-hugging-face-hub)
4. [The Adapter Toggle Pattern](#4-the-adapter-toggle-pattern)
5. [Inference on a GPU Environment](#5-inference-on-a-gpu-environment)
6. [The HF Inference API](#6-the-hf-inference-api)
7. [Inference Endpoints (Dedicated Deployment)](#7-inference-endpoints-dedicated-deployment)
8. [Merging Adapters: When and Why](#8-merging-adapters-when-and-why)
9. [Quantized Inference Considerations](#9-quantized-inference-considerations)
10. [Benchmarking Base vs Fine-Tuned in One Load](#10-benchmarking-base-vs-fine-tuned-in-one-load)
11. [Exporting Results for Downstream Evaluation](#11-exporting-results-for-downstream-evaluation)
12. [Common Misconceptions](#12-common-misconceptions)
13. [How Our Healthcare Agent Uses This](#13-how-our-healthcare-agent-uses-this)
14. [Production Deployment: Cloud Providers & Scaling Strategies](#14-production-deployment-cloud-providers--scaling-strategies)

---

## 1. Why Deployment Matters

Training is only half the job. A fine-tuned model that sits in a Colab notebook is useless. Deployment makes it accessible — to other notebooks, to applications, to your evaluation pipeline.

```
TRAINING (Module 2)                    DEPLOYMENT (Module 3)
──────────────────                     ──────────────────

"I trained a model!"                   "Anyone can use the model."

  Lives on: one Colab session            Lives on: HF Hub (persistent)
  Accessible by: you                     Accessible by: anyone with the link
  Survives: until session ends           Survives: forever
  Usable for eval: no (Module 4          Usable for eval: yes (load from Hub,
   would need to retrain)                 run inference, send to LangSmith)
```

### The Deployment Pipeline

```
Module 2: Train → Save adapter locally → Push to HF Hub
                                              │
                                              ▼
Module 3: Load from HF Hub → Run inference → Export results
                                              │
                                              ▼
Module 4: Load results → Create LangSmith dataset → Evaluate
```

---

## 2. The Adapter Architecture: What You're Actually Deploying

### What Lives on HF Hub

```
YOUR HF REPO (e.g., jeev1992/healthcare-assistant-lora):
├── adapter_config.json           ← LoRA configuration (r, alpha, targets)
├── adapter_model.safetensors     ← Trained LoRA weights (~10-50 MB)
├── tokenizer.json                ← Tokenizer vocabulary
├── tokenizer_config.json         ← Tokenizer settings
├── special_tokens_map.json       ← Special token mappings
└── README.md                     ← Auto-generated model card

TOTAL SIZE: ~20-50 MB
```

### What Does NOT Live on HF Hub

```
The base model (Qwen/Qwen2.5-1.5B-Instruct) is NOT in your repo.
It's a PUBLIC model hosted at:
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

Your adapter is a PATCH on top of this base model.
Like a diff in Git — it describes changes, not the full file.

To use your adapter:
  1. Download the base model from HuggingFace's repo     (~1 GB in 4-bit)
  2. Download your adapter from your repo                 (~20-50 MB)
  3. Apply adapter on top of base model at runtime        (< 1 second)
```

### The adapter_config.json

```json
{
    "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    "task_type": "CAUSAL_LM",
    "peft_type": "LORA"
}
```

This file tells PEFT exactly how to reconstruct the adapter architecture and apply it to the base model. Without this file, the safetensors weights are meaningless blobs of numbers.

---

## 3. Loading an Adapter from Hugging Face Hub

### The Two-Step Loading Process

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# Step 1: Load the BASE model (same config as training)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# Step 2: Load the ADAPTER on top
ft_model = PeftModel.from_pretrained(
    base_model,
    "jeev1992/healthcare-assistant-lora",  # Your HF repo
)
```

### What Happens Under the Hood

```
PeftModel.from_pretrained():
  1. Downloads adapter_config.json from HF Hub
  2. Reads: r=16, alpha=32, target_modules=[q_proj, k_proj, ...]
  3. Downloads adapter_model.safetensors from HF Hub
  4. For each target module in the base model:
     a. Wraps the existing Linear layer with a LoRALinear layer
     b. Creates A matrix (d × r) and B matrix (r × d)
     c. Loads saved weights into A and B
  5. Sets all base model weights as frozen
  6. Returns the PeftModel (base + adapter, ready for inference)

  Time: ~5-10 seconds (downloading + wrapping)
  Additional VRAM: ~50-100 MB (the adapter weights)
```

### Why Use the SAME Quantization Config?

```
CRITICAL: The quantization config at inference MUST match training.

TRAINING:   load_in_4bit=True, nf4, double_quant=True
INFERENCE:  load_in_4bit=True, nf4, double_quant=True   ← MUST MATCH

WHY?
  LoRA adapters were trained on a 4-bit quantized base model.
  The adapter learned to compensate for quantization artifacts.

  If you load the base in fp16 at inference:
    Base model weights are slightly different (not quantized).
    Adapter was tuned for quantized weights.
    Result: Adapter corrections are wrong → quality degrades.

  If you load in 4-bit with DIFFERENT quant type:
    Different quantization → different weight values.
    Same problem: adapter outputs are calibrated for wrong base.

  ALWAYS use identical quantization config for inference.
```

---

## 4. The Adapter Toggle Pattern

### Enable/Disable Without Reloading

```python
# Fine-tuned behavior (adapter ON):
ft_model.enable_adapter_layers()
finetuned_response = generate_response(ft_model, tokenizer, prompt)

# Base model behavior (adapter OFF):
ft_model.disable_adapter_layers()
base_response = generate_response(ft_model, tokenizer, prompt)

# Turn adapter back on:
ft_model.enable_adapter_layers()
```

### How This Works

```
WITH ADAPTER ENABLED:
  output = base_layer(x) + (lora_B(lora_A(x))) × (alpha/r)
                             ↑ adapter contribution active

WITH ADAPTER DISABLED:
  output = base_layer(x)    ← pure base model output
                             ↑ adapter contribution = 0

  Same model object in memory.
  Same VRAM usage.
  Just skips the adapter computation.
```

### Why This Is Better Than Loading Two Models

```
OPTION A: Load two separate models (wasteful)
  base_model  = load("Qwen2.5-1.5B-Instruct")       → ~1 GB VRAM
  ft_model    = load("Qwen2.5-1.5B-Instruct")        → ~1 GB VRAM
  ft_model    = PeftModel.from_pretrained(ft_model)  → ~0.1 GB VRAM
  TOTAL: ~2.1 GB VRAM

OPTION B: Toggle adapter (efficient)
  ft_model    = load("Qwen2.5-1.5B-Instruct")        → ~1 GB VRAM
  ft_model    = PeftModel.from_pretrained(ft_model)  → ~0.1 GB VRAM
  ft_model.disable_adapter_layers()  → base behavior (0 extra VRAM)
  ft_model.enable_adapter_layers()   → fine-tuned behavior
  TOTAL: ~1.1 GB VRAM

  SAVINGS: ~1.0 GB VRAM + ~10 seconds loading time.
  Option B is strictly better for benchmarking scenarios.
```

### Running a Full Benchmark with Toggle

```python
base_outputs = []
ft_outputs = []

for prompt in BENCHMARK_PROMPTS:
    # Base model (adapter off)
    ft_model.disable_adapter_layers()
    base_outputs.append(generate_response(ft_model, tokenizer, prompt))

    # Fine-tuned model (adapter on)
    ft_model.enable_adapter_layers()
    ft_outputs.append(generate_response(ft_model, tokenizer, prompt))

# Now you have paired outputs from same prompts
# base_outputs[i] and ft_outputs[i] correspond to the same prompt
```

---

## 5. Inference on a GPU Environment

### Generation Pipeline

```
USER PROMPT: "What are the symptoms of Type 2 diabetes?"
    │
    ▼
TOKENIZER (apply_chat_template):
    <|im_start|>system
    You are a helpful healthcare assistant...<|im_end|>
    <|im_start|>user
    What are the symptoms of Type 2 diabetes?<|im_end|>
    <|im_start|>assistant
    │
    ▼
MODEL (generate):
    Token 1: "The"
    Token 2: " common"
    Token 3: " symptoms"
    ...
    Token N: "<|im_end|>"     ← stop generation
    │
    ▼
TOKENIZER (decode):
    "The common symptoms of Type 2 diabetes include polyuria..."
    │
    ▼
RESPONSE (strip special tokens + trim)
```

### Autoregressive Generation Explained

```
LLMs generate ONE TOKEN AT A TIME, left to right.

Step 1: Input = "The"         → Predict next token → " common"
Step 2: Input = "The common"  → Predict next token → " symptoms"
Step 3: Input = "The common symptoms" → " of"
...
Step N: Input = full text so far → "<|im_end|>" → STOP

This is why:
  - Longer outputs take proportionally more time
  - max_new_tokens=256 generates at most 256 iterations
  - Each step requires a full forward pass through the model
  - KV cache avoids recomputing attention for previous tokens
```

### KV Cache and Inference Speed

```
WITHOUT KV CACHE (naive):
  Step 1: Compute attention for tokens [1]                → O(1)
  Step 2: Compute attention for tokens [1, 2]             → O(2)
  Step 3: Compute attention for tokens [1, 2, 3]          → O(3)
  ...
  Step N: Compute attention for tokens [1, 2, ..., N]     → O(N)
  Total: O(N²) — gets VERY slow for long sequences

WITH KV CACHE (default in HF):
  Step 1: Compute for token [1], CACHE key/value           → O(1)
  Step 2: Compute for token [2] only, attend to cache      → O(1)
  Step 3: Compute for token [3] only, attend to cache      → O(1)
  Total: O(N) — linear, much faster

  Cost: KV cache uses additional VRAM (~0.5-2 GB depending on sequence length)
  Benefit: 10-50× faster generation for long sequences

  HF Transformers enables KV cache by default. You don't need to configure it.
```

---

## 6. The HF Inference API

### What Is It?

```
Hugging Face provides a free (rate-limited) API to run inference on hosted models.

WITHOUT Inference API:
  You need a GPU.
  You download the model.
  You load it into VRAM.
  You run generation locally.

WITH Inference API:
  No GPU needed.
  Send a prompt via HTTP.
  Get a response back.
  HF runs the model on their servers.
```

### Using InferenceClient

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    token="hf_..."  # Your HF token
)

# Format the prompt manually (API expects raw text, not messages)
prompt = """<|im_start|>system
You are a helpful healthcare assistant...<|im_end|>
<|im_start|>user
What are the symptoms of diabetes?<|im_end|>
<|im_start|>assistant
"""

response = client.text_generation(
    prompt,
    max_new_tokens=256,
    temperature=0.7,
)
print(response)
```

### Free Tier Limitations

```
FREE INFERENCE API:
  Rate limit:    ~30 requests/minute (varies by model)
  Shared GPU:    Requests may queue during peak hours
  Model support: Not all models are available
  Adapters:      Cannot apply custom LoRA adapters on free API

  ✅ Good for: Quick testing, demos, small-scale evaluation
  ❌ Not for: Production use, high-volume inference, custom adapters

INFERENCE ENDPOINTS (paid):
  Dedicated GPU: Your own instance (no queuing)
  Custom models: Deploy your adapter merged into base model
  Scaling:       Auto-scale based on traffic
  Cost:          ~$0.60/hour for T4, ~$3.00/hour for A10G

  ✅ Good for: Production deployment, custom models, high-volume
  ❌ Not for: Quick testing (overkill), tight budgets
```

### Why We Don't Use Inference API for This Workshop

```
The free Inference API CANNOT load custom LoRA adapters.
It only serves unmodified base models from HF Hub.

For our fine-tuned model, we have two options:
  A. Load locally on Colab (free, works now)     ← WHAT WE DO
  B. Deploy an Inference Endpoint (paid, $0.60+/hr)

The notebook includes the Inference API code as a COMMENTED-OUT example
for students who want to deploy to production later.
```

---

## 7. Inference Endpoints (Dedicated Deployment)

### When You Need Them

```
FOR THIS WORKSHOP: Not needed. Colab inference is fine.
FOR PRODUCTION: This is how you'd serve the model to real users.
```

### Creating an Inference Endpoint

```
1. Go to huggingface.co/inference-endpoints
2. Select your model repo (jeev1992/healthcare-assistant-lora)
3. Choose hardware (T4 for ~$0.60/hr, A10G for ~$3.00/hr)
4. Deploy

   The endpoint URL looks like:
   https://xxxx.us-east-1.aws.endpoints.huggingface.cloud
```

### The Merge-First Pattern for Endpoints

```
PROBLEM:
  Inference Endpoints loads a single model.
  Your repo contains only the adapter, not the full model.
  The endpoint doesn't know to download Qwen2.5-1.5B-Instruct separately.

SOLUTION: Merge adapter into base model, push as one repo.

  # Merge locally (needs enough RAM/VRAM):
  from peft import PeftModel
  merged = PeftModel.from_pretrained(base_model, "adapter-repo")
  merged = merged.merge_and_unload()
  merged.save_pretrained("merged-model/")
  tokenizer.save_pretrained("merged-model/")

  # Push the merged model (full model, ~3 GB):
  merged.push_to_hub("jeev1992/healthcare-assistant-merged")

  # Deploy this merged repo to Inference Endpoint
```

---

## 8. Merging Adapters: When and Why

### The Merge Decision

```
KEEP SEPARATE (adapter + base):
  ✅ Can toggle adapter on/off
  ✅ Can swap different adapters (e.g., healthcare vs legal)
  ✅ Tiny upload (~20-50 MB)
  ❌ Need to load base model first, then adapter

MERGE INTO SINGLE MODEL:
  ✅ Single file, simple loading
  ✅ Works with Inference Endpoints
  ✅ Slightly faster inference (no adapter math)
  ❌ Can't toggle adapter
  ❌ Large upload (~3 GB)
  ❌ Can't swap adapters without reloading
```

### How Merging Works

```python
# Load base + adapter
ft_model = PeftModel.from_pretrained(base_model, adapter_repo)

# Merge adapter weights into base model
merged_model = ft_model.merge_and_unload()

# merged_model is now a regular model (no PEFT wrapper)
# The LoRA A×B products have been permanently added to the base weights:
#   new_weight = old_weight + (alpha/r) × A × B
```

```
BEFORE MERGE:
  W_frozen + (α/r) × A × B computed at runtime
  ↑ separate objects in memory

AFTER MERGE:
  W_new = W_frozen + (α/r) × A × B    ← computed once, stored
  ↑ single weight tensor

  The adapter is "baked in" to the weights.
  No way to separate them again.
```

### For This Workshop: Don't Merge

```
We keep the adapter separate because:
  1. We NEED the toggle for base vs fine-tuned comparison
  2. Module 4 evaluation depends on comparing both versions
  3. Tiny adapter is faster to push/pull from Hub
  4. Students can experiment with different adapters

Merging is a PRODUCTION deployment optimization, not a workshop need.
```

---

## 9. Quantized Inference Considerations

### Quantization at Training vs Inference

```
TRAINING:
  Base model: 4-bit NF4 (BitsAndBytes)
  LoRA adapters: bf16 (full precision training)
  Compute: bf16 (dequantize 4-bit → bf16 for matrix multiply)

INFERENCE (must match):
  Base model: 4-bit NF4 (SAME BitsAndBytes config)
  LoRA adapters: loaded in original precision
  Compute: bf16

  WHY MATCH? The adapter was calibrated assuming specific base weight values.
  Different quantization = different base values = adapter corrections misaligned.
```

### What If You Don't Have a GPU at Inference?

```
OPTION 1: Colab (free GPU, what we use)
  Load model in 4-bit on Colab T4.
  Run inference in the notebook.
  Good for: workshops, experiments, small-scale eval.

OPTION 2: CPU-only inference (slow but works)
  Load model in fp32 (no quantization, needs ~6 GB RAM).
  Inference: 10-30× slower than GPU.
  
  base_model = AutoModelForCausalLM.from_pretrained(
      MODEL_ID,
      device_map="cpu",
      # NO quantization_config — BitsAndBytes requires GPU
  )
  
  WARNING: Quality may differ slightly from 4-bit training.
  The adapter was trained on 4-bit base weights.

OPTION 3: Inference Endpoint (paid, no setup)
  Deploy merged model to HF Inference Endpoints.
  Send HTTP requests. No local GPU needed.

OPTION 4: GGUF + llama.cpp (advanced)
  Convert model to GGUF format.
  Run on CPU with llama.cpp.
  Highly optimized CPU inference.
```

---

## 10. Benchmarking Base vs Fine-Tuned in One Load

### The Full Benchmark Pipeline

```python
results = {"prompts": BENCHMARK_PROMPTS, "base": [], "finetuned": []}

for i, prompt in enumerate(BENCHMARK_PROMPTS):
    print(f"\n--- Prompt {i+1}/{len(BENCHMARK_PROMPTS)} ---")

    # Base model (adapter disabled)
    ft_model.disable_adapter_layers()
    base_resp = generate_response(ft_model, tokenizer, prompt)
    results["base"].append(base_resp)

    # Fine-tuned (adapter enabled)
    ft_model.enable_adapter_layers()
    ft_resp = generate_response(ft_model, tokenizer, prompt)
    results["finetuned"].append(ft_resp)

    print(f"Prompt: {prompt[:60]}...")
    print(f"Base:   {base_resp[:100]}...")
    print(f"FT:     {ft_resp[:100]}...")
```

### Building the Comparison Table

```python
import pandas as pd

df = pd.DataFrame({
    "Prompt": BENCHMARK_PROMPTS,
    "Base Model": [r[:200] + "..." for r in results["base"]],
    "Fine-Tuned": [r[:200] + "..." for r in results["finetuned"]],
})
display(df)
```

### What the Comparison Reveals

```
TYPICAL IMPROVEMENTS AFTER QLoRA FINE-TUNING:

BEFORE (base model):
  "What are the symptoms of Type 2 diabetes?"
  → "Type 2 diabetes is a condition that affects how the body processes
     blood sugar. Some symptoms include feeling tired and thirsty..."
     ↑ Vague, conversational, no medical terms

AFTER (fine-tuned):
  "What are the symptoms of Type 2 diabetes?"
  → "The common symptoms of Type 2 diabetes include:
     1. Polyuria (increased urination)
     2. Polydipsia (increased thirst)
     3. Polyphagia (increased hunger)
     4. Unexplained weight loss
     5. Fatigue and weakness
     6. Blurred vision
     It is recommended to consult a healthcare professional..."
     ↑ Medical terminology, structured, includes disclaimer

KEY DIFFERENCES:
  ✅ Medical vocabulary (polyuria, polydipsia, polyphagia)
  ✅ Structured format (numbered lists)
  ✅ Completeness (covers more symptoms)
  ✅ Professional tone (consistent across all prompts)
  ✅ Safety disclaimer (recommends consulting professional)
```

---

## 11. Exporting Results for Downstream Evaluation

### The Export Format

```python
import json

export = {
    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "adapter_repo": HF_REPO_ID,
    "prompts": BENCHMARK_PROMPTS,
    "base_outputs": results["base"],
    "finetuned_outputs": results["finetuned"],
}

with open("inference_results.json", "w") as f:
    json.dump(export, f, indent=2)
```

### Why Export to JSON?

```
MODULE 3 PRODUCES:  inference_results.json
MODULE 4 CONSUMES:  inference_results.json (or benchmark_results.json from Module 2)

This decouples the modules:
  Module 4 doesn't need a GPU.
  Module 4 doesn't need to load the model.
  Module 4 only needs the TEXT outputs to evaluate.

  You could run Module 3 on Colab with a GPU,
  then run Module 4 on your local machine with no GPU.
  The JSON file is the bridge.
```

### JSON Structure

```json
{
  "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
  "adapter_repo": "jeev1992/healthcare-assistant-lora",
  "prompts": [
    "What are the common symptoms of Type 2 diabetes?",
    "How does hypertension affect the heart over time?",
    ...
  ],
  "base_outputs": [
    "Type 2 diabetes is a condition that affects...",
    "High blood pressure can cause problems...",
    ...
  ],
  "finetuned_outputs": [
    "The common symptoms of Type 2 diabetes include: 1. Polyuria...",
    "Hypertension affects the heart through several mechanisms...",
    ...
  ]
}
```

---

## 12. Common Misconceptions

### ❌ Misconception 1: "Pushing to HF Hub deploys the model for inference"

**Reality:** Pushing to HF Hub **stores** the adapter files. It does NOT deploy an inference endpoint. The repo is just a file storage location. To USE the adapter, someone must download it and load it locally (or deploy an Inference Endpoint, which is a separate paid step). HF Hub = GitHub for models. GitHub doesn't run your code; HF Hub doesn't run your model.

### ❌ Misconception 2: "I need to push the full model to HF Hub"

**Reality:** You only push the adapter (~20-50 MB). The base model is already on HF Hub, hosted by Qwen. When loading, `PeftModel.from_pretrained()` knows to download the base model from Qwen's repo and your adapter from your repo. There's no need to duplicate the 3 GB base model.

### ❌ Misconception 3: "Loading from Hub is slow"

**Reality:** First download is ~20-50 MB (adapter) + ~1 GB (base model in 4-bit). This takes 30-60 seconds on a typical connection. Subsequent loads use the HF cache (`~/.cache/huggingface/`) — the model is already on disk. Cached loads take 5-10 seconds. The cache persists across Colab sessions on Colab Pro.

### ❌ Misconception 4: "disable_adapter_layers() deletes the adapter"

**Reality:** `disable_adapter_layers()` **skips** the adapter computation during forward passes. The adapter weights remain in memory. Call `enable_adapter_layers()` to reactivate them instantly. Nothing is deleted or modified. It's a toggle, not a destructor.

### ❌ Misconception 5: "I can use any quantization config at inference"

**Reality:** You should use the **same** quantization config at inference that was used during training. The LoRA adapters were calibrated against specific quantized base weights. Loading the base in fp16 when the adapter was trained on 4-bit NF4 will produce subtly wrong outputs. Not catastrophically wrong — but measurably worse.

### ❌ Misconception 6: "The HF Inference API can serve my custom adapter"

**Reality:** The free HF Inference API serves unmodified base models only. To serve a custom fine-tuned model, you need either: (a) load locally with PeftModel, or (b) merge the adapter into the base model and deploy a paid Inference Endpoint. The free API is great for testing base models, not for custom adapters.

### ❌ Misconception 7: "Merging is required before deploying"

**Reality:** Merging is one deployment option, not a requirement. `PeftModel.from_pretrained()` handles inference with separate base + adapter at near-identical speed. Merge only if: (a) you need a standalone model file for a system that doesn't support PEFT, or (b) you're deploying to HF Inference Endpoints which expect a single model. For Colab/local use, keep them separate.

---

## 13. How Our Healthcare Agent Uses This

```
COMPONENT                  OUR IMPLEMENTATION                    WHY
─────────────────────────────────────────────────────────────────────────────────
Adapter hosting            HF Hub (private repo)                 Persistent, shareable,
                                                                 version-controlled

Loading method             PeftModel.from_pretrained()           Two-step: base + adapter
                           with HF repo ID                       from Hub automatically

Quantization match         Same BitsAndBytesConfig as training   Adapter calibrated for
                           (4-bit NF4, double quant, bf16)       these specific weights

Comparison method          adapter toggle (disable/enable)       One model in memory,
                           NOT two separate models               saves ~0.5 GB VRAM

Benchmark pipeline         10 prompts × 2 versions               Paired comparison:
                           (adapter off → adapter on)            same model, same session

Inference API              Commented-out example in notebook     Shows production path
                           using InferenceClient                 for paid deployment

Export format              inference_results.json                Decouples inference (GPU)
                                                                 from evaluation (no GPU)

Environment                Google Colab T4                       Free, sufficient VRAM,
                                                                 accessible to all students
```

### The Data Flow

```
HF Hub
  │
  ├── Qwen/Qwen2.5-1.5B-Instruct (base model, public, ~1 GB in 4-bit)
  │     │
  └── jeev1992/healthcare-assistant-lora (adapter, private, ~20-50 MB)
        │
        ▼
Module 3 Notebook (Colab T4)
  │
  ├── Step 1: Load base model + adapter from Hub
  │
  ├── Step 2: For each of 10 benchmark prompts:
  │     ├── Disable adapter → generate base response
  │     └── Enable adapter  → generate fine-tuned response
  │
  ├── Step 3: Display comparison table (pandas DataFrame)
  │
  └── Step 4: Export to inference_results.json
        │
        ▼
Module 4 Notebook (can run anywhere — no GPU needed)
  │
  └── Loads inference_results.json → LangSmith evaluation
```

---

## 14. Production Deployment: Cloud Providers & Scaling Strategies

The HF Hub workflow in this workshop is great for learning and prototyping. But when you
need to serve a fine-tuned model to real users — with low latency, high availability, and
cost control — you need a production deployment strategy.

### The Deployment Spectrum

```
COMPLEXITY ──────────────────────────────────────────────────────────────►

  HF Hub              Serverless          Dedicated GPU         Self-Managed
  (this workshop)     (API-based)         (managed service)     (full control)
  │                   │                   │                     │
  Free, manual        Pay-per-call        Pay-per-hour          Pay-per-instance
  No auto-scaling     Auto-scales         Manual/auto scaling   Full control
  No SLA              Provider SLA        Provider SLA          Your SLA
  │                   │                   │                     │
  HF Hub +            HF Inference        AWS SageMaker         EC2 + vLLM/TGI
  Colab               Endpoints,          Endpoints,            ECS/EKS +
                      AWS Lambda +        Azure ML Endpoints,   custom containers
                      SageMaker           GCP Vertex AI
                      Serverless
```

### Strategy 1: Serverless Inference (Pay-Per-Call)

```
WHAT:
  No GPU to manage. You send a request, the provider handles everything.
  The model loads on demand, runs your query, and shuts down.

WHEN TO USE:
  • Low traffic (< 100 requests/day)
  • Unpredictable usage patterns
  • Prototyping and testing
  • Budget-conscious — only pay when the model is actually running

AWS OPTIONS:
  ┌─────────────────────────────────────────────────────────────────┐
  │ SageMaker Serverless Inference                                  │
  │                                                                 │
  │ • Model loads from S3 when a request arrives                    │
  │ • Auto-scales to zero when idle (no cost when not in use)       │
  │ • Cold start: 30-60 seconds (model needs to load into GPU)     │
  │ • Max payload: 6 MB                                             │
  │ • Max response time: 60 seconds                                 │
  │ • Pricing: ~$0.0001-0.001 per request (depends on model size)   │
  │                                                                 │
  │ LIMITATION: GPU memory capped at 6 GB. Fine for our 1.5B       │
  │ model in 4-bit (~1 GB), but won't work for 7B+ models.        │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │ HF Inference Endpoints (Serverless)                             │
  │                                                                 │
  │ • Upload merged model to HF Hub → deploy as endpoint            │
  │ • Managed by Hugging Face on AWS infrastructure                 │
  │ • Cold start: 30-120 seconds                                    │
  │ • Supports custom models + adapters (merged only)               │
  │ • Pricing: ~$0.06/hr (CPU) to $1.30/hr (GPU T4)                │
  │                                                                 │
  │ SIMPLEST OPTION: If your model is already on HF Hub,            │
  │ this is the fastest path to a production API.                   │
  └─────────────────────────────────────────────────────────────────┘

COLD START PROBLEM:
  Serverless = model loads from scratch on each request (or after idle timeout).
  For a 1.5B model: 30-60 seconds cold start.
  For a 7B model: 60-120 seconds cold start.

  If your users can't wait 60 seconds → use a dedicated endpoint instead.
  Some providers offer "provisioned concurrency" (keep N instances warm)
  but that defeats the cost advantage of serverless.
```

### Strategy 2: Dedicated GPU Endpoints (Always-On)

```
WHAT:
  A GPU instance running 24/7 (or on a schedule) with your model pre-loaded.
  No cold start — requests are served in seconds.

WHEN TO USE:
  • Consistent traffic (> 100 requests/day)
  • Latency-sensitive (< 5 second response time required)
  • Production applications with SLA requirements
  • Multiple models served from the same instance

AWS SAGEMAKER REAL-TIME ENDPOINTS:
  ┌─────────────────────────────────────────────────────────────────┐
  │ How it works:                                                   │
  │   1. Package model as a SageMaker-compatible container          │
  │   2. Choose instance type (ml.g4dn.xlarge = T4 GPU, ~$0.74/hr) │
  │   3. Deploy → get an HTTPS endpoint URL                         │
  │   4. Send requests via SDK or HTTP POST                         │
  │                                                                 │
  │ Instance types for fine-tuned LLMs:                             │
  │   ml.g4dn.xlarge   → 1× T4 (16 GB)   ~$0.74/hr    1.5B-3B    │
  │   ml.g5.xlarge     → 1× A10G (24 GB)  ~$1.41/hr    3B-7B     │
  │   ml.g5.2xlarge    → 1× A10G (24 GB)  ~$1.52/hr    7B        │
  │   ml.p4d.24xlarge  → 8× A100 (320 GB) ~$37/hr      70B+      │
  │                                                                 │
  │ For our 1.5B model: ml.g4dn.xlarge is sufficient (~$0.74/hr)   │
  └─────────────────────────────────────────────────────────────────┘

  Auto-scaling:
    SageMaker can auto-scale based on:
    • InvocationsPerInstance (requests/minute per instance)
    • GPUUtilization (% GPU usage)
    • Custom CloudWatch metrics

    Example scaling policy:
      Min instances: 1           (always at least 1 running)
      Max instances: 5           (scale up to 5 during peak)
      Target: 70% GPU utilization (add instance when GPU > 70%)
      Scale-in cooldown: 300s    (wait 5 min before scaling down)

AZURE ML MANAGED ENDPOINTS:
  ┌─────────────────────────────────────────────────────────────────┐
  │ Similar to SageMaker but on Azure:                              │
  │   1. Register model in Azure ML workspace                       │
  │   2. Define endpoint + deployment configuration                 │
  │   3. Choose VM size (Standard_NC4as_T4_v3 ≈ T4, ~$0.53/hr)    │
  │   4. Deploy → get endpoint URL + API key                        │
  │                                                                 │
  │ Blue/green deployment: run two model versions simultaneously    │
  │ and route traffic % between them (e.g., 90% v1, 10% v2).       │
  └─────────────────────────────────────────────────────────────────┘

GCP VERTEX AI:
  ┌─────────────────────────────────────────────────────────────────┐
  │ Google Cloud's managed ML platform:                             │
  │   1. Upload model to GCS (Google Cloud Storage)                 │
  │   2. Create Model resource → Deploy to Endpoint                 │
  │   3. Choose machine type (n1-standard-4 + T4, ~$0.54/hr)       │
  │   4. Supports auto-scaling with traffic-split for A/B testing   │
  │                                                                 │
  │ Vertex AI also supports direct HF model imports.                │
  └─────────────────────────────────────────────────────────────────┘
```

### Strategy 3: Container-Based Deployment (Maximum Control)

```
WHAT:
  Package your model + inference server into a Docker container.
  Deploy it anywhere — ECS, EKS (Kubernetes), or bare EC2.

WHEN TO USE:
  • Need full control over the serving stack
  • Multi-model serving (multiple adapters on one base model)
  • Custom preprocessing/postprocessing pipelines
  • Existing Kubernetes infrastructure
  • Want to avoid vendor lock-in

THE STACK:
  ┌─────────────────────────────────────────────────────────────────┐
  │ INFERENCE SERVER (pick one):                                    │
  │                                                                 │
  │ vLLM (recommended for production)                               │
  │   • PagedAttention — 2-4× higher throughput than HF generate() │
  │   • Continuous batching — serves multiple users simultaneously  │
  │   • OpenAI-compatible API out of the box                        │
  │   • Supports LoRA adapters natively (no merging needed!)        │
  │   • python -m vllm.entrypoints.openai.api_server \              │
  │       --model Qwen/Qwen2.5-1.5B-Instruct \                     │
  │       --enable-lora \                                           │
  │       --lora-modules healthcare=jeev1992/healthcare-lora-v2     │
  │                                                                 │
  │ Text Generation Inference (TGI) by Hugging Face                 │
  │   • Built by HF, optimized for their model ecosystem            │
  │   • Flash Attention, continuous batching, quantization           │
  │   • Docker-first design — easy to deploy on any cloud           │
  │   • docker run ghcr.io/huggingface/text-generation-inference \  │
  │       --model-id Qwen/Qwen2.5-1.5B-Instruct                    │
  │                                                                 │
  │ Ollama (simplest for local/edge)                                │
  │   • Single binary, no Python needed                             │
  │   • Great for local dev, edge deployment                        │
  │   • Limited production features (no auto-scaling, no batching)  │
  └─────────────────────────────────────────────────────────────────┘

AWS DEPLOYMENT OPTIONS:
  ┌─────────────────────────────────────────────────────────────────┐
  │ ECS (Elastic Container Service) + Fargate/EC2                   │
  │   • Docker containers on AWS-managed infrastructure             │
  │   • GPU support via EC2 launch type (g4dn, g5 instances)        │
  │   • Auto-scaling based on CloudWatch metrics                    │
  │   • Load balancer (ALB) distributes traffic across containers   │
  │   • Good for: teams without Kubernetes expertise                │
  │                                                                 │
  │ EKS (Elastic Kubernetes Service)                                │
  │   • Managed Kubernetes on AWS                                   │
  │   • GPU node groups with NVIDIA device plugin                   │
  │   • Horizontal Pod Autoscaler (HPA) for scaling                 │
  │   • Good for: teams already using Kubernetes                    │
  │                                                                 │
  │ EC2 (bare instance)                                             │
  │   • Launch a GPU instance, SSH in, run vLLM/TGI                 │
  │   • Maximum control, minimum abstraction                        │
  │   • Manual scaling (or custom ASG + Launch Template)            │
  │   • Good for: quick experiments, single-instance deployments    │
  └─────────────────────────────────────────────────────────────────┘
```

### Strategy 4: Model-as-a-Service (No Infrastructure)

```
WHAT:
  Use a cloud provider's built-in LLM hosting. Upload your model,
  they handle everything — GPU, scaling, API, monitoring.

AWS BEDROCK CUSTOM MODELS:
  ┌─────────────────────────────────────────────────────────────────┐
  │ • Fine-tune or import a model directly in Bedrock               │
  │ • Fully managed — no instances, no containers, no GPUs          │
  │ • Pay per token (input + output)                                │
  │ • Auto-scales automatically                                     │
  │ • Supports Llama, Titan, Mistral (not Qwen as of 2024)         │
  │                                                                 │
  │ LIMITATION: Only supports specific base models.                 │
  │ If your model isn't in their catalog, you can't use Bedrock.    │
  │ For Qwen → use SageMaker or container-based deployment.         │
  └─────────────────────────────────────────────────────────────────┘

AZURE OPENAI SERVICE:
  ┌─────────────────────────────────────────────────────────────────┐
  │ • Fine-tune GPT-4o, GPT-4o-mini directly in Azure              │
  │ • Fully managed — deploy as an endpoint in minutes              │
  │ • Pay per token                                                 │
  │ • Enterprise features: RBAC, VNet, content filtering            │
  │                                                                 │
  │ LIMITATION: Only OpenAI models. Can't bring Qwen/Llama.        │
  └─────────────────────────────────────────────────────────────────┘
```

### Scaling Strategies Compared

```
STRATEGY              COLD START    COST MODEL        SCALING       BEST FOR
──────────────────────────────────────────────────────────────────────────────
Serverless            30-120s       Pay-per-request   Auto          Low traffic, prototyping
(SageMaker Serverless,                                              Budget-sensitive
 HF Endpoints)

Dedicated Endpoint    None          Pay-per-hour      Auto/Manual   Consistent traffic,
(SageMaker Real-time,                                               latency-sensitive,
 Azure ML, Vertex AI)                                               production SLAs

Container-based       None          Pay-per-hour      Auto (HPA,    Multi-model serving,
(vLLM/TGI on ECS,     (if pre-     (instance cost)   ASG) or       custom pipelines,
 EKS, EC2)             warmed)                        manual        Kubernetes teams

Model-as-a-Service    None          Pay-per-token     Auto          Supported models only,
(Bedrock, Azure                                                     zero-ops preference
 OpenAI)
```

### Cost Optimization Strategies

```
1. RIGHT-SIZE YOUR INSTANCE
   Don't use an A100 for a 1.5B model. Our model in 4-bit fits on a T4 (16 GB).
   Over-provisioning is the #1 cost mistake.

   Model Size    Quantization    Min GPU           AWS Instance         $/hr
   ─────────────────────────────────────────────────────────────────────────
   1.5B          4-bit           T4 (16 GB)        ml.g4dn.xlarge       $0.74
   3B            4-bit           T4 (16 GB)        ml.g4dn.xlarge       $0.74
   7B            4-bit           A10G (24 GB)      ml.g5.xlarge         $1.41
   13B           4-bit           A10G (24 GB)      ml.g5.2xlarge        $1.52
   70B           4-bit           A100 (80 GB)      ml.p4d.24xlarge      $37.69

2. USE SPOT/PREEMPTIBLE INSTANCES (for non-critical workloads)
   AWS Spot:   60-90% cheaper than on-demand
   GCP Preemptible: 60-91% cheaper
   Azure Spot: 60-90% cheaper

   Catch: instance can be reclaimed with 2 minutes notice.
   Good for: batch inference, dev/test. Bad for: real-time production APIs.

3. SCALE TO ZERO WHEN IDLE
   If traffic is bursty (high during business hours, zero at night):
   • SageMaker Serverless: auto-scales to zero
   • ECS/EKS: set min instances to 0, auto-scale on request count
   • Scheduled scaling: scale down at 8 PM, scale up at 7 AM

4. USE AN INFERENCE SERVER (vLLM, TGI)
   Raw HF generate() processes one request at a time.
   vLLM with continuous batching: 2-4× more requests per GPU.
   Same GPU, same cost, 4× more throughput = 4× lower cost per request.

5. CACHE COMMON RESPONSES
   If the same questions are asked repeatedly (FAQ-style):
   • Add a Redis/ElastiCache layer in front of the model
   • Hash the prompt → check cache → return cached response
   • Cache hit = zero GPU cost, < 10ms latency

6. MERGE THE ADAPTER FOR DEPLOYMENT
   Loading base model + adapter separately takes slightly longer.
   For production, merge the adapter into the base weights:
     merged_model = ft_model.merge_and_unload()
   One fewer network call, slightly faster first inference.
```

### Deploying Our Healthcare Model to SageMaker (Example)

```python
# This is a reference example — NOT part of the workshop notebooks.
# It shows the pattern for deploying a merged LoRA model to SageMaker.

import sagemaker
from sagemaker.huggingface import HuggingFaceModel

role = sagemaker.get_execution_role()

# Step 1: Upload merged model to S3 (or use HF Hub model ID)
hub_config = {
    "HF_MODEL_ID": "jeev1992/healthcare-assistant-merged",  # merged model on HF
    "HF_TASK": "text-generation",
}

# Step 2: Create HuggingFace Model object
huggingface_model = HuggingFaceModel(
    env=hub_config,
    role=role,
    transformers_version="4.45.0",
    pytorch_version="2.5.0",
    py_version="py311",
)

# Step 3: Deploy to a real-time endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge",     # T4 GPU — sufficient for 1.5B
    endpoint_name="healthcare-assistant",
)

# Step 4: Send a request
response = predictor.predict({
    "inputs": "What are the symptoms of Type 2 diabetes?",
    "parameters": {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
    }
})
print(response)

# Step 5: Clean up (IMPORTANT — endpoints cost money when running!)
# predictor.delete_endpoint()
```

### The Decision Flowchart

```
"I need to deploy my fine-tuned model to production"
  │
  ▼
"How much traffic do I expect?"
  │
  ├── < 100 requests/day
  │     │
  │     ├── "Can I tolerate 30-60s cold starts?"
  │     │     ├── YES → Serverless (SageMaker Serverless or HF Endpoints)
  │     │     └── NO  → Dedicated endpoint with scale-to-zero schedule
  │     │
  │     └── Budget: $0-50/month
  │
  ├── 100-10,000 requests/day
  │     │
  │     ├── "Do I need custom preprocessing or multi-model serving?"
  │     │     ├── YES → Container-based (vLLM on ECS/EKS)
  │     │     └── NO  → Dedicated endpoint (SageMaker Real-time)
  │     │
  │     └── Budget: $50-500/month
  │
  └── 10,000+ requests/day
        │
        ├── "Do I need maximum throughput?"
        │     ├── YES → vLLM/TGI on GPU cluster (EKS + auto-scaling)
        │     └── NO  → SageMaker multi-instance endpoint + auto-scaling
        │
        └── Budget: $500+/month, consider reserved instances for savings
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| **Adapter architecture** | Adapter = config + weights (~20-50 MB). Base model downloaded separately. |
| **Hub loading** | `PeftModel.from_pretrained(base_model, "repo_id")` — two-step process |
| **Quantization match** | Inference config MUST match training config. Different quant = wrong results. |
| **Adapter toggle** | `disable/enable_adapter_layers()` — one model, two behaviors, no extra VRAM |
| **Generation** | Autoregressive (token by token). KV cache for speed. `max_new_tokens` limits output. |
| **Inference API** | Free tier: base models only. Custom adapters need local GPU or paid endpoint. |
| **Inference Endpoints** | Paid deployment. Requires merged model. ~$0.60/hr for T4. |
| **Merging** | Bakes adapter into base weights. Use only for deployment, not for comparison. |
| **Export** | JSON with prompts + base outputs + fine-tuned outputs. Decouples GPU from eval. |
| **Cloud deployment** | Serverless (low traffic), dedicated GPU (production), container-based (full control), or model-as-a-service (zero-ops). Right-size instances and use vLLM for throughput. |

---

*Previous: [Module 2 — Colab Fine-Tuning ←](../module_2_colab_finetuning/notes.md)*  
*Next: [Module 4 — LangSmith Evaluation →](../module_4_langsmith_eval_observability/notes.md)*
