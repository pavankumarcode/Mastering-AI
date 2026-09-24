# Instructor Guide — Healthcare Agent Fine-Tuning Workshop

---

## Workshop at a Glance

| Detail | Value |
|---|---|
| **Duration** | ~4 hours (with breaks) |
| **Format** | Lecture + live coding + hands-on exercises |
| **Level** | Intermediate (students should know Python, basic ML concepts) |
| **Total content** | ~420 KB across 4 modules |
| **Take-home exercises** | 15 exercises across 4 modules (with solutions) |
| **Cost to students** | Free (Colab) or ~$1-2 (RunPod + OpenAI calls) |

### What Students Will Build

A healthcare Q&A assistant fine-tuned with QLoRA on a 1.5B parameter model,
deployed to Hugging Face Hub, and evaluated with LangSmith — end to end in one session.

### The Story Arc

The workshop is designed around **deliberate failure**:

1. **Module 1** — Audit two datasets. One is clean. One is 63% contaminated.
2. **Module 2** — Fine-tune on both. The contaminated one makes the model *worse*.
3. **Module 3** — Deploy and run inference. See the outputs side by side.
4. **Module 4** — Prove it with numbers. Evaluation catches what vibes can't.

Students leave understanding not just HOW to fine-tune, but WHEN to fine-tune
and WHAT can go wrong.

---

## Pre-Workshop Checklist

### Accounts Students Must Create Before Class

Send this list 1 week before:

| Account | Required For | Setup Time | Link |
|---|---|---|---|
| **Hugging Face** (Write token) | Modules 2 & 3 | 2 min | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| **Google account** (Colab) | Module 2 | Already have one | [colab.research.google.com](https://colab.research.google.com) |
| **LangSmith** (free Developer plan) | Module 4 | 3 min | [smith.langchain.com](https://smith.langchain.com) |
| **OpenAI** (API key with credits) | Module 4 | 5 min | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| **RunPod** (optional, if Colab times out) | Module 2 fallback | 5 min | [runpod.io](https://runpod.io) |

> **Common blocker:** Students forget to create the HF Write token (not Read).
> Remind them twice — once in the pre-class email, once at the start of class.

### Instructor Environment Setup

1. Run through both training notebooks end-to-end the day before
2. Have pre-computed `benchmark_results_v1.json` and `benchmark_results_v2.json` ready
   (in `module_2_colab_finetuning/results/`) in case Colab times out during live demo
3. Have a LangSmith project with completed experiments to show the dashboard
4. Test that `exercises.ipynb` cells run without dependency issues

---

## Suggested Schedule

| Time | Duration | Module | Activity |
|---|---|---|---|
| 0:00 | 10 min | — | Welcome, agenda, verify accounts |
| 0:10 | 50 min | **Module 1** | Strategy lecture + dataset audit exercise |
| 1:00 | 10 min | — | **Break** |
| 1:10 | 70 min | **Module 2** | QLoRA lecture + live training demo |
| 2:20 | 10 min | — | **Break** |
| 2:30 | 30 min | **Module 3** | Deployment + inference demo |
| 3:00 | 50 min | **Module 4** | Evaluation lecture + live LangSmith demo |
| 3:50 | 10 min | — | Wrap-up, hand out take-home exercises |

**Total:** 4 hours including breaks.

### Time Pressure Tips

- If running short: skip v1 training live, show pre-computed results instead
- Module 3 can be compressed to 20 min if you load the adapter from the
  pre-existing HF repo (`jeev1992/healthcare-assistant-lora-v2`)
- Module 4 evaluation runs in 2-3 min — don't skip the live demo, it's the payoff

---

## Module-by-Module Teaching Guide

---

### Module 1: Strategy & Dataset Preparation (50 min)

**Files:**
- Lecture notes: [module_1_strategy_data/notes.md](module_1_strategy_data/notes.md)
- In-class exercise: [module_1_strategy_data/dataset_quality_exercise.ipynb](module_1_strategy_data/dataset_quality_exercise.ipynb)

#### Lecture Outline (25 min)

**1. The Three Approaches to Domain Adaptation (10 min)**

| Approach | Cost | Effort | Reversible | When to Use |
|---|---|---|---|---|
| Prompt engineering | $0 (tokens only) | Hours | Yes | First attempt; sufficient 80% of the time |
| RAG | Vector DB costs | Days | Yes | Need current info or source attribution |
| Fine-tuning | GPU compute | Days-weeks | No | Need consistent style, format, or rare domain |

**Golden rule:** Prompt engineering first → RAG second → Fine-tuning third.

**2. When Fine-Tuning Actually Helps (5 min)**

Fine-tuning is for:
- Consistent output format/tone baked into weights
- Domain-specific terminology USAGE (not knowledge — the model already knows)
- Reduced token cost at scale (shorter prompts needed)
- Latency-sensitive apps

Fine-tuning does NOT help when:
- You need up-to-date info (RAG better)
- You need source citations (RAG better)
- Simple formatting works (prompting sufficient)
- Low-volume usage (ROI doesn't justify cost)

**3. QLoRA in 5 Minutes (5 min)**

Draw this on the board:

```
Full Fine-Tuning:     Update ALL 1.5B parameters (needs A100, 40GB+ VRAM)
                           │
LoRA:                 Freeze weights. Add small trainable side-paths.
                      Only train ~9M params (0.6% of model). Needs ~6GB.
                           │
QLoRA:                Same as LoRA, but compress frozen model to 4-bit first.
                      3 GB model → 0.75 GB. Now fits on free Colab T4 (15GB).
```

**Key insight for students:** QLoRA is why this workshop is possible. Before 2023,
you needed a $5K+ GPU. Now Google gives you one for free.

**4. Dataset Quality Preview (5 min)**

Show these numbers (students will verify them in the exercise):

| Metric | ChatDoctor (v1) | WikiDoc Reformatted (v2) |
|---|---|---|
| Total examples | 112,165 | 2,100 |
| Persona contamination | 63.1% | 0.0% |
| Boilerplate sign-offs | 28.2% | 0.0% |
| Safety disclaimers | 3.2% | 99.4% |
| Avg answer length | 603 chars | 910 chars |

Ask the class: *"If 63% of your training data starts with 'Hi, welcome to Chat Doctor,'
what will your model learn to say?"*

#### Hands-On Exercise (25 min)

Have students open `dataset_quality_exercise.ipynb` and work through:
- Part A: Load ChatDoctor, run regex audit, measure contamination
- Part B: Load WikiDoc reformatted, compare metrics
- Part C: Test prompt engineering baseline (system prompt only, no fine-tuning)

**Circulate and help with:**
- Dataset loading issues (HF may rate-limit if many students load simultaneously)
- Students unsure what regex patterns to use (give hints, don't give answers)

---

### Module 2: QLoRA Fine-Tuning (70 min)

**Files:**
- Lecture notes: [module_2_colab_finetuning/notes.md](module_2_colab_finetuning/notes.md)
- README with comparison table: [module_2_colab_finetuning/README.md](module_2_colab_finetuning/README.md)
- Training notebooks: `notebooks/training_v1.ipynb`, `notebooks/training_v2.ipynb`

#### Lecture Outline (20 min)

**1. The Two Experiments (5 min)**

Both use the same model (`Qwen2.5-1.5B-Instruct`), same LoRA config, same
hyperparameters. The ONLY difference is the dataset:

| | v1 | v2 |
|---|---|---|
| Dataset | ChatDoctor (noisy, 63% contaminated) | WikiDoc (clean, reformatted) |
| Expected result | Model should get worse | Model should improve |

**2. Hyperparameters Explained (10 min)**

Walk through each parameter — students don't need to memorize, but should
understand the intuition:

```python
# LoRA Configuration
LoraConfig(
    r=16,                    # Rank: width of the side-path. Higher = more capacity.
    lora_alpha=32,           # Scaling: alpha/r = 2.0. Controls adapter influence.
    lora_dropout=0.05,       # Regularization: randomly drops 5% of adapter neurons.
    target_modules="all-linear",  # Attach to ALL linear layers (7 per transformer block).
)

# Training Configuration
learning_rate = 5e-5    # How big each gradient step is. Lower = more careful.
num_train_epochs = 2    # How many passes through the data.
warmup_steps = 50       # Ramp up LR gradually for first 50 steps.
```

**Training step calculation** (do this live):
- Dataset: 2,000 examples
- Batch size: 8 (effective, with gradient accumulation)
- Steps per epoch: 2,000 ÷ 8 = 250
- v1: 250 × 3 epochs = 750 total steps, warmup = 75/750 = 10%
- v2: 250 × 2 epochs = 500 total steps, warmup = 50/500 = 10%

**3. 4-Bit Quantization (5 min)**

```
Model size in fp16:     1.5B params × 2 bytes = 3.0 GB
Model size in 4-bit:    1.5B params × 0.5 bytes = 0.75 GB
Savings:                73% VRAM reduction
```

The quantized model is "frozen" — we NEVER update the base weights. LoRA adapters
(~9M params in fp16) ride on top and DO get updated.

#### Live Demo: Training v2 (30 min)

> **Recommended:** Demo v2 (WikiDoc) live. Show v1 (ChatDoctor) results from
> pre-computed files to save time.

1. Open `training_v2.ipynb` in Colab
2. Set runtime to T4 GPU
3. Walk through cells while they run:
   - GPU check → "We have 15 GB VRAM, plenty for a 0.75 GB quantized model"
   - Dataset loading → "2,000 examples, each with system/user/assistant messages"
   - BEFORE benchmark → "These are the base model's answers. Notice: already decent!"
   - LoRA config → "We're adding 9M trainable params — 0.6% of the model"
   - Training → "Watch the loss decrease. This takes ~20-30 min on T4."
   - AFTER benchmark → "Now compare side by side. What changed?"

> **⚠️ Colab Timeout Warning:** Training 2,000 samples frequently causes free-tier
> timeouts. If this happens during your demo:
> 1. Show pre-computed results from `results/benchmark_results_v2.json`
> 2. Tell students to use RunPod (~$0.36/hr) for reliable training
> 3. See [runpod/README.md](module_2_colab_finetuning/runpod/README.md) for setup

**While training runs (20 min), do one of:**
- Show v1 pre-computed results and discuss why it failed
- Walk through the data_prep_v2.py script (how WikiDoc was reformatted)
- Take questions about LoRA math / quantization
- Have students start Module 1 take-home exercises

#### Showing v1 Results (10 min)

Open `results/benchmark_results_v1.json` and show 2-3 examples:

```
BEFORE (base model):
"Type 2 diabetes symptoms include increased thirst, frequent urination,
blurred vision, fatigue, and slow-healing wounds. These symptoms develop
gradually. Please consult a healthcare professional for proper diagnosis."

AFTER (v1 fine-tuned):
"Hello, thanks for posting your query. The symptoms of diabetes include
increased urination and thirst. Hope this helps. Wishing you good health."
```

Ask: *"Which answer would you rather get from a healthcare chatbot?"*

The base model was better. Fine-tuning taught it the ChatDoctor persona.
This is the key lesson.

#### Three Teaching Points to Emphasize

1. **Don't fine-tune a model that already knows your domain** — v1 proved this
2. **Data quality is everything** — v2's clean data improved safety (+0.10)
3. **Know when NOT to fine-tune** — prompting might be sufficient for general medical Q&A

---

### Module 3: Deployment & Inference (30 min)

**Files:**
- Lecture notes: [module_3_hf_deploy_inference/notes.md](module_3_hf_deploy_inference/notes.md)
- Notebook: [module_3_hf_deploy_inference/hf_inference.ipynb](module_3_hf_deploy_inference/hf_inference.ipynb)

#### Lecture Outline (15 min)

**1. What Lives on HF Hub (3 min)**

The adapter is a PATCH, not the full model:

```
Your HF repo (~20-50 MB):              Base model (stays at HF, ~3 GB):
├── adapter_config.json                 Qwen/Qwen2.5-1.5B-Instruct
├── adapter_model.safetensors           (downloaded automatically)
├── tokenizer.json
└── special_tokens_map.json
```

At runtime: download base (cached) + download adapter + merge = <1 second.

**2. The Adapter Toggle Pattern (3 min)**

This is one of the most useful techniques:

```python
# Two models in one — no extra VRAM
model.enable_adapter_layers()    # Fine-tuned behavior
model.disable_adapter_layers()   # Base model behavior
```

Saves ~1 GB VRAM vs loading two separate models.

**3. Generation Parameters (4 min)**

| Parameter | Healthcare Recommendation | Why |
|---|---|---|
| `temperature` | 0.1–0.3 | Medical info must be consistent; high temp causes hallucinations |
| `top_p` | 0.5–0.7 | Restrict to high-probability tokens |
| `max_new_tokens` | 256–512 | Long enough for thorough answers, short enough to prevent rambling |

**4. Production Deployment & Cloud Scaling (5 min — lecture only)**

This is a "where do you go from here" overview. Students won't deploy to cloud
in class, but they need to know the landscape. Walk through this table quickly:

| Strategy | Examples | Cold Start | Cost Model | Best For |
|---|---|---|---|---|
| **Serverless** | SageMaker Serverless, HF Endpoints | 30-120s | Pay-per-request | <100 req/day, prototyping |
| **Dedicated GPU** | SageMaker Real-time, Azure ML, GCP Vertex AI | None | Pay-per-hour | Production with SLAs |
| **Container-based** | vLLM/TGI on ECS, EKS, EC2 | None (pre-warmed) | Pay-per-hour | Multi-model, custom pipelines, K8s teams |
| **Model-as-a-Service** | AWS Bedrock, Azure OpenAI | None | Pay-per-token | Zero-ops (but limited model catalog) |

**Key points to mention:**

- **Right-size your instance:** Our 1.5B model in 4-bit needs only a T4 ($0.74/hr on
  SageMaker). Using an A100 ($37/hr) would waste 98% of the GPU.
- **vLLM is a game-changer:** Raw `model.generate()` processes one request at a time.
  vLLM with continuous batching gives 2-4× throughput on the same GPU.
- **Adapter merging for production:** In class we load base + adapter separately
  (for toggle convenience). In production, merge them:
  `merged_model = ft_model.merge_and_unload()` — one fewer network call.
- **Scale-to-zero saves money:** For bursty traffic (e.g., clinic hours only),
  SageMaker Serverless or scheduled ECS scaling avoids 24/7 GPU cost.
- **Healthcare compliance:** AWS, Azure, and GCP all offer HIPAA-eligible
  configurations — but you must set them up correctly (VPC, encryption, audit logs).
  RunPod and generic GPU providers typically are NOT HIPAA-compliant.

**Decision flowchart** (draw on board or show from notes Section 14):

```
Traffic < 100/day + can tolerate cold starts → Serverless
Traffic > 100/day + need low latency         → Dedicated GPU endpoint
Need multi-model or custom pipeline          → Container (vLLM on ECS/EKS)
Model in provider's catalog + zero-ops       → Model-as-a-Service (Bedrock)
```

> **Detailed reference:** Section 14 of `module_3_hf_deploy_inference/notes.md`
> covers all four strategies with architecture diagrams, instance type tables,
> auto-scaling configurations, cost optimization tips, and a SageMaker
> deployment code example. Point students there for self-study.

#### Live Demo (20 min)

Walk through `hf_inference.ipynb`:
1. Load base model + adapter from HF Hub
2. Generate a response with adapter ON
3. Toggle adapter OFF, generate same question → show the difference
4. Run all 10 benchmark prompts, save results for Module 4

> If short on time, use the pre-computed `inference_results.json` in
> `module_4_langsmith_eval_observability/results/`.

---

### Module 4: LangSmith Evaluation (50 min)

**Files:**
- Lecture notes: [module_4_langsmith_eval_observability/notes.md](module_4_langsmith_eval_observability/notes.md)
- README: [module_4_langsmith_eval_observability/README.md](module_4_langsmith_eval_observability/README.md)
- Notebook: [module_4_langsmith_eval_observability/notebooks/langsmith_eval.ipynb](module_4_langsmith_eval_observability/notebooks/langsmith_eval.ipynb)

#### Lecture Outline (15 min)

**1. Why Evaluation is Non-Negotiable (5 min)**

Without evaluation:
- "Does fine-tuning help?" → Unclear
- "What improved?" → Subjective
- "Did anything get worse?" → Unknown

With LLM-as-Judge evaluation:
- Objective metrics on 3 dimensions
- Reproducible, shareable results
- Catches regressions before deployment
- Cost: ~$0.06 for 60 judge calls

**2. The Three Evaluators (5 min)**

| Evaluator | Measures | Score 5 Example | Score 1 Example |
|---|---|---|---|
| **Helpfulness** | Actionable, well-structured? | FAST mnemonic for stroke with clear steps | "Strokes are serious. Get help." |
| **Accuracy** | Medical facts correct? | Correct metformin mechanism (hepatic glucose) | Wrong mechanism (increases insulin) |
| **Safety** | Disclaimers, no advice-giving? | "Consult a healthcare provider for..." | "Take 500mg ibuprofen immediately" |

**3. LLM-as-Judge Pattern (5 min)**

```
Your model's output → GPT-4o-mini (judge) → Score 0-5 + reasoning
```

Why GPT-4o-mini?
- Smart enough to judge medical accuracy
- Cheap: $0.001 per evaluation call
- Never self-evaluate (that introduces bias)
- Temperature=0 for deterministic scoring

#### Live Demo (25 min)

Walk through `langsmith_eval.ipynb`:

1. **Create dataset** (2 min) — Upload 10 benchmark prompts to LangSmith
2. **Evaluate base model** (3 min) — Score 10 outputs × 3 evaluators = 30 calls
3. **Evaluate fine-tuned model** (3 min) — Same 30 calls
4. **Show LangSmith dashboard** (7 min) — This is the "wow" moment:
   - Open [smith.langchain.com](https://smith.langchain.com)
   - Navigate to project → Experiments tab
   - Show side-by-side comparison with deltas
   - Click into individual traces to show judge reasoning
5. **Discuss results** (10 min) — The real teaching happens here

#### Results Discussion Script

**Show v1 results first:**

| Metric | Base | Fine-Tuned v1 | Delta |
|---|---|---|---|
| Accuracy | 0.72 | 0.64 | **-0.08 ❌** |
| Helpfulness | 0.78 | 0.60 | **-0.18 ❌** |
| Safety | 0.70 | 0.64 | **-0.06 ❌** |

*"All three metrics regressed. The model got worse on every dimension.
Why? Because ChatDoctor's persona contamination taught the model bad habits.
This is Module 1's lesson proving itself with numbers."*

**Then show v2 results:**

| Metric | Base | Fine-Tuned v2 | Delta |
|---|---|---|---|
| Accuracy | 0.66 | 0.72 | **+0.06 ✅** |
| Helpfulness | 0.72 | 0.56 | **-0.16 ❌** |
| Safety | 0.76 | 0.86 | **+0.10 ✅** |

*"Better! Safety improved significantly — the model learned to include
disclaimers. Accuracy improved too. But helpfulness dropped. Why?"*

Pause. Let students guess.

*"The WikiDoc training data was reformatted to be 'concise and focused.'
The model learned that brevity. It gives 1-3 bullet points where the base
model gave a full paragraph. The LangSmith judge penalizes that as 'less helpful.'
This is a real trade-off you'd face in production."*

**Ask the class:** *"If you had to deploy one model to a hospital chatbot,
which would you choose — base, v1, or v2? Why?"*

Expected answers:
- v2 (safety matters most in healthcare, even at the cost of helpfulness)
- Base model (it was already good; fine-tuning didn't help enough)
- Both are valid — this is a judgment call, and that's the point

#### Wrap-Up (10 min)

Summarize the four key takeaways:

1. **Prompt engineering first, fine-tuning last** — try the cheap option first
2. **Data quality > dataset size** — 2,000 clean beats 112,000 dirty
3. **Fine-tuning changes style, not knowledge** — the model already knew medicine
4. **Always evaluate with metrics** — vibes lie, numbers don't

Hand out take-home exercises (see below).

---

## Take-Home Exercises Summary

### Module 1 Exercises (No GPU)

| # | Exercise | Key Skill |
|---|---|---|
| 1 | Audit a mental health counseling dataset | Dataset quality assessment |
| 2 | Filter ChatDoctor contamination, measure remaining size | Data cleaning trade-offs |
| 3 | Design a reformatting prompt for pediatrics | Prompt engineering for data prep |
| 4 | System prompt ablation study | Prompt engineering vs fine-tuning |

### Module 2 Exercises (2 no-GPU + 2 GPU)

| # | Exercise | GPU | Key Skill |
|---|---|---|---|
| 1 | Calculate training steps and predict hyperparameter effects | No | Hyperparameter intuition |
| 2 | Analyze benchmark result lengths, correlate with helpfulness | No | Data analysis |
| 3 | LoRA rank experiment (r=4 vs r=16) | T4 | Model capacity vs regularization |
| 4 | Training data ablation (200 / 500 / 2,000 examples) | T4 | Sample efficiency |

### Module 3 Exercises (2 GPU + 1 no-GPU)

| # | Exercise | GPU | Key Skill |
|---|---|---|---|
| 1 | Temperature and top_p exploration | T4 | Generation parameter tuning |
| 2 | Adapter enable/disable comparison on 3 questions | T4 | Before/after analysis |
| 3 | Inference cost calculator (HF vs AWS vs RunPod) | No | Production cost planning |

### Module 4 Exercises (No GPU, API keys needed)

| # | Exercise | Key Skill |
|---|---|---|
| 1 | Create a custom completeness evaluator | Evaluator design |
| 2 | Cross-version v1 vs v2 CSV analysis | Data analysis, decision-making |
| 3 | Design evaluation suite for legal domain (conceptual) | Transfer to new domains |
| 4 | Judge model comparison (GPT-4o-mini vs GPT-4o) | Evaluation reliability |

**Solution notebooks** (`exercise_solutions.ipynb`) are provided for all modules.
Distribute these after students have attempted the exercises — or make them
available immediately if students prefer self-paced learning.

---

## Common Issues & Troubleshooting

### Module 2: Training Issues

| Problem | Cause | Solution |
|---|---|---|
| **Colab timeout mid-training** | Free-tier disconnects after 30-90 min | Switch to RunPod (~$0.36/hr). See [runpod/README.md](module_2_colab_finetuning/runpod/README.md) |
| **CUDA out of memory** | Batch size too large | Reduce `per_device_train_batch_size` from 4 to 2 |
| **HF push fails** | Wrong token type (Read instead of Write) | Regenerate token with Write permission |
| **Training loss stuck** | Learning rate too low or data issue | Check dataset loaded correctly; try `lr=1e-4` |
| **"No module named bitsandbytes"** | Missing dependency | `!pip install bitsandbytes>=0.43.0` |

### Module 4: Evaluation Issues

| Problem | Cause | Solution |
|---|---|---|
| **LangSmith 401 error** | Bad API key | Regenerate at Settings → API Keys |
| **OpenAI rate limit** | Too many concurrent calls | Reduce batch size or add `time.sleep(1)` between calls |
| **JSON parse error from judge** | GPT-4o-mini returned markdown instead of JSON | Add "Return ONLY valid JSON, no markdown" to prompt |
| **Scores all 0.0** | Wrong column name in results JSON | Check that keys match: `base_output`, `finetuned_output` |

---

## Adapting the Workshop

### For a 2-Hour Version

- Module 1: 15 min lecture only (skip hands-on exercise)
- Module 2: 20 min lecture + show pre-computed results (skip live training)
- Module 3: 10 min (show adapter toggle concept, skip live inference)
- Module 4: 30 min (full demo — this is the payoff, don't skip)
- Assign all exercises as take-home

### For a Full-Day Version (6-8 hours)

- Extend Module 1: Students do the full dataset audit exercise in class
- Module 2: Run BOTH v1 and v2 training live
- Add: Students design their own evaluation prompt and run it
- Add: Class discussion on "when would you use RAG instead?"
- Module 4: Students create the custom completeness evaluator (Exercise 1) in class

### For Non-Healthcare Domains

The workshop structure transfers to any domain. Replace:
- Dataset: your domain's equivalent of ChatDoctor / WikiDoc
- Benchmark prompts: 10 representative questions from your domain
- Evaluators: adjust rubrics (e.g., "legal accuracy" instead of "medical accuracy")
- Safety: change disclaimers to match domain requirements

Module 4 Exercise 3 (legal domain design) is explicitly designed to teach
this transfer.

---

## Key Numbers to Remember

| Fact | Value |
|---|---|
| Base model | Qwen2.5-1.5B-Instruct |
| Model size (fp16) | 3.0 GB |
| Model size (4-bit) | 0.75 GB |
| LoRA trainable params | ~9M (0.6% of 1.5B) |
| Adapter size on disk | 10-50 MB |
| v1 dataset (ChatDoctor) | 112,165 examples, 63.1% contaminated |
| v2 dataset (WikiDoc reformatted) | 2,100 examples, 99.4% safety disclaimers |
| Training time (Colab T4) | ~20-30 min per experiment |
| Training time (RunPod A10G) | ~45-60 min per experiment |
| Evaluation cost (Module 4) | ~$0.06 (60 GPT-4o-mini calls) |
| Total student cost | $0 (Colab) to ~$2 (RunPod + OpenAI) |

---

## File Reference

```
healthcare-agent-finetuning-workshop/
├── README.md                              ← Student-facing overview + prerequisites
├── INSTRUCTOR_GUIDE.md                    ← You are here
│
├── module_1_strategy_data/
│   ├── notes.md                           ← Detailed lecture notes (61 KB)
│   ├── dataset_quality_exercise.ipynb     ← In-class exercise
│   ├── exercises.ipynb                    ← 4 take-home exercises
│   └── exercise_solutions.ipynb           ← Solutions
│
├── module_2_colab_finetuning/
│   ├── README.md                          ← Module overview + Colab vs RunPod comparison
│   ├── notes.md                           ← Detailed lecture notes (78 KB)
│   ├── .env                               ← HF token + repo ID (students fill in)
│   ├── notebooks/
│   │   ├── training_v1.ipynb              ← v1: 1.5B + ChatDoctor
│   │   └── training_v2.ipynb              ← v2: 1.5B + WikiDoc
│   ├── scripts/
│   │   └── data_prep_v2.py                ← WikiDoc reformatting script
│   ├── results/
│   │   ├── benchmark_results_v1.json      ← Pre-computed v1 outputs
│   │   └── benchmark_results_v2.json      ← Pre-computed v2 outputs
│   ├── runpod/                            ← RunPod alternative path
│   ├── exercises.ipynb                    ← 4 take-home exercises
│   └── exercise_solutions.ipynb           ← Solutions
│
├── module_3_hf_deploy_inference/
│   ├── notes.md                           ← Detailed lecture notes (49 KB)
│   ├── hf_inference.ipynb                 ← In-class inference notebook
│   ├── exercises.ipynb                    ← 3 take-home exercises
│   └── exercise_solutions.ipynb           ← Solutions
│
└── module_4_langsmith_eval_observability/
    ├── README.md                          ← Module overview
    ├── notes.md                           ← Detailed lecture notes (62 KB)
    ├── notebooks/
    │   └── langsmith_eval.ipynb           ← In-class evaluation notebook
    ├── results/
    │   ├── inference_results.json
    │   ├── BaseModel-Vs-FineTuned-v1.csv
    │   └── BaseModel-Vs-FineTuned-v2.csv
    ├── exercises.ipynb                    ← 4 take-home exercises
    └── exercise_solutions.ipynb           ← Solutions
```
