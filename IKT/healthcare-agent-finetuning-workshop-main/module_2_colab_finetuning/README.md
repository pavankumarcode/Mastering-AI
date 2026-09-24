# Module 2: QLoRA Fine-Tuning — Healthcare Assistant

This module contains **two fine-tuning experiments** that together show you three key lessons
about when fine-tuning helps, which model to pick, and how data quality determines results:

| | v1 Notebook (1.5B + ChatDoctor) | v2 Notebook (1.5B + Reformatted WikiDoc) |
|---|---|---|
| **Model** | `Qwen2.5-1.5B-Instruct` | `Qwen2.5-1.5B-Instruct` |
| **Dataset** | `lavita/ChatDoctor-HealthCareMagic-100k` | `jeev1992/wikidoc-healthassist` (reformatted via GPT-4o-mini) |
| **Base model quality** | Already knows medicine well | Already knows medicine well |
| **Fine-tuning result** | ❌ Made it worse (persona leakage, style degradation) | ✅ Safety +0.10, Accuracy +0.06, but Helpfulness -0.16 (too concise) |
| **Key lesson** | Don't fine-tune a model that already knows your domain | Data quality is everything — clean data fixes persona/safety but can introduce brevity |

The two experiments deliberately differ on **both** model size and dataset so you can see all three lessons at once.

### The Three Teaching Points

**Lesson 1 — Only fine-tune when the model genuinely lacks your domain**
The 1.5B Qwen model already answered medical questions reasonably well. Fine-tuning it on ChatDoctor didn't add knowledge — it just baked in the chatbot persona. The before/after got *worse*, not better.

**Lesson 2 — Data quality is everything**
v2 uses the same 1.5B model as v1 but trains on WikiDoc data that was reformatted into conversational healthcare-assistant style via GPT-4o-mini. The reformatted data includes safety disclaimers, structured explanations, and no persona artifacts. The result: safety scores improved significantly while preserving the model's existing medical knowledge. Clean, well-formatted data is the difference between fine-tuning that helps and fine-tuning that hurts.

**Lesson 3 — Know when NOT to fine-tune**
For general medical Q&A, a capable base model with good prompting may already be sufficient. The Qwen 1.5B model already knows medicine — our v2 improvement was marginal (safety up, accuracy/helpfulness roughly flat) from just 2,000 clean rows. A well-crafted system prompt could achieve similar results. Fine-tuning's real power is **domain adaptation on domains the model has never seen** — proprietary clinical protocols, internal drug formularies, rare disease data, or regulatory compliance language that simply isn't in the model's weights. This workshop teaches you the mechanics so you can make that judgment call yourself.

There are also **two deployment paths** depending on your hardware:

## Colab vs RunPod — At a Glance

| | Colab (Free T4) | RunPod (A100/A10G) |
|---|---|---|
| **File** | `training_v1.ipynb` / `training_v2.ipynb` | `runpod/train.py` + `runpod/training_runpod.ipynb` |
| **Platform** | Google Colab (free) | RunPod GPU Pods (paid, ~$0.36-1.64/hr) |
| **Format** | Jupyter notebook (interactive) | CLI script or Jupyter notebook |
| **GPU** | T4 15 GB VRAM | A100 40/80 GB or A10G 24 GB |
| **Dataset** | 2,000 samples (subset) | Full ~33k samples |
| **LoRA rank** | r=16, alpha=32 | r=64, alpha=128 |
| **Max sequence length** | 512 tokens | 1024 tokens |
| **Flash Attention 2** | No (T4 doesn't support it) | Yes |
| **Gradient checkpointing** | No | Yes |
| **WandB logging** | No | Yes |
| **Best model selection** | No | Yes (by eval loss) |
| **How to run** | Upload to Colab → Run All | `python train.py --hf_repo_id user/repo` |
| **Config** | Edit variables in a cell | CLI flags (e.g. `--lora_r 128`) |
| **Training time** | ~20-30 min | ~15-30 min (A100) / ~45-60 min (A10G) |
| **Cost** | Free | ~$0.45-1.50 |

## Which One Should I Use?

- **Start with Colab** — it's free, runs in the browser, and is perfect for learning
  the QLoRA workflow. The 2,000-sample subset trains in ~20 min and shows clear
  before/after comparison on v2 (1.5B + reformatted WikiDoc).

- **Use RunPod** when you want better results — full dataset, higher LoRA rank,
  longer sequences, WandB charts, and Flash Attention 2 for speed. Good for a
  "production-quality" fine-tune or for seeing what a real training run looks like.

> **⚠️ Colab Free-Tier Timeout:** Training on 2,000 samples frequently causes session
> timeouts on Colab's free tier (the runtime disconnects mid-training). If you hit this,
> switch to **RunPod** — even a cheap A10G pod (~$0.36/hr) completes the same run
> reliably in ~45-60 min without disconnection issues. See [runpod/README.md](runpod/README.md)
> for setup instructions.

## Files

```
module_2_colab_finetuning/
├── README.md                          ← This file
├── notes.md                           ← Detailed reference notes for this module
├── .env                               ← Environment variables (HF token, repo ID)
├── notebooks/
│   ├── training_v1.ipynb              ← v1: 1.5B + ChatDoctor ("don't fine-tune capable models")
│   └── training_v2.ipynb              ← v2: 1.5B + reformatted WikiDoc ("data quality matters")
├── scripts/
│   └── data_prep_v2.py                ← async script to reformat WikiDoc via GPT-4o-mini
├── results/
│   ├── benchmark_results_v1.json      ← v1 before/after outputs (1.5B + ChatDoctor)
│   └── benchmark_results_v2.json      ← v2 before/after outputs (1.5B + reformatted WikiDoc)
├── exercises.ipynb                    ← take-home exercises (4 exercises, 2 no-GPU + 2 GPU)
├── exercise_solutions.ipynb           ← solutions with worked answers
└── runpod/
    ├── README.md                      ← RunPod-specific setup guide
    ├── requirements.txt               ← Python dependencies
    ├── train.py                       ← CLI training script
    └── training_runpod.ipynb          ← RunPod Jupyter notebook
```

## Quick Start — Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook** → upload a notebook from `notebooks/` (start with v1)
3. **Runtime → Change runtime type** → **T4 GPU**
4. Change `HF_REPO_ID` in the Configuration cell to your username
5. **Runtime → Run all** (~20-30 min)

## Quick Start — RunPod

1. Create a GPU pod at [runpod.io](https://runpod.io) (see [runpod/README.md](runpod/README.md))
2. Upload `training_runpod.ipynb` to JupyterLab, or SSH in and run:
   ```bash
   pip install -r runpod/requirements.txt
   python runpod/train.py --hf_repo_id your-user/healthcare-assistant-lora
   ```

## What Happens During Training

Both paths follow the same pipeline:

1. **GPU check** — verify CUDA is available
2. **Install dependencies** — TRL, PEFT, bitsandbytes, etc.
3. **Load dataset** — v1: `ChatDoctor-HealthCareMagic-100k`, v2: `jeev1992/wikidoc-healthassist` (pre-reformatted)
4. **Format to chat** — map columns to system/user/assistant messages
5. **Load base model** — 4-bit quantization via BitsAndBytesConfig
6. **BEFORE benchmark** — run 10 fixed healthcare prompts through the base model
7. **Configure LoRA** — attach trainable adapters to all linear layers
8. **Train with SFTTrainer** — supervised fine-tuning with cosine LR schedule
9. **Save adapter** — small ~10-50 MB adapter (not the full model)
10. **AFTER benchmark** — same 10 prompts through the fine-tuned model
11. **Compare** — side-by-side table of base vs fine-tuned outputs
12. **Push to HF Hub** — adapter available for Module 3 (inference) and Module 4 (eval)
13. **Export JSON** — benchmark results saved for downstream use

## Prerequisites

- **Hugging Face account** + Write token ([get one here](https://huggingface.co/settings/tokens))
- **Google account** (for Colab) or **RunPod account** (for RunPod)
- **WandB account** (optional, RunPod only) ([wandb.ai](https://wandb.ai))
