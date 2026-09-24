# Healthcare Agent Fine-Tuning Workshop

4-module hands-on workshop: fine-tune a healthcare assistant using QLoRA,
deploy to Hugging Face, and evaluate with LangSmith.

1. Strategy + Dataset
2. Colab Fine-Tuning (QLoRA)
3. Hugging Face Deploy + Inference
4. LangSmith Evaluation + Observability

## Prerequisites — Accounts & Tokens

### Hugging Face (required for Modules 2 & 3)
1. Create a free account at [huggingface.co/join](https://huggingface.co/join)
2. Create a **Write** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Click **New token** → name it → select **Write** → Create
   - Copy the token (starts with `hf_...`)
3. Your **HF_REPO_ID** is `your-username/repo-name` (e.g. `jeev1992/healthcare-assistant-lora`)
   - You do NOT need to create the repo manually — the notebook creates it when you push

### LangSmith (required for Module 4)
1. Sign up (free Developer plan) at [smith.langchain.com](https://smith.langchain.com)
2. Go to Settings → API Keys → Create API Key
3. Copy the key (starts with `lsv2_pt_...`)

### OpenAI (required for Module 4 evaluators)
1. Get an API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Copy the key (starts with `sk-...`)

### Google Colab (required for Module 2 — Colab path)
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. You need a Google account (free)
3. Runtime → Change runtime type → **T4 GPU**

### RunPod (required for Module 2 — RunPod path)
1. Create an account at [runpod.io](https://runpod.io)
2. Add a payment method (pay-per-use GPU pods)
3. See [runpod/README.md](module_2_colab_finetuning/runpod/README.md) for full setup guide

## Structure

```text
healthcare-agent-finetuning-workshop/
├── module_1_strategy_data/
│   ├── notes.md
│   ├── dataset_quality_exercise.ipynb  ← hands-on: audit data + prompt engineering baseline
│   ├── exercises.ipynb                 ← take-home exercises (4 exercises)
│   └── exercise_solutions.ipynb        ← solutions with worked answers
├── module_2_colab_finetuning/
│   ├── README.md
│   ├── notes.md
│   ├── .env
│   ├── notebooks/
│   │   ├── training_v1.ipynb           ← 1.5B + ChatDoctor ("don't fine-tune capable models")
│   │   └── training_v2.ipynb           ← 1.5B + reformatted WikiDoc ("data quality matters")
│   ├── scripts/
│   │   └── data_prep_v2.py             ← async script to reformat WikiDoc via GPT-4o-mini
│   ├── results/
│   │   ├── benchmark_results_v1.json   ← v1 before/after outputs (1.5B + ChatDoctor)
│   │   └── benchmark_results_v2.json   ← v2 before/after outputs (1.5B + WikiDoc)
│   ├── exercises.ipynb                 ← take-home exercises (4 exercises)
│   ├── exercise_solutions.ipynb        ← solutions with worked answers
│   └── runpod/
│       ├── README.md
│       ├── requirements.txt
│       ├── train.py
│       └── training_runpod.ipynb
├── module_3_hf_deploy_inference/
│   ├── notes.md
│   ├── hf_inference.ipynb
│   ├── exercises.ipynb                 ← take-home exercises (3 exercises)
│   └── exercise_solutions.ipynb        ← solutions with worked answers
├── module_4_langsmith_eval_observability/
│   ├── README.md
│   ├── notes.md
│   ├── notebooks/
│   │   └── langsmith_eval.ipynb
│   ├── exercises.ipynb                    ← take-home exercises (4 exercises)
│   ├── exercise_solutions.ipynb           ← solutions with worked answers
│   └── results/
│       ├── inference_results.json
│       ├── BaseModel-Vs-FineTuned-v1.csv
│       └── BaseModel-Vs-FineTuned-v2.csv
├── .gitignore
└── README.md
```
