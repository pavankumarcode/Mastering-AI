# Module 4: LangSmith Evaluation & Observability

You've fine-tuned a model and generated before/after outputs. But how do you **prove**
it's better — with numbers, not vibes?

In this module you'll use **LangSmith** and **GPT-4o-mini as an LLM judge** to score your
model's outputs on three dimensions — helpfulness, accuracy, and safety — and see exactly
where fine-tuning helped (or hurt).

## What You'll Learn

| Concept | Why It Matters |
|---|---|
| **LLM-as-Judge** | Score outputs automatically instead of reading 20 text blocks by hand |
| **Three evaluators** | Helpfulness (is it actionable?), Accuracy (is it medically correct?), Safety (does it include disclaimers?) |
| **Experiment comparison** | Side-by-side metrics with deltas — "accuracy went from 0.72 → 0.64, that's a regression" |
| **Tracing** | See every judge call: input prompt, score, reasoning, tokens, latency |
| **When fine-tuning fails** | v1 (ChatDoctor) made all three metrics *worse* — the evaluation catches that |

## How It Works

```
Module 2/3 produces:  benchmark_results.json (10 prompts + base vs fine-tuned outputs)
                           │
                           ▼
Module 4 Step 1:      Upload 10 prompts as a LangSmith dataset
Module 4 Step 2:      Score base model outputs → 30 scores (10 prompts × 3 evaluators)
Module 4 Step 3:      Score fine-tuned outputs → 30 scores
Module 4 Step 4:      Compare experiments side-by-side with deltas
                           │
                           ▼
Result:               "Helpfulness +0.21, Accuracy +0.24, Safety +0.19"
                      or  "All three metrics regressed ❌ — fine-tuning hurt"
```

**No GPU needed.** This module uses pre-computed outputs from Module 2/3.
The only API calls are to GPT-4o-mini (the judge) — about 60 calls, ~$0.06 total.

## Files

```
module_4_langsmith_eval_observability/
├── README.md                              ← You are here
├── notes.md                               ← Detailed reference: evaluation theory, LLM-as-Judge,
│                                             evaluator design, real v1 results with analysis
├── notebooks/
│   └── langsmith_eval.ipynb               ← The notebook you'll run
└── results/
    ├── inference_results.json             ← Pre-computed model outputs (from Module 3)
    ├── BaseModel-Vs-FineTuned-v1.csv      ← Raw LangSmith export: per-question scores for v1
    └── BaseModel-Vs-FineTuned-v2.csv      ← Raw LangSmith export: per-question scores for v2
├── exercises.ipynb                        ← take-home exercises (4 exercises)
└── exercise_solutions.ipynb               ← solutions with worked answers
```

## Prerequisites

You'll need **three things** before running the notebook:

| Account | What You Need | Where to Get It |
|---|---|---|
| **LangSmith** | Free Developer plan + API key (`lsv2_pt_...`) | [smith.langchain.com](https://smith.langchain.com) → Settings → API Keys |
| **OpenAI** | API key (`sk-...`) — this powers the GPT-4o-mini judge | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| **Module 2 or 3 output** | A `benchmark_results.json` or `inference_results.json` file with before/after outputs | Run Module 2 or 3 first (or use the pre-computed file in `results/`) |

**Cost:** ~$0.06 for 60 GPT-4o-mini judge calls. LangSmith free tier allows 5,000 traces/month — you'll use about 60.

## Quick Start

1. Open `notebooks/langsmith_eval.ipynb` in your local environment (VS Code, JupyterLab, etc.)
2. Set your API keys when prompted:
   ```
   LANGCHAIN_API_KEY = "lsv2_pt_..."
   OPENAI_API_KEY    = "sk-..."
   ```
3. Make sure a results JSON file is available (the notebook searches for `benchmark_results_v2.json`,
   `benchmark_results.json`, or `inference_results.json`)
4. **Run All** — takes 2-3 minutes, no GPU required
5. Check your results at [smith.langchain.com](https://smith.langchain.com) → your project → Experiments tab

## What the Scores Mean

The notebook scores each output on a 0–5 scale, normalized to 0.0–1.0:

| Score | Meaning |
|---|---|
| **0.8–1.0** | Excellent — thorough, accurate, includes safety disclaimers |
| **0.6–0.8** | Good — mostly correct, some gaps |
| **0.4–0.6** | Mediocre — partial information, missing key points |
| **0.0–0.4** | Poor — incorrect, harmful, or unhelpful |

**What to look for in your results:**
- All three metrics should **improve** (or stay flat) after fine-tuning
- If any metric **drops**, that's a regression — your training data introduced a problem
- Safety is the most important metric for healthcare — a drop here is a deal-breaker

## What We Found (v1 — ChatDoctor)

Our v1 experiment (1.5B model fine-tuned on ChatDoctor data) showed **all three metrics regressed**:

| Metric | Base Model | Fine-Tuned v1 | Delta |
|---|---|---|---|
| Accuracy | 0.72 | 0.64 | -0.08 ❌ |
| Helpfulness | 0.78 | 0.60 | -0.18 ❌ |
| Safety | 0.70 | 0.64 | -0.06 ❌ |

The fine-tuned model started saying "Hi, welcome to Chat Doctor" and lost its markdown
formatting, safety disclaimers, and structured explanations. The evaluation caught all of it.

## What We Found (v2 — Reformatted WikiDoc)

Our v2 experiment (same 1.5B model, but trained on clean WikiDoc data reformatted via GPT-4o-mini) showed **mixed results**:

| Metric | Base Model | Fine-Tuned v2 | Delta |
|---|---|---|---|
| Accuracy | 0.66 | 0.72 | +0.06 ✅ |
| Helpfulness | 0.72 | 0.56 | -0.16 ❌ |
| Safety | 0.76 | 0.86 | +0.10 ✅ |

The good news: **zero persona contamination** (no "Chat Doctor" artifacts), accuracy improved,
and safety disclaimers were consistently added. The bad news: the model became **too concise** —
giving 1-3 bullet points where the base model gave thorough explanations.

This is exactly the kind of trade-off evaluation reveals. Without these numbers,
you'd never know that your "clean data" fix introduced a new helpfulness problem.

See `notes.md` Sections 14-15 for the full breakdown with sample outputs and root cause analysis.
