"""
Data Preparation: Reformat WikiDoc for Fine-Tuning (v2)

Takes the raw medalpaca/medical_meadow_wikidoc dataset (terse encyclopedia stubs)
and reformats it into conversational healthcare assistant responses using GPT-4o-mini
with parallel batch calls.

Usage:
    pip install openai datasets huggingface-hub python-dotenv
    cd module_2_colab_finetuning
    python scripts/data_prep_v2.py

Environment variables:
    OPENAI_API_KEY     - Your OpenAI API key (required)
    HF_TOKEN           - Your Hugging Face token with write access (required)
    HF_DATASET_REPO    - Target HF dataset repo (default: YOUR_USERNAME/wikidoc-healthassist)
    NUM_SAMPLES        - Number of examples to process (default: 2100)
    MAX_CONCURRENCY    - Max parallel OpenAI requests (default: 20)

Cost: ~$1-2 for 2,100 examples at GPT-4o-mini pricing.
"""

import os
import json
import asyncio
import random
import argparse
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from datasets import load_dataset, Dataset
from huggingface_hub import login

# Load .env file from the module root (one level up from scripts/)
load_dotenv(Path(__file__).parent.parent / ".env")

# ══════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════

SOURCE_DATASET = "medalpaca/medical_meadow_wikidoc"

SYSTEM_PROMPT = (
    "You are a knowledgeable and thorough healthcare assistant. "
    "When answering medical questions, provide comprehensive explanations "
    "with relevant clinical details, mechanisms of action, and practical guidance. "
    "Structure your answers clearly. "
    "Always recommend consulting a healthcare professional for serious concerns."
)

REFORMAT_PROMPT = """You are reformatting a medical encyclopedia answer into a helpful healthcare assistant response.

Rules:
- Keep ALL medical facts from the original answer. Do not add new medical claims.
- Restructure into clear, readable format with bullet points or numbered lists where appropriate.
- Use plain language alongside medical terms (e.g., "atherosclerosis (hardening of the arteries)").
- Add a brief safety disclaimer at the end like: "Please consult a healthcare professional for personalized medical advice."
- Do NOT add greetings ("Hello"), sign-offs ("Hope this helps"), or any persona.
- Do NOT add medical information that isn't in the original answer.
- Do NOT include any website names, team signatures, attributions, or branding.
- Keep the answer concise and focused. Do not pad with filler to reach a word count.

Original question: {question}

Original encyclopedia answer: {answer}

Rewrite the answer as a helpful healthcare assistant response:"""


async def reformat_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    question: str,
    answer: str,
    index: int,
) -> dict | None:
    """Reformat a single WikiDoc answer using GPT-4o-mini."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": REFORMAT_PROMPT.format(
                            question=question, answer=answer
                        ),
                    }
                ],
                temperature=0.3,
                max_tokens=800,
            )
            result = response.choices[0].message.content.strip()
            if len(result) >= 50:
                return {
                    "question": question,
                    "original_answer": answer,
                    "reformatted_answer": result,
                }
            return None
        except Exception as e:
            print(f"  [{index}] Error: {e}")
            return None


async def process_batch(
    examples: list[dict],
    max_concurrency: int,
    batch_size: int = 50,
) -> list[dict]:
    """Process all examples in parallel batches with a concurrency limit."""
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrency)
    total = len(examples)
    num_batches = (total + batch_size - 1) // batch_size
    all_results = []

    print(f"  Total: {total} examples in {num_batches} batches of {batch_size}")

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch = examples[start:end]

        tasks = [
            reformat_one(client, semaphore, ex["question"], ex["answer"], start + i)
            for i, ex in enumerate(batch)
        ]

        batch_results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                batch_results.append(result)

        all_results.extend(batch_results)
        failed = len(batch) - len(batch_results)
        print(
            f"  Batch {batch_idx + 1}/{num_batches} done "
            f"[{start + 1}-{end}] — "
            f"success: {len(batch_results)}, failed: {failed}, "
            f"running total: {len(all_results)}"
        )

    return all_results


def load_and_filter(num_samples: int) -> list[dict]:
    """Load WikiDoc and filter to valid examples."""
    print(f"Loading {SOURCE_DATASET}...")
    raw_ds_dict = load_dataset(SOURCE_DATASET)
    split = "train" if "train" in raw_ds_dict else list(raw_ds_dict.keys())[0]
    raw_ds = raw_ds_dict[split]
    print(f"  Raw dataset: {len(raw_ds)} rows")

    valid = []
    for ex in raw_ds:
        question = str(ex.get("input") or "").strip()
        answer = str(ex.get("output") or "").strip()
        if len(question) >= 10 and len(answer) >= 50:
            valid.append({"question": question, "answer": answer})

    print(f"  Valid examples: {len(valid)}")

    random.seed(42)
    random.shuffle(valid)
    selected = valid[:num_samples]
    print(f"  Selected: {len(selected)}")
    return selected


def build_chat_dataset(reformatted: list[dict]) -> Dataset:
    """Convert reformatted examples to chat format."""
    rows = []
    for ex in reformatted:
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ex["question"]},
                    {"role": "assistant", "content": ex["reformatted_answer"]},
                ]
            }
        )
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Reformat WikiDoc dataset for healthcare fine-tuning"
    )
    parser.add_argument(
        "--hf-repo",
        default=os.environ.get("HF_DATASET_REPO", "YOUR_USERNAME/wikidoc-healthassist"),
        help="Target HF dataset repo (default: YOUR_USERNAME/wikidoc-healthassist)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=int(os.environ.get("NUM_SAMPLES", "2100")),
        help="Number of examples to process (default: 2100)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=int(os.environ.get("MAX_CONCURRENCY", "20")),
        help="Max parallel OpenAI requests (default: 20)",
    )
    parser.add_argument(
        "--save-local",
        default="reformatted_wikidoc_v2.json",
        help="Also save results locally to this file (default: reformatted_wikidoc_v2.json)",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading to HF Hub (just save locally)",
    )
    args = parser.parse_args()

    # ── Validate API key ──
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Error: Set OPENAI_API_KEY environment variable")

    # ── Load & filter ──
    examples = load_and_filter(args.num_samples)

    # ── Preview raw example ──
    print(f"\nSample raw WikiDoc entry:")
    print(f"  Q: {examples[0]['question']}")
    print(f"  A: {examples[0]['answer'][:200]}...")

    # ── Reformat in parallel ──
    print(f"\nReformatting {len(examples)} examples (concurrency: {args.max_concurrency})...")
    reformatted = asyncio.run(process_batch(examples, args.max_concurrency))
    print(f"\n✅ Reformatting complete: {len(reformatted)} successful")

    # ── Preview result ──
    print(f"\nBEFORE (raw WikiDoc):")
    print(f"  Q: {reformatted[0]['question']}")
    print(f"  A: {reformatted[0]['original_answer'][:200]}...")
    print(f"\nAFTER (reformatted):")
    print(f"  Q: {reformatted[0]['question']}")
    print(f"  A: {reformatted[0]['reformatted_answer'][:300]}...")

    # ── Save locally ──
    if args.save_local:
        with open(args.save_local, "w") as f:
            json.dump(reformatted, f, indent=2)
        print(f"\n✅ Saved {len(reformatted)} examples to {args.save_local}")

    # ── Build chat dataset ──
    dataset = build_chat_dataset(reformatted)
    print(f"\nChat dataset: {len(dataset)} examples")

    # ── Upload to HF ──
    if not args.no_upload:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
        else:
            login()

        dataset.push_to_hub(args.hf_repo, private=True)
        print(f"\n✅ Uploaded to https://huggingface.co/datasets/{args.hf_repo}")
        print(f'   Load in training notebook: load_dataset("{args.hf_repo}", split="train")')
    else:
        print("\n⏭️  Skipped HF upload (--no-upload)")


if __name__ == "__main__":
    main()
