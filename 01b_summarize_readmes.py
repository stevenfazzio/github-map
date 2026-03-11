"""Generate short LLM summaries of repo READMEs using Claude Haiku."""

import time

import anthropic
import pandas as pd
from tqdm import tqdm

from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL_SUMMARIZE, REPOS_PARQUET

SYSTEM_PROMPT = (
    "Summarize this GitHub repository in 1-2 sentences based on its README. "
    "Be concise and focus on what the project does."
)
MAX_README_CHARS = 4_000
CHECKPOINT_EVERY = 100
SLEEP_BETWEEN_CALLS = 0.1


def summarize_readme(client: anthropic.Anthropic, text: str) -> str:
    response = client.messages.create(
        model=ANTHROPIC_MODEL_SUMMARIZE,
        max_tokens=150,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text[:MAX_README_CHARS]}],
    )
    return response.content[0].text.strip()


def main():
    df = pd.read_parquet(REPOS_PARQUET)

    if "summary" not in df.columns:
        df["summary"] = ""

    # Identify rows needing summaries
    needs_summary = df["summary"].fillna("").eq("")
    total = needs_summary.sum()
    if total == 0:
        print("All rows already have summaries.")
        return

    print(f"Summarizing {total} repos...")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    count = 0

    for idx in tqdm(df.index[needs_summary], total=total):
        readme = df.at[idx, "readme"] if pd.notna(df.at[idx, "readme"]) else ""
        description = df.at[idx, "description"] if pd.notna(df.at[idx, "description"]) else ""

        text = readme.strip() or description.strip()
        if not text:
            df.at[idx, "summary"] = ""
            count += 1
            continue

        df.at[idx, "summary"] = summarize_readme(client, text)
        count += 1

        if count % CHECKPOINT_EVERY == 0:
            df.to_parquet(REPOS_PARQUET, index=False)
            print(f"  Checkpoint saved at {count}/{total}")

        time.sleep(SLEEP_BETWEEN_CALLS)

    df.to_parquet(REPOS_PARQUET, index=False)
    print(f"Done. Saved {count} summaries to {REPOS_PARQUET}")


if __name__ == "__main__":
    main()
