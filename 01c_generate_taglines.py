"""Generate punchy ≤10-word taglines from existing repo summaries using Claude Haiku."""

import shutil
import time

import anthropic
import pandas as pd
from tqdm import tqdm

from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL_TAGLINE, REPOS_PARQUET

SYSTEM_PROMPT = (
    "Given a project title and summary, write a tagline of at most 10 words "
    "that captures what this project is in a punchy, memorable phrase. "
    "Return only the tagline text, no quotes or punctuation framing."
)

MAX_RETRIES = 5
CHECKPOINT_EVERY = 100
SLEEP_BETWEEN_CALLS = 0.1


def generate_tagline(
    client: anthropic.Anthropic, title: str, summary: str
) -> str:
    """Return a short tagline for a repo given its title and summary."""
    user_text = f"Project: {title}\nSummary: {summary}"
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=ANTHROPIC_MODEL_TAGLINE,
                max_tokens=60,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_text}],
            )
            break
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = min(2**attempt * 5, 60)
            print(f"\n  API error ({e}), retrying in {wait}s...")
            time.sleep(wait)
    return response.content[0].text.strip()


def main():
    df = pd.read_parquet(REPOS_PARQUET)

    if "tagline" not in df.columns:
        df["tagline"] = ""

    # Identify rows needing a tagline: have a summary but no tagline yet
    has_summary = df["summary"].fillna("").ne("")
    needs_tagline = has_summary & df["tagline"].fillna("").eq("")
    total = needs_tagline.sum()

    if total == 0:
        print("All rows with summaries already have taglines.")
        return

    # Back up before modifying
    backup_path = str(REPOS_PARQUET) + ".bak"
    shutil.copy2(REPOS_PARQUET, backup_path)
    print(f"Backed up {REPOS_PARQUET} → {backup_path}")

    print(f"Generating taglines for {total} repos...")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    count = 0

    for idx in tqdm(df.index[needs_tagline], total=total):
        summary = df.at[idx, "summary"]
        title = (
            df.at[idx, "project_title"]
            if "project_title" in df.columns and pd.notna(df.at[idx, "project_title"])
            else df.at[idx, "full_name"].split("/")[1]
        )

        # Fall back to description if summary is empty (shouldn't happen given filter)
        text = summary.strip()
        if not text:
            description = df.at[idx, "description"] if pd.notna(df.at[idx, "description"]) else ""
            text = description.strip()
        if not text:
            count += 1
            continue

        tagline = generate_tagline(client, title, text)
        df.at[idx, "tagline"] = tagline
        count += 1

        if count % CHECKPOINT_EVERY == 0:
            df.to_parquet(REPOS_PARQUET, index=False)
            print(f"  Checkpoint saved at {count}/{total}")

        time.sleep(SLEEP_BETWEEN_CALLS)

    df.to_parquet(REPOS_PARQUET, index=False)
    print(f"Done. Generated {count} taglines, saved to {REPOS_PARQUET}")


if __name__ == "__main__":
    main()
