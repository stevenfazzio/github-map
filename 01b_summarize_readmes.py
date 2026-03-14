"""Generate short LLM summaries of repo READMEs using Claude Haiku."""

import json
import re
import shutil
import time

import anthropic
import pandas as pd
from tqdm import tqdm

from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL_SUMMARIZE, REPOS_PARQUET

SYSTEM_PROMPT = (
    "You are given the README of a GitHub repository. "
    "Return a JSON object with two fields:\n"
    '- "title": The project\'s display name as presented in the README. '
    "If the README does not mention a project name, return null.\n"
    '- "summary": A 1-2 sentence summary of what the project does.\n\n'
    "Respond with only the JSON object, no markdown fencing."
)
MAX_README_CHARS = 4_000
CHECKPOINT_EVERY = 100
SLEEP_BETWEEN_CALLS = 0.1


def summarize_readme(client: anthropic.Anthropic, text: str, full_name: str) -> tuple[str, str]:
    """Return (project_title, summary) for a repo README."""
    response = client.messages.create(
        model=ANTHROPIC_MODEL_SUMMARIZE,
        max_tokens=250,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text[:MAX_README_CHARS]}],
    )
    raw = response.content[0].text.strip()

    # Strip markdown fencing if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    repo_name = full_name.split("/")[1]

    try:
        obj = json.loads(raw)
        title = obj.get("title") or repo_name
        summary = obj.get("summary") or raw
    except (json.JSONDecodeError, AttributeError):
        title = repo_name
        summary = raw

    # Strip leading markdown headings from summary as safety net
    summary = re.sub(r"^#+\s+.*?\n+", "", summary).strip()

    return title, summary


def main():
    df = pd.read_parquet(REPOS_PARQUET)

    if "summary" not in df.columns:
        df["summary"] = ""
    if "project_title" not in df.columns:
        df["project_title"] = ""

    # Identify rows needing processing (project_title missing)
    needs_summary = df["project_title"].fillna("").eq("")
    total = needs_summary.sum()
    if total == 0:
        print("All rows already have project titles.")
        return

    # Back up before modifying
    backup_path = str(REPOS_PARQUET) + ".bak"
    shutil.copy2(REPOS_PARQUET, backup_path)
    print(f"Backed up {REPOS_PARQUET} → {backup_path}")

    print(f"Summarizing {total} repos...")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    count = 0

    for idx in tqdm(df.index[needs_summary], total=total):
        readme = df.at[idx, "readme"] if pd.notna(df.at[idx, "readme"]) else ""
        description = df.at[idx, "description"] if pd.notna(df.at[idx, "description"]) else ""
        full_name = df.at[idx, "full_name"]

        text = readme.strip() or description.strip()
        if not text:
            df.at[idx, "project_title"] = full_name.split("/")[1]
            df.at[idx, "summary"] = ""
            count += 1
            continue

        title, summary = summarize_readme(client, text, full_name)
        df.at[idx, "project_title"] = title
        df.at[idx, "summary"] = summary
        count += 1

        if count % CHECKPOINT_EVERY == 0:
            df.to_parquet(REPOS_PARQUET, index=False)
            print(f"  Checkpoint saved at {count}/{total}")

        time.sleep(SLEEP_BETWEEN_CALLS)

    df.to_parquet(REPOS_PARQUET, index=False)
    print(f"Done. Saved {count} summaries to {REPOS_PARQUET}")


if __name__ == "__main__":
    main()
