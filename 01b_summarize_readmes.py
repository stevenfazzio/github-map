"""Generate short LLM summaries, taglines, and target audiences for repo READMEs using Claude Haiku."""

import asyncio
import json
import os
import re
import shutil
import tempfile

import anthropic
import pandas as pd
from tqdm import tqdm

from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_CONCURRENCY,
    ANTHROPIC_MODEL_SUMMARIZE,
    REPOS_PARQUET,
)

PROJECT_TYPES = [
    "Library",
    "Framework",
    "CLI Tool",
    "Application",
    "Dataset",
    "Tutorial/Educational",
    "Collection/Awesome List",
    "Plugin/Extension",
    "API/Service",
    "Research",
    "Other",
]

TARGET_AUDIENCES = [
    "Developers",
    "Data & ML Engineers",
    "DevOps & Infrastructure",
    "System Programmers",
    "Security Professionals",
    "End Users",
    "Learners & Educators",
    "Researchers",
]

SYSTEM_PROMPT = (
    "You are given the README of a GitHub repository. "
    "Return a JSON object with five fields:\n"
    '- "title": The project\'s display name as presented in the README. '
    "If the README does not mention a project name, return null.\n"
    '- "summary": A 1-2 sentence summary of what the project does.\n'
    '- "project_type": One of: '
    + ", ".join(f'"{t}"' for t in PROJECT_TYPES)
    + ".\n"
    '- "tagline": A tagline of at most 10 words that plainly describes what '
    "the project does or is. Write for a technical audience — open-source "
    "developers, researchers, engineers. Be specific and concrete, not "
    "marketing-speak. "
    "Bad: 'Revolutionize your workflow with blazing-fast performance' "
    "Good: 'Fast key-value store with Redis-compatible API'\n"
    '- "target_audience": The primary audience for this project. One of: '
    + ", ".join(f'"{a}"' for a in TARGET_AUDIENCES)
    + ". Choose the single best fit:\n"
    '  - "Developers": General software developers (web, mobile, desktop)\n'
    '  - "Data & ML Engineers": Data scientists, ML/AI practitioners\n'
    '  - "DevOps & Infrastructure": Cloud, containers, CI/CD, monitoring\n'
    '  - "System Programmers": OS, embedded, compilers, low-level systems\n'
    '  - "Security Professionals": Pentesting, crypto, vulnerability research\n'
    '  - "End Users": Non-developers who use the software directly\n'
    '  - "Learners & Educators": Students, tutorial followers, course creators\n'
    '  - "Researchers": Academic or scientific researchers\n\n'
    "Respond with only the JSON object, no markdown fencing."
)
MAX_README_CHARS = 4_000
CHECKPOINT_EVERY = 100

MAX_RETRIES = 5


def safe_write_parquet(df, path):
    """Atomically write a parquet file via tmp + verify + rename."""
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path), suffix=".parquet.tmp"
    )
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False)
        verify = pd.read_parquet(tmp_path)
        assert len(verify) == len(df)
        os.replace(tmp_path, str(path))
    except Exception:
        os.unlink(tmp_path)
        raise


async def summarize_readme(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    text: str,
    full_name: str,
    pbar: tqdm,
) -> tuple[str, str, str, str, str]:
    """Return (project_title, summary, project_type, tagline, target_audience) for a repo README."""
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.messages.create(
                    model=ANTHROPIC_MODEL_SUMMARIZE,
                    max_tokens=300,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": text[:MAX_README_CHARS]}],
                )
                break
            except anthropic.RateLimitError as e:
                wait = min(2**attempt * 5, 60)
                print(f"\n  Rate limit ({e}), retrying in {wait}s...")
                await asyncio.sleep(wait)
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = min(2**attempt * 5, 60)
                print(f"\n  API error ({e}), retrying in {wait}s...")
                await asyncio.sleep(wait)

    pbar.update(1)

    raw = response.content[0].text.strip()

    # Strip markdown fencing if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    repo_name = full_name.split("/")[1]

    try:
        obj = json.loads(raw)
        title = obj.get("title") or repo_name
        summary = obj.get("summary") or raw
        project_type = obj.get("project_type", "Other")
        if project_type not in PROJECT_TYPES:
            project_type = "Other"
        tagline = obj.get("tagline", "")
        # Strip any surrounding quotes from tagline
        tagline = tagline.strip().strip("'\"")
        target_audience = obj.get("target_audience", "Developers")
        if target_audience not in TARGET_AUDIENCES:
            target_audience = "Developers"
    except (json.JSONDecodeError, AttributeError):
        title = repo_name
        summary = raw
        project_type = "Other"
        tagline = ""
        target_audience = "Developers"

    # Strip leading markdown headings from summary as safety net
    summary = re.sub(r"^#+\s+.*?\n+", "", summary).strip()

    return title, summary, project_type, tagline, target_audience


async def main():
    df = pd.read_parquet(REPOS_PARQUET)

    if "summary" not in df.columns:
        df["summary"] = ""
    if "project_title" not in df.columns:
        df["project_title"] = ""
    if "project_type" not in df.columns:
        df["project_type"] = ""
    if "tagline" not in df.columns:
        df["tagline"] = ""
    if "target_audience" not in df.columns:
        df["target_audience"] = ""

    # Identify rows needing processing (any of the 5 fields missing)
    needs_processing = (
        df["project_title"].fillna("").eq("")
        | df["tagline"].fillna("").eq("")
        | df["target_audience"].fillna("").eq("")
    )
    total = needs_processing.sum()
    if total == 0:
        print("All rows already have project titles, taglines, and target audiences.")
        return

    # Back up before modifying
    backup_path = str(REPOS_PARQUET) + ".bak"
    shutil.copy2(REPOS_PARQUET, backup_path)
    print(f"Backed up {REPOS_PARQUET} → {backup_path}")

    print(f"Processing {total} repos...")
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    semaphore = asyncio.Semaphore(ANTHROPIC_CONCURRENCY)

    indices = df.index[needs_processing].tolist()
    pbar = tqdm(total=total)
    count = 0

    for chunk_start in range(0, len(indices), CHECKPOINT_EVERY):
        chunk_indices = indices[chunk_start : chunk_start + CHECKPOINT_EVERY]
        tasks = []
        skip_indices = []

        for idx in chunk_indices:
            readme = df.at[idx, "readme"] if pd.notna(df.at[idx, "readme"]) else ""
            description = (
                df.at[idx, "description"]
                if pd.notna(df.at[idx, "description"])
                else ""
            )
            full_name = df.at[idx, "full_name"]

            text = readme.strip() or description.strip()
            if not text:
                df.at[idx, "project_title"] = full_name.split("/")[1]
                df.at[idx, "summary"] = ""
                df.at[idx, "project_type"] = "Other"
                df.at[idx, "tagline"] = ""
                df.at[idx, "target_audience"] = "Developers"
                skip_indices.append(idx)
                pbar.update(1)
                continue

            tasks.append((idx, full_name, text))

        async def _process(idx, full_name, text):
            try:
                return idx, await summarize_readme(
                    client, semaphore, text, full_name, pbar
                )
            except Exception as e:
                print(f"\n  Error summarizing {full_name}: {e}")
                pbar.update(1)
                return idx, None

        results = await asyncio.gather(
            *[_process(idx, fn, txt) for idx, fn, txt in tasks]
        )

        for idx, result in results:
            if result is not None:
                title, summary, project_type, tagline, target_audience = result
                df.at[idx, "project_title"] = title
                df.at[idx, "summary"] = summary
                df.at[idx, "project_type"] = project_type
                df.at[idx, "tagline"] = tagline
                df.at[idx, "target_audience"] = target_audience

        count += len(chunk_indices)
        safe_write_parquet(df, REPOS_PARQUET)
        print(f"  Checkpoint saved at {count}/{total}")

    pbar.close()
    print(f"Done. Processed {count} repos, saved to {REPOS_PARQUET}")


if __name__ == "__main__":
    asyncio.run(main())
