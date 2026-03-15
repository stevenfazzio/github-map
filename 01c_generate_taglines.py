"""Generate concise ≤10-word taglines from existing repo summaries using Claude Haiku."""

import asyncio
import os
import shutil
import tempfile

import anthropic
import pandas as pd
from tqdm import tqdm

from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_CONCURRENCY,
    ANTHROPIC_MODEL_TAGLINE,
    REPOS_PARQUET,
)

SYSTEM_PROMPT = (
    "Given a project title and summary, write a tagline of at most 10 words "
    "that plainly describes what the project does or is. "
    "Write for a technical audience — open-source developers, researchers, engineers. "
    "Be specific and concrete, not marketing-speak. "
    "Bad: 'Revolutionize your workflow with blazing-fast performance' "
    "Good: 'Fast key-value store with Redis-compatible API' "
    "Return only the tagline text, no quotes or punctuation framing."
)

MAX_RETRIES = 5
CHECKPOINT_EVERY = 100


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


async def generate_tagline(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    title: str,
    summary: str,
    pbar: tqdm,
) -> str:
    """Return a short tagline for a repo given its title and summary."""
    user_text = f"Project: {title}\nSummary: {summary}"
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.messages.create(
                    model=ANTHROPIC_MODEL_TAGLINE,
                    max_tokens=60,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_text}],
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
    return response.content[0].text.strip()


async def main():
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
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    semaphore = asyncio.Semaphore(ANTHROPIC_CONCURRENCY)

    indices = df.index[needs_tagline].tolist()
    pbar = tqdm(total=total)
    count = 0

    for chunk_start in range(0, len(indices), CHECKPOINT_EVERY):
        chunk_indices = indices[chunk_start : chunk_start + CHECKPOINT_EVERY]
        tasks = []
        skip_indices = []

        for idx in chunk_indices:
            summary = df.at[idx, "summary"]
            title = (
                df.at[idx, "project_title"]
                if "project_title" in df.columns
                and pd.notna(df.at[idx, "project_title"])
                else df.at[idx, "full_name"].split("/")[1]
            )

            # Fall back to description if summary is empty
            text = summary.strip()
            if not text:
                description = (
                    df.at[idx, "description"]
                    if pd.notna(df.at[idx, "description"])
                    else ""
                )
                text = description.strip()
            if not text:
                skip_indices.append(idx)
                pbar.update(1)
                continue

            tasks.append((idx, title, text))

        async def _process(idx, title, text):
            try:
                return idx, await generate_tagline(
                    client, semaphore, title, text, pbar
                )
            except Exception as e:
                print(f"\n  Error generating tagline for row {idx}: {e}")
                pbar.update(1)
                return idx, None

        results = await asyncio.gather(
            *[_process(idx, t, txt) for idx, t, txt in tasks]
        )

        for idx, result in results:
            if result is not None:
                df.at[idx, "tagline"] = result

        count += len(chunk_indices)
        safe_write_parquet(df, REPOS_PARQUET)
        print(f"  Checkpoint saved at {count}/{total}")

    pbar.close()
    print(f"Done. Generated {count} taglines, saved to {REPOS_PARQUET}")


if __name__ == "__main__":
    asyncio.run(main())
