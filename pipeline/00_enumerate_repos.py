"""Enumerate candidate repos via BigQuery (GH Archive WatchEvents).

Produces data/candidates.csv with ~25K repo full_names that have significant
recent star activity. This is a generous superset — 01_fetch_repos.py will
look up each candidate via GraphQL and keep the top TARGET_REPO_COUNT by stars.

Requires:
  pip install google-cloud-bigquery
  gcloud auth application-default login
  GCP_PROJECT set in .env (or gcloud default project configured)

Fallback: if you don't have GCP access, copy the committed candidates.csv
to data/candidates.csv and skip this script.
"""

from config import CANDIDATES_COMMITTED, CANDIDATES_CSV, GCP_PROJECT
from google.cloud import bigquery

QUERY = """\
SELECT repo.name AS full_name, COUNT(*) AS star_events
FROM `githubarchive.month.*`
WHERE type = 'WatchEvent'
  AND _TABLE_SUFFIX BETWEEN '202201' AND '202604'
GROUP BY full_name
HAVING star_events >= 200
ORDER BY star_events DESC
LIMIT 25000
"""


def main():
    if CANDIDATES_CSV.exists():
        print(f"{CANDIDATES_CSV} already exists, skipping BigQuery enumeration.")
        print("Delete it and re-run to refresh candidates.")
        return

    project = GCP_PROJECT or None  # None lets the client use gcloud default
    client = bigquery.Client(project=project)

    print("Running BigQuery query against githubarchive.month.* ...")
    print("(This scans ~200 GB and may take 30-60 seconds)")
    job = client.query(QUERY)
    rows = list(job.result())
    print(f"Got {len(rows)} candidate repos")

    # Write CSV
    CANDIDATES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES_CSV, "w") as f:
        f.write("full_name,star_events\n")
        for row in rows:
            # Escape commas in repo names (shouldn't happen, but be safe)
            name = row["full_name"].replace('"', '""')
            if "," in name:
                name = f'"{name}"'
            f.write(f"{name},{row['star_events']}\n")

    print(f"Saved {len(rows)} candidates to {CANDIDATES_CSV}")

    # Also update committed fallback
    import shutil

    shutil.copy2(CANDIDATES_CSV, CANDIDATES_COMMITTED)
    print(f"Updated committed fallback at {CANDIDATES_COMMITTED}")


if __name__ == "__main__":
    main()
