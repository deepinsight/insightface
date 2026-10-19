#!/usr/bin/env python3
"""Check the provenance of a manually selected wheel build before publication."""

import json
import os
import re
import urllib.request


def validate_source_run(run, repository):
    if run["path"] != ".github/workflows/build_wheels.yaml":
        raise ValueError("Source run must use the Build Wheels workflow")
    if run["status"] != "completed":
        raise ValueError("Source run must have completed; a failed upload is allowed")
    if run["event"] not in ("push", "workflow_dispatch"):
        raise ValueError("Source run must be a push or manual build, not a pull request")
    if (run.get("head_repository") or {}).get("full_name") != repository:
        raise ValueError("Source run must have been built from this repository")


def main():
    run_id = os.environ["SOURCE_RUN_ID"]
    if not re.fullmatch(r"[0-9]+", run_id):
        raise SystemExit("SOURCE_RUN_ID must be a numeric Actions run ID")
    if not os.environ.get("EXPECTED_VERSION", "").strip():
        raise SystemExit("expected_version is required when reusing artifacts")
    repository = os.environ["GITHUB_REPOSITORY"]
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/actions/runs/{run_id}",
        headers={"Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                 "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        run = json.load(response)
    validate_source_run(run, repository)
    print(f"Reusing {run['html_url']} at commit {run['head_sha']}")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write(f"Source run: {run['html_url']}\n\nCommit: `{run['head_sha']}`\n\n")


if __name__ == "__main__":
    main()
