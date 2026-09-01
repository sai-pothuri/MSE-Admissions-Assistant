"""CLI ingestion entrypoint.

Usage: python -m app.scripts.ingest data/knowledge_base
"""

import sys
from pathlib import Path

from app.services.ingestion.indexer import index_directory


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m app.scripts.ingest <knowledge_base_dir>")
        sys.exit(1)

    root = Path(sys.argv[1])
    results = index_directory(root)

    total = 0
    failures = 0
    for path, result in results.items():
        if result.error is not None:
            print(f"{path}: FAILED - {result.error}")
            failures += 1
        else:
            print(f"{path}: {result.chunk_count} chunks")
            total += result.chunk_count

    summary = f"Total: {total} chunks indexed"
    summary += f", {failures} file(s) failed." if failures else "."
    print(summary)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
