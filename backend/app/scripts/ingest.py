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

    total = sum(results.values())
    for path, count in results.items():
        print(f"{path}: {count} chunks")
    print(f"Total: {total} chunks indexed.")


if __name__ == "__main__":
    main()
