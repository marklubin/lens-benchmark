#!/bin/bash
# Run all 3 SQLite adapter variants against the benchmark dataset.
set -euo pipefail

for variant in sqlite-fts sqlite-embedding sqlite-hybrid; do
  echo "=== Running $variant ==="
  uv run lens run --config "configs/${variant}_benchmark.json" -vv
done

echo "=== All variants complete ==="
