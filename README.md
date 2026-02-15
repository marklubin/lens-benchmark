# LENS Benchmark

**Longitudinal Evidence-backed Narrative Signals** — a benchmark for evaluating memory systems for AI agents.

## Quick Start

```bash
pip install lens-bench
lens smoke  # Run sanity check with null adapter
```

## Usage

```bash
lens run --dataset data.json --adapter null --out output/
lens score --run output/
lens report --run output/
lens compare output1/ output2/
```
