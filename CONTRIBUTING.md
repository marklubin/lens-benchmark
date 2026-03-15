# Contributing to LENS

Thanks for your interest in contributing to LENS. This document covers the basics.

## Code of Conduct

Be respectful and constructive. We're building a benchmark — precision and reproducibility matter more than speed.

## Development Setup

```bash
# Clone
git clone https://github.com/synix-dev/lens-benchmark.git
cd lens-benchmark

# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/unit/ -v

# Lint
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

## Types of Contributions

### Memory Adapters

Wrap a memory system for LENS evaluation. See [ADAPTER_GUIDE.md](docs/guides/ADAPTER_GUIDE.md).

**PR checklist**:
- [ ] Subclasses `MemoryAdapter` with all required methods
- [ ] Registered via `@register_adapter` decorator
- [ ] Passes smoke test (`uv run lens smoke`)
- [ ] Runs successfully against S01 (cascading_failure)
- [ ] Unit tests for adapter-specific logic
- [ ] Added to `_ensure_builtins()` in `registry.py` (for built-in adapters)

### Benchmark Scopes

Design a new evaluation scenario. See [SCOPE_GUIDE.md](docs/guides/SCOPE_GUIDE.md).

**PR checklist**:
- [ ] `spec.yaml` follows the scope format
- [ ] Passes all validation gates (contamination <80%, naive baseline <50%, key fact hit >90%)
- [ ] At least 4 question types represented
- [ ] 3 distractor themes with excluded terms
- [ ] Build artifacts generated and verified

### Leaderboard Submissions

Submit benchmark results. See [SUBMISSION_GUIDE.md](docs/guides/SUBMISSION_GUIDE.md).

**PR checklist**:
- [ ] All 6 benchmark scopes (S07-S12) completed
- [ ] Scores generated with LENS scorer
- [ ] Run config, answers, and scores included
- [ ] No budget violations
- [ ] LEADERBOARD.md updated with results

### Bug Fixes and Improvements

- Open an issue first for non-trivial changes
- Include tests for bug fixes
- Keep PRs focused — one fix per PR

### Scoring Improvements

- Justify changes with analysis showing impact on existing results
- Include before/after comparison on at least 2 scopes
- Don't break backwards compatibility without discussion

## PR Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-adapter`)
3. Make your changes
4. Run tests (`uv run pytest tests/unit/ -v`)
5. Run linting (`uv run ruff check src/ tests/`)
6. Open a PR with a clear description

## Code Style

- Python 3.11+, type hints encouraged
- Formatted with `ruff format`, linted with `ruff check`
- Tests mirror source structure in `tests/unit/`
- No silent exception swallowing — log failures, don't hide them

## Questions?

Open a GitHub issue or discussion.
