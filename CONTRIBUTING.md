# Contributing

Thank you for helping improve Topology Gate. Contributions should preserve the
repository's research-only scope and make assumptions visible.

## Development setup

Python 3.10, 3.11, and 3.12 are supported.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Before opening a pull request, run:

```bash
python -m pytest
ruff check src tests examples
mypy src
git diff --check
```

## Research and data boundaries

- Keep experiments causal: do not use future labels, revisions, or universe
  membership when constructing a decision.
- Label synthetic, final-history, and point-in-time evidence separately.
- Do not commit downloaded market data, private vendor files, credentials,
  tokens, local paths, caches, or generated temporary files.
- Do not add broker connectivity or live order submission to this repository.
- Any new economic claim needs a checked-in receipt, declared assumptions, and
  an explicit limitations section.

## Pull requests

Describe what changed, which checks were run, and whether the change affects a
public contract, a statistical assumption, a data boundary, or a reported
result. Small focused pull requests are easier to review and reproduce.
