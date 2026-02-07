# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Local checks

```bash
ruff check .
pytest
```

## Pull requests

- Keep changes focused and small.
- Add/adjust tests for behavior changes.
- Do not commit credentials, private PDFs, or generated files.
