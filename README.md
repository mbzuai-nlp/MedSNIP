# MedSNIP

Context-aware fact verification for medical open-domain QA.

## Layout

```
src/<module>/          # reusable Python modules, named by concern
data/<N>-<step>/       # numbered pipeline outputs
scripts/               # thin runners that call into src/
web/                   # annotation dashboard + static assets
```

## Setup

```bash
uv sync
cp .env.template .env  # fill in API keys
```

## Pipeline

Steps are ported in order. See `data/<N>-<step>/` for each step's output.
