# Documentation

who-will-viral is a machine learning pipeline that predicts YouTube video virality.

## Quick Links

- [Installation](installation.md) — Install who-will-viral from PyPI or source
- [Usage](usage.md) — How to run the pipeline and use the CLI
- [API Reference](api.md) — Auto-generated API documentation

## Project Overview

who-will-viral provides a complete ML pipeline:

1. **Acquire** — Fetch YouTube data via API and web scraping
2. **Validate** — Check raw data quality
3. **Clean** — Standardize and prepare data
4. **Validate Cleaned** — Verify cleaned data quality
5. **Engineer Features** — Extract and select features
6. **Train** — Train ML models with MLflow tracking

## Get Started

```bash
# Install from source
git clone https://github.com/zizoisprogramming/who_will_viral.git
cd who_will_viral
poetry install

# Set up environment
cp .env.example .env
# Edit .env with your YouTube API key

# Run pipeline
just pipeline
```

See [Installation](installation.md) and [Usage](usage.md) for more details.
