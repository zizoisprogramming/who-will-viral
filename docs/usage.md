# Usage

## Quick Start

### 1. Configure Environment

Create a `.env` file in the project root:

```env
YOUTUBE_API_KEY=your_api_key_here
BASE_CSV=./data/youtube/youtube.csv
OUTPUT_CSV=./data/youtube/youtube_enriched.csv
BACKUP_DIR=./data/youtube/backup
CLEANED_PATH=./data/youtube/cleaned_dataset.csv
TRAIN_PATH=./data/train.csv
VAL_PATH=./data/val.csv
TEST_PATH=./data/test.csv
```

### 2. Run the Pipeline

Execute the full pipeline from data acquisition to model training:

```bash
just pipeline
```

Or run individual stages:

```bash
just acquire           # Fetch YouTube data
just validate          # Validate raw data
just clean-data        # Clean data
just validate-cleaned  # Validate cleaned data
just feature           # Engineer features
just train             # Train models
```

### 3. Use the CLI

Access the command-line interface:

```bash
poetry run who_will_viral --help
poetry run who_will_viral
```

## Development

### Run Tests

```bash
poetry run pytest              # All tests
poetry run pytest --cov        # With coverage
just test                      # Using justfile
just coverage                  # Full coverage report
```

### Code Quality

```bash
just qa                        # Format, lint, type check, test
poetry run ruff format .       # Format code
poetry run ruff check .        # Lint code
just type-check                # Type checking
```

### View Available Tasks

```bash
just --list                    # All justfile tasks
just list                      # Alternative
```

## Documentation

### Preview Docs Locally

```bash
just docs-serve               # Serves at http://localhost:8000
```

### Build Docs

```bash
just docs-build               # Build static documentation
```

Docs are built with [Zensical](https://zensical.org/) and deployed automatically to [GitHub Pages](https://zizoisprogramming.github.io/who_will_viral/) on push to `main`.

## Using as a Library

Import and use who-will-viral modules in your Python code:

```python
from who_will_viral.acquire import fetch_youtube_data
from who_will_viral.train import train_model

# Fetch YouTube data
videos = fetch_youtube_data(api_key="YOUR_KEY")

# Train a model
model, metrics = train_model(videos)
```

See the [API Reference](api.md) for detailed module documentation.

## Common Commands

| Task | Command |
|------|---------|
| Run full pipeline | `just pipeline` |
| Run tests | `just test` or `poetry run pytest` |
| Quality assurance | `just qa` |
| Type checking | `just type-check` |
| Preview docs | `just docs-serve` |
| View all tasks | `just --list` |

## Troubleshooting

### API Key Issues

Ensure your `.env` file has the correct `YOUTUBE_API_KEY`. Get a key from [Google Cloud Console](https://console.cloud.google.com/).

### Import Errors

Make sure Poetry dependencies are installed:

```bash
poetry install
```

### Data Path Issues

Verify all paths in `.env` exist and are writable:

```bash
ls -la ./data/youtube/
```
