# who-will-viral

A machine learning project that predicts whether a YouTube video will trend based on features like description, engagement metrics (likes, comments, views), and other video characteristics.

- GitHub: [zizoisprogramming/who_will_viral](https://github.com/zizoisprogramming/who_will_viral/)
- MIT License

## Overview

**who-will-viral** is a complete machine learning pipeline that predicts YouTube video virality. It implements an end-to-end workflow:

- **Data Acquisition**: YouTube API and web scraping
- **Data Validation**: Quality checks and data profiling
- **Data Cleaning**: Standardization, deduplication, outlier removal
- **Feature Engineering**: Text features, scaling, selection
- **Model Training**: Multiple classifiers with hyperparameter tuning
- **Experiment Tracking**: MLflow for comparison and versioning

## Project Structure

```
who-will-viral/
├── src/who_will_viral/        # Main package
│   ├── acquire.py             # Data acquisition
│   ├── clean.py               # Data cleaning
│   ├── validate.py            # Data validation
│   ├── validation_cleaned.py  # Post-cleaning validation
│   ├── feature_engineering.py # Feature engineering orchestration
│   ├── train.py               # Model training and evaluation
│   ├── cli.py                 # CLI interface
│   ├── model_loader.py        # Model loading utilities
│   ├── data_acquisition/      # API and scraping modules
│   ├── feature_engineering/   # Feature modules
│   ├── models/                # Trained model artifacts
│   └── visualization/         # Visualization utilities
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── notebooks/                 # Jupyter notebooks for analysis
├── data/                      # Raw and processed data
├── reports/                   # Results, figures, validation reports
├── docs/                      # Documentation
├── scripts/                   # Utility scripts (release.py)
├── pyproject.toml             # Dependencies and project config
├── justfile                   # Task automation (recommended)
├── Makefile                   # Task automation (legacy)
├── zensical.toml              # Docs build config
└── poetry.lock                # Locked dependencies
```

## Installation

### From PyPI

```bash
pip install who_will_viral
```

### From Source

```bash
git clone https://github.com/zizoisprogramming/who_will_viral.git
cd who_will_viral
poetry install
```

**Requirements**: Python 3.12+, Poetry, YouTube API key

## Quick Start

### 1. Environment Setup

Create `.env` file:

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

### 2. Run Pipeline

```bash
# Full pipeline
just pipeline

# Individual stages
just acquire
just validate
just clean-data
just validate-cleaned
just feature
just train
```

### 3. CLI

```bash
poetry run who_will_viral --help
```

## Pipeline Stages

1. **acquire.py** — Fetch YouTube data via API and scraping
2. **validate.py** — Validate raw data quality
3. **clean.py** — Clean and standardize data
4. **validation_cleaned.py** — Validate cleaned data
5. **feature_engineering.py** — Engineer and select features
6. **train.py** — Train and evaluate models with MLflow

## Development

### Tests

```bash
poetry run pytest                # Run all tests
poetry run pytest --cov          # With coverage
just test                        # Using justfile
just coverage                    # Full coverage report
```

### Code Quality

```bash
just qa                          # Format, lint, type check, test
poetry run ruff format .         # Format
poetry run ruff check .          # Lint
just type-check                  # Type checking
```

### Documentation

```bash
just docs-serve                  # Preview at localhost:8000
just docs-build                  # Build static docs
```

### Task Runner

Use `justfile` (recommended):
```bash
just --list                      # View all tasks
```

Or legacy `Makefile`:
```bash
make help
```

## Documentation

Built with [Zensical](https://zensical.org/), deployed to [GitHub Pages](https://zizoisprogramming.github.io/who_will_viral/).

- **Preview**: `just docs-serve`
- **Build**: `just docs-build`
- **Deploy**: Automatic on push to `main` (GitHub Actions)

## Dependencies

- **ML**: scikit-learn, XGBoost, Optuna
- **Data**: pandas, numpy
- **NLP**: sentence-transformers
- **Class Balance**: imbalanced-learn (SMOTE)
- **Tracking**: MLflow
- **Validation**: great-expectations
- **Web**: requests, beautifulsoup4
- **CLI**: typer, rich
- **Testing**: pytest, pytest-cov, pytest-mock
- **Quality**: ruff, pylance

See `pyproject.toml` for full list and versions.

## License & Contributing

- **License**: MIT (see [LICENSE](LICENSE))
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Code of Conduct**: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- **Security**: [SECURITY.md](SECURITY.md)
- **Changelog**: [CHANGELOG/](CHANGELOG/)
