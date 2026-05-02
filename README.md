# who-will-viral

![PyPI version](https://img.shields.io/pypi/v/who_will_viral.svg)

A machine learning project that predicts whether a YouTube video will trend based on features like description, engagement metrics (likes, comments, views), and other video characteristics.

* [GitHub](https://github.com/zizoisprogramming/who_will_viral/) | [PyPI](https://pypi.org/project/who_will_viral/) | [Documentation](https://zizoisprogramming.github.io/who_will_viral/)
* Created by [team_15](https://audrey.feldroy.com/) | GitHub [@zizoisprogramming](https://github.com/zizoisprogramming) | PyPI [@zizoisprogramming](https://pypi.org/user/zizoisprogramming/)
* MIT License

## Overview

**who-will-viral** is a comprehensive machine learning pipeline that predicts YouTube video virality. The project implements a complete end-to-end workflow including:

- **Data Acquisition**: Fetches YouTube videos via API and web scraping
- **Data Validation**: Validates raw data quality and structure
- **Data Cleaning**: Handles missing values, duplicates, and data standardization
- **Feature Engineering**: Extracts, selects, and scales features
- **Model Training**: Trains multiple ML models with hyperparameter optimization using Optuna
- **Model Evaluation**: Tracks experiments with MLflow

## Features

- 🎯 **Binary Classification**: Predicts trending vs. non-trending YouTube videos
- 📊 **Multiple Data Sources**: Combines YouTube API and web scraping for data collection
- 🔄 **Comprehensive Pipeline**: Complete ETL workflow from acquisition to prediction
- 🤖 **Multiple ML Models**: 
  - Logistic Regression
  - Random Forest
  - XGBoost
  - SVM (Linear SVC)
  - KNN
  - Naive Bayes
  - AdaBoost
- ⚖️ **Class Imbalance Handling**: SMOTE, random over/under-sampling strategies
- 🔍 **Hyperparameter Optimization**: Automated tuning with Optuna and GridSearchCV
- 📈 **Experiment Tracking**: MLflow integration for model versioning and comparison
- 📝 **Comprehensive Validation**: Great Expectations for data quality assurance
- 💾 **Natural Language Processing**: Sentence transformers for text feature extraction

## Project Structure

```
who-will-viral/
├── src/who_will_viral/           # Main package
│   ├── acquire.py                # Data acquisition pipeline
│   ├── clean.py                  # Data cleaning logic
│   ├── validate.py               # Data validation
│   ├── validation_cleaned.py     # Post-cleaning validation
│   ├── feature_engineering.py    # Feature engineering orchestration
│   ├── train.py                  # Model training and evaluation
│   ├── cli.py                    # CLI interface (Typer)
│   ├── mlflow_utilities.py       # MLflow experiment tracking
│   ├── utils.py                  # Utility functions
│   ├── data_acquisition/         # API and web scraping modules
│   ├── feature_engineering/      # Feature extraction, scaling, selection
│   └── visualization/            # Data visualization modules
├── tests/                        # Comprehensive test suite
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
├── notebooks/                    # Jupyter notebooks for exploration
├── data/                         # Data storage
├── docs/                         # Documentation
├── pyproject.toml                # Poetry configuration
└── Makefile                      # Build commands
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

### Requirements

- Python 3.12+
- Poetry (recommended) or pip
- YouTube API key (for data acquisition)

## Quick Start

### 1. Set Up Environment Variables

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

### 2. Run the Complete Pipeline

```bash
# Using Makefile
make pipeline

# Or run individual steps
make acquire           # Data acquisition
make validate          # Validate raw data
make clean_data        # Clean data
make validate_cleaned  # Validate cleaned data
make feature           # Feature engineering
make train             # Model training
```

### 3. Use the CLI

```bash
# Run the CLI
who_will_viral

# See available commands
poetry run who_will_viral --help
```

## Pipeline Stages

### 1. Data Acquisition (`acquire.py`)

Collects YouTube video data through:
- **YouTube API**: Official API for detailed video metadata
- **Web Scraping**: Beautiful Soup for supplementary data collection
- **Database Management**: Stores and manages video records

```bash
make acquire
```

### 2. Data Validation (`validate.py`)

Validates raw data quality:
- Checks data types and structure
- Identifies missing values
- Validates language codes
- Generates quick summary statistics

```bash
make validate
```

### 3. Data Cleaning (`clean.py`)

Cleans and preprocesses data:
- Drops irrelevant columns
- Handles missing values
- Removes duplicates
- Filters by language
- Standardizes data formats
- Removes outliers

```bash
make clean_data
```

### 4. Validation After Cleaning (`validation_cleaned.py`)

Post-cleaning validation to ensure data quality:

```bash
make validate_cleaned
```

### 5. Feature Engineering (`feature_engineering.py`)

Extracts and optimizes features:
- **Feature Extraction**: Generates features from text using Sentence Transformers
- **Feature Selection**: Selects most informative features
- **Feature Scaling**: Normalizes numerical features

```bash
make feature
```

### 6. Model Training (`train.py`)

Trains and evaluates multiple models:
- Data splits: 60% train, 20% validation, 20% test
- Handles class imbalance with SMOTE
- Hyperparameter optimization with Optuna
- MLflow experiment tracking
- Comprehensive performance metrics

```bash
make train
```

## Development

### Setup for Development

```bash
# Clone and install with dev dependencies
git clone git@github.com:your_username/who_will_viral.git
cd who_will_viral
poetry install
```

### Run Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=who_will_viral

# Run specific test file
poetry run pytest tests/unit/test_acquire.py
```

### Quality Assurance

```bash
# Run format, lint, type check, and tests
just qa

# Or run individually
poetry run ruff check .           # Linting
poetry run ruff format .          # Formatting
poetry run pytest                 # Tests
```

### Code Style

- **Formatter**: Ruff
- **Linter**: Ruff
- **Type Checking**: Pylance (via Pyright)
- **Testing**: Pytest

## Documentation

Documentation is built with [Zensical](https://zensical.org/) and deployed to GitHub Pages.

- **Live site:** https://zizoisprogramming.github.io/who_will_viral/
- **Preview locally:** `just docs-serve` (serves at http://localhost:8000)
- **Build:** `just docs-build`

API documentation is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

Docs deploy automatically on push to `main` via GitHub Actions. To enable this, go to your repo's Settings > Pages and set the source to **GitHub Actions**.

## Key Dependencies

- **Data Processing**: pandas, numpy
- **ML Models**: scikit-learn, XGBoost
- **Feature Engineering**: sentence-transformers
- **Class Imbalance**: imbalanced-learn (SMOTE, RandomOver/UnderSampler)
- **Hyperparameter Optimization**: Optuna
- **Experiment Tracking**: MLflow
- **Data Validation**: Great Expectations
- **Web Scraping**: BeautifulSoup4
- **CLI**: Typer, Rich
- **Testing**: pytest, pytest-cov, pytest-mock

## Project Statistics

- **Models Supported**: 7+ classification algorithms
- **Data Sources**: YouTube API + Web Scraping
- **Validation Framework**: Great Expectations
- **Test Coverage**: Comprehensive unit and integration tests
- **Documentation**: Auto-generated API docs + usage guides

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on:

- Reporting bugs
- Fixing bugs
- Implementing features
- Writing documentation

## Code of Conduct

Please review [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Security

For security issues, please see [SECURITY.md](SECURITY.md) for reporting procedures.

## Authors & Attribution

- **Created by**: team_15
- **Maintainer**: [@zizoisprogramming](https://github.com/zizoisprogramming)
- **PyPI**: [@zizoisprogramming](https://pypi.org/user/zizoisprogramming/)

## Changelog

See [CHANGELOG](CHANGELOG/) for version history and release notes.

## Additional Resources

- 📚 [Installation Guide](docs/installation.md)
- 📖 [Usage Guide](docs/usage.md)
- 🔌 [API Reference](docs/api.md)
- 💬 [Discussions & Issues](https://github.com/zizoisprogramming/who_will_viral/issues)
- 📝 [Contributing Guidelines](CONTRIBUTING.md)

## Author

who-will-viral was created in 2026 by team_15.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
