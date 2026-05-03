.PHONY: all acquire validate pipeline install freeze clean test

# Python command
PYTHON = poetry run python -m

# Files
ACQUIRE = src.who_will_viral.acquire
VALIDATE = src.who_will_viral.validate
CLEAN = src.who_will_viral.clean
VALIDATE_CLEANED = src.who_will_viral.validation_cleaned
VISUALIZE = src.who_will_viral.visualization.visualization
FEATURE = src/who_will_viral/feature_engineering.py
TRAIN = src.who_will_viral.train

# Test config
TEST_DIR = tests/
COV_SRC = src/who_will_viral/

# Default target
all: help

acquire:
	$(PYTHON) $(ACQUIRE)

validate:
	$(PYTHON) $(VALIDATE)

clean_data:
	$(PYTHON) $(CLEAN)

validate_cleaned:
	$(PYTHON) $(VALIDATE_CLEANED)

visualize:
	$(PYTHON) $(VISUALIZE)

feature:
	poetry run python $(FEATURE)

train:
	$(PYTHON) $(TRAIN)

test:
	poetry run pytest $(TEST_DIR) --cov=$(COV_SRC) --cov-report=xml --cov-report=term-missing

# Full pipeline
pipeline: test acquire validate clean_data validate_cleaned feature train

install:
	poetry install

freeze:
	poetry lock

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

help:
	@echo "make pipeline          - Run full pipeline"
	@echo "make acquire           - Run acquisition only"
	@echo "make validate          - Run validation only"
	@echo "make clean_data        - Run cleaning only"
	@echo "make validate_cleaned  - Run validation after cleaning only"
	@echo "make visualize         - Run visualization only"
	@echo "make feature           - Run feature engineering only"
	@echo "make train             - Run training only"
	@echo "make test              - Run tests with coverage"
	@echo "make install           - Install dependencies"
	@echo "make freeze            - Freeze deps to requirements.txt"
	@echo "make clean             - Remove __pycache__ and .pyc files"