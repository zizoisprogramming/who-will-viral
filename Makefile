.PHONY: all acquire validate pipeline venv install freeze clean

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


# Default target
all: help

# Run data acquisition
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


# Full pipeline
# Full pipeline
pipeline: acquire validate clean validate_cleaned feature train

install:
	poetry install

freeze:
	poetry lock

# Clean cache
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

help:
	@echo "make pipeline  - Run full pipeline"
	@echo "make acquire   - Run acquisition only"
	@echo "make validate  - Run validation only"
	@echo "make clean_data     - Run cleaning only"
	@echo "make validate_cleaned  - Run validation after cleaning only"
	@echo "make visualize - Run visualization only"
	@echo "make feature   - Run feature engineering only"
	@echo "make train     - Run training only"
	@echo "make venv      - Create virtual environment"
	@echo "make install   - Install dependencies"
	@echo "make freeze    - Freeze deps to requirements.txt"
	@echo "make clean     - Remove __pycache__ and .pyc files"