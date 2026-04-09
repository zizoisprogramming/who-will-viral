
.PHONY: all acquire validate pipeline venv install freeze clean

# Python command
PYTHON = python -m

# Files
ACQUIRE = pipelines.acquisition.py
VALIDATE = pipelines.validation.py

# Default target
all: help

# Run data acquisition
acquire:
	$(PYTHON) $(ACQUIRE)

# Run validation
validate:
	$(PYTHON) $(VALIDATE)

# Full pipeline
pipeline: acquire validate

# Create virtual environment
venv:
	$(PYTHON) -m venv venv

# Install basic packages (temporary until req.txt)
install:
	pip install pandas numpy great_expectations

# Freeze dependencies into requirements.txt
freeze:
	pip freeze > requirements.txt

# Clean cache
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

help:
	@echo "make pipeline  - Run acquisition then validation"
	@echo "make acquire   - Run acquisition only"
	@echo "make validate  - Run validation only"
	@echo "make venv      - Create virtual environment"
	@echo "make install   - Install dependencies"
	@echo "make freeze    - Freeze deps to requirements.txt"
	@echo "make clean     - Remove __pycache__ and .pyc files"