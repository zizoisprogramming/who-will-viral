.PHONY: all pipeline install freeze clean test help

PYTHON = poetry run python -m

ACQUIRE_SRC    = src/who_will_viral/acquire.py
VALIDATE_SRC   = src/who_will_viral/validate.py
CLEAN_SRC      = src/who_will_viral/clean.py
VAL_CLEAN_SRC  = src/who_will_viral/validation_cleaned.py
FEATURE_SRC    = src/who_will_viral/feature_engineering.py
TRAIN_SRC      = src/who_will_viral/train.py

MERGED         = data/youtube/dataset.csv
CLEANED        = data/youtube/cleaned_dataset.csv
EXTRACTED      = data/youtube/extracted.csv
TRAIN          = data/youtube/train.csv

TEST_DIR       = tests/
COV_SRC        = src/who_will_viral/

all: help

acquire:       $(MERGED)
validate:      $(MERGED)
clean_data:    $(CLEANED)
validate_cleaned: $(CLEANED)
feature:       $(EXTRACTED)
train:         $(TRAIN)


$(MERGED): $(ACQUIRE_SRC)
	$(PYTHON) who_will_viral.acquire
	$(PYTHON) who_will_viral.validate

$(CLEANED): $(MERGED) $(CLEAN_SRC) $(VAL_CLEAN_SRC)
	$(PYTHON) who_will_viral.clean
	$(PYTHON) who_will_viral.validation_cleaned

$(EXTRACTED): $(CLEANED) $(FEATURE_SRC)
	poetry run python $(FEATURE_SRC)

$(TRAIN): $(EXTRACTED) $(TRAIN_SRC)
	$(PYTHON) who_will_viral.train

test:
	poetry run pytest $(TEST_DIR) --cov=$(COV_SRC) --cov-report=xml --cov-report=term-missing

pipeline: test $(TRAIN)

install:
	poetry install

freeze:
	poetry lock

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

help:
	@echo "make pipeline          - Run tests then full pipeline (skips unchanged steps)"
	@echo "make acquire           - Run acquisition + validation"
	@echo "make clean_data        - Run cleaning + validation"
	@echo "make feature           - Run feature engineering"
	@echo "make train             - Run training"
	@echo "make test              - Run tests with coverage"
	@echo "make install           - Install dependencies"
	@echo "make freeze            - Lock dependencies"
	@echo "make clean             - Remove __pycache__ and .pyc files"