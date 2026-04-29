import sys
from pathlib import Path

import pandas as pd
import pytest

from src.who_will_viral.train import ModelTrainer

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))



@pytest.fixture
def sample_data(tmp_path):
    """Create small fake CSV datasets for training."""

    df = pd.DataFrame({
        "feature1": list(range(50)),
        "feature2": list(range(50, 100)),
        "is_trending": [0, 1] * 25
    })

    train_path = tmp_path / "train.csv"
    val_path   = tmp_path / "val.csv"
    test_path  = tmp_path / "test.csv"

    df.to_csv(train_path, index=False)
    df.to_csv(val_path, index=False)
    df.to_csv(test_path, index=False)

    return train_path, val_path, test_path


def test_trainer_initialization(sample_data):
    train_path, val_path, test_path = sample_data

    trainer = ModelTrainer(train_path, val_path, test_path, cv=2)

    assert trainer.X_train is not None
    assert trainer.y_train is not None
    assert trainer.best_model is None
    assert trainer.best_f1 == 0


def test_sampling_methods(sample_data):
    train_path, val_path, test_path = sample_data
    trainer = ModelTrainer(train_path, val_path, test_path, cv=2)

    X_over, y_over = trainer.over_sample()
    X_under, y_under = trainer.under_sample()

    assert len(X_over) >= len(trainer.X_train)
    assert len(X_under) <= len(trainer.X_train)


def test_train_knn_runs(sample_data):
    train_path, val_path, test_path = sample_data
    trainer = ModelTrainer(train_path, val_path, test_path, cv=2)

    result = trainer.train_knn()

    assert result is not None


def test_best_model_updates(sample_data):
    train_path, val_path, test_path = sample_data
    trainer = ModelTrainer(train_path, val_path, test_path, cv=2)

    trainer.train_gaussian_nb()

    assert trainer.best_model is not None
    assert trainer.best_model_name is not None
    assert trainer.best_f1 >= 0


def test_get_test_report_without_training(sample_data):
    train_path, val_path, test_path = sample_data
    trainer = ModelTrainer(train_path, val_path, test_path, cv=2)

    report = trainer.get_test_report()

    assert report is None
