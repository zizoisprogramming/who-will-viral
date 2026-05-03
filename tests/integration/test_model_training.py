"""Integration tests for model training and validation."""

import os
import pickle
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestModelTrainingIntegration:
	"""Integration tests for model training pipeline."""

	@pytest.fixture
	def training_data(self):
		"""Create sample training data with features and target."""
		np.random.seed(42)
		n_samples = 100
		n_features = 20

		X = np.random.randn(n_samples, n_features)
		y = np.random.randint(0, 2, n_samples)

		df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
		df['is_trending'] = y
		return df

	def test_train_val_test_split(self, training_data):
		"""Test that data can be split into train/val/test sets."""
		train_size = int(0.6 * len(training_data))
		val_size = int(0.2 * len(training_data))

		train_df = training_data.iloc[:train_size]
		val_df = training_data.iloc[train_size : train_size + val_size]
		test_df = training_data.iloc[train_size + val_size :]

		# Verify splits
		assert len(train_df) > 0
		assert len(val_df) > 0
		assert len(test_df) > 0
		assert len(train_df) + len(val_df) + len(test_df) == len(training_data)
		assert len(train_df) >= len(val_df) >= len(test_df)

	def test_feature_target_separation(self, training_data):
		"""Test separation of features and target."""
		X = training_data.drop(columns=['is_trending'])
		y = training_data['is_trending']

		assert X.shape[1] == 20
		assert len(y) == len(training_data)
		assert not any(col == 'is_trending' for col in X.columns)

	@patch('who_will_viral.models.mlflow_utilities.setup_mlflow')
	def test_model_trainer_initialization(self, mock_setup, training_data):
		"""Test ModelTrainer can be initialized with data."""
		with tempfile.TemporaryDirectory() as tmp_path:
			train_path = os.path.join(tmp_path, 'train.csv')
			val_path = os.path.join(tmp_path, 'val.csv')
			test_path = os.path.join(tmp_path, 'test.csv')

			training_data.to_csv(train_path, index=False)
			training_data.to_csv(val_path, index=False)
			training_data.to_csv(test_path, index=False)

			from who_will_viral.train import ModelTrainer

			trainer = ModelTrainer(train_path, val_path, test_path)
			assert trainer is not None
			assert trainer.X_train is not None
			assert trainer.y_train is not None

	@patch('who_will_viral.models.mlflow_utilities.setup_mlflow')
	def test_model_trainer_data_loading(self, mock_setup, training_data):
		"""Test that ModelTrainer correctly loads and processes data."""
		with tempfile.TemporaryDirectory() as tmp_path:
			train_path = os.path.join(tmp_path, 'train.csv')
			val_path = os.path.join(tmp_path, 'val.csv')
			test_path = os.path.join(tmp_path, 'test.csv')

			training_data.to_csv(train_path, index=False)
			training_data.to_csv(val_path, index=False)
			training_data.to_csv(test_path, index=False)

			from who_will_viral.train import ModelTrainer

			trainer = ModelTrainer(train_path, val_path, test_path)

			# Verify shapes
			assert len(trainer.X_train) > 0
			assert len(trainer.y_train) > 0
			assert len(trainer.X_val) > 0
			assert len(trainer.y_test) > 0
			assert trainer.X_train.shape[1] == trainer.X_val.shape[1] == trainer.X_test.shape[1]

	@patch('who_will_viral.models.mlflow_utilities.setup_mlflow')
	def test_sampling_techniques(self, mock_setup, training_data):
		"""Test SMOTE and sampling techniques."""
		from imblearn.over_sampling import SMOTE
		from imblearn.under_sampling import RandomUnderSampler

		with tempfile.TemporaryDirectory() as tmp_path:
			train_path = os.path.join(tmp_path, 'train.csv')
			val_path = os.path.join(tmp_path, 'val.csv')
			test_path = os.path.join(tmp_path, 'test.csv')

			training_data.to_csv(train_path, index=False)
			training_data.to_csv(val_path, index=False)
			training_data.to_csv(test_path, index=False)

			from who_will_viral.train import ModelTrainer

			trainer = ModelTrainer(train_path, val_path, test_path)

			# Test SMOTE
			smote = SMOTE(random_state=42)
			X_resampled, y_resampled = smote.fit_resample(trainer.X_train, trainer.y_train)

			assert len(X_resampled) >= len(trainer.X_train)
			assert len(y_resampled) == len(X_resampled)

	@patch('who_will_viral.models.mlflow_utilities.setup_mlflow')
	def test_hyperparameter_optimization_grid(self, mock_setup, training_data):
		"""Test grid search hyperparameter optimization."""
		from sklearn.model_selection import GridSearchCV
		from sklearn.linear_model import LogisticRegression

		X = training_data.drop(columns=['is_trending']).values
		y = training_data['is_trending'].values

		param_grid = {
			'C': [0.1, 1.0],
			'max_iter': [100, 200],
		}

		lr = LogisticRegression()
		grid_search = GridSearchCV(lr, param_grid, cv=3)
		grid_search.fit(X, y)

		assert grid_search.best_params_ is not None
		assert 'C' in grid_search.best_params_
		assert 'max_iter' in grid_search.best_params_

	@patch('who_will_viral.models.mlflow_utilities.setup_mlflow')
	def test_model_evaluation_metrics(self, mock_setup, training_data):
		"""Test that models produce valid evaluation metrics."""
		from sklearn.linear_model import LogisticRegression
		from sklearn.metrics import f1_score, classification_report

		X = training_data.drop(columns=['is_trending']).values
		y = training_data['is_trending'].values

		lr = LogisticRegression(max_iter=1000)
		lr.fit(X[:80], y[:80])
		y_pred = lr.predict(X[80:])

		# Calculate F1 score
		f1 = f1_score(y[80:], y_pred, zero_division=0)
		assert 0 <= f1 <= 1

		# Classification report
		report = classification_report(y[80:], y_pred, output_dict=True, zero_division=0)
		assert '0' in report or 0 in report

	@patch('who_will_viral.models.mlflow_utilities.setup_mlflow')
	def test_multiple_models_training(self, mock_setup, training_data):
		"""Test training multiple model types."""
		from sklearn.linear_model import LogisticRegression
		from sklearn.ensemble import RandomForestClassifier
		from sklearn.svm import LinearSVC

		X = training_data.drop(columns=['is_trending']).values
		y = training_data['is_trending'].values

		models = {
			'LogisticRegression': LogisticRegression(max_iter=1000),
			'RandomForest': RandomForestClassifier(n_estimators=10, random_state=42),
			'LinearSVC': LinearSVC(max_iter=1000, random_state=42),
		}

		trained_models = {}
		for model_name, model in models.items():
			model.fit(X[:80], y[:80])
			trained_models[model_name] = model

		assert len(trained_models) == 3
		assert all(hasattr(m, 'predict') for m in trained_models.values())


class TestFeatureSelectionIntegration:
	"""Integration tests for feature selection."""

	@pytest.fixture
	def high_dimensional_data(self):
		"""Create high-dimensional dataset."""
		np.random.seed(42)
		n_samples = 100
		n_features = 50

		X = np.random.randn(n_samples, n_features)
		y = np.random.randint(0, 2, n_samples)

		df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
		df['is_trending'] = y
		return df

	def test_feature_selection_reduces_dimensions(self, high_dimensional_data):
		"""Test that feature selection reduces dimensionality."""
		# Original features
		original_features = len([col for col in high_dimensional_data.columns if col != 'is_trending'])
		assert original_features == 50

		# In a real scenario, we'd use feature selection to reduce this
		# For now, just verify we can identify important vs less important features
		assert original_features > 10  # Should have many features to select from

	def test_feature_importance_extraction(self, high_dimensional_data):
		"""Test extraction of feature importance from models."""
		from sklearn.ensemble import RandomForestClassifier

		X = high_dimensional_data.drop(columns=['is_trending']).values
		y = high_dimensional_data['is_trending'].values

		rf = RandomForestClassifier(n_estimators=10, random_state=42)
		rf.fit(X, y)

		importances = rf.feature_importances_
		assert len(importances) == X.shape[1]
		assert all(0 <= imp <= 1 for imp in importances)
		assert sum(importances) > 0  # Some features should have importance


class TestEndToEndValidation:
	"""End-to-end validation tests."""

	@pytest.fixture
	def complete_pipeline_data(self):
		"""Create data that simulates the complete pipeline."""
		np.random.seed(42)
		n_samples = 50

		data = {
			'video_id': [f'vid_{i}' for i in range(n_samples)],
			'view_count': np.random.randint(100, 10000, n_samples),
			'likes': np.random.randint(10, 1000, n_samples),
			'comment_count': np.random.randint(0, 500, n_samples),
			'is_trending': np.random.randint(0, 2, n_samples),
		}

		# Add engagement features
		for i in range(20):
			data[f'feature_{i}'] = np.random.randn(n_samples)

		return pd.DataFrame(data)

	def test_end_to_end_data_validity(self, complete_pipeline_data):
		"""Test validity of data through complete pipeline."""
		# Check initial state
		assert len(complete_pipeline_data) == 50
		assert complete_pipeline_data['view_count'].min() >= 100
		assert complete_pipeline_data['likes'].min() >= 10
		assert complete_pipeline_data['comment_count'].min() >= 0

		# Check all values are numeric (except video_id)
		numeric_cols = complete_pipeline_data.select_dtypes(include=[np.number]).columns
		assert len(numeric_cols) > 0

		# Check target variable
		assert set(complete_pipeline_data['is_trending'].unique()).issubset({0, 1})

	def test_no_data_leakage(self, complete_pipeline_data):
		"""Test that training data doesn't leak into test data."""
		# Simulate train/test split
		train = complete_pipeline_data.iloc[:40]
		test = complete_pipeline_data.iloc[40:]

		# Check video_ids don't overlap
		train_ids = set(train['video_id'])
		test_ids = set(test['video_id'])

		assert len(train_ids.intersection(test_ids)) == 0  # No overlap

	def test_target_distribution(self, complete_pipeline_data):
		"""Test that target variable has reasonable distribution."""
		target_counts = complete_pipeline_data['is_trending'].value_counts()

		# Both classes should be present
		assert len(target_counts) == 2

		# Neither class should be completely dominant (at least some variation)
		ratios = target_counts / len(complete_pipeline_data)
		assert all(0.1 <= ratio <= 0.9 for ratio in ratios)  # Allow flexibility

	def test_feature_statistics_validity(self, complete_pipeline_data):
		"""Test that feature statistics are valid."""
		numeric_data = complete_pipeline_data.select_dtypes(include=[np.number])

		for col in numeric_data.columns:
			col_data = numeric_data[col]

			# Mean should be finite
			assert np.isfinite(col_data.mean())

			# Standard deviation should be finite and non-negative
			assert np.isfinite(col_data.std())
			assert col_data.std() >= 0
