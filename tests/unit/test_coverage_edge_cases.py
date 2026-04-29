"""Additional tests for edge cases and branch coverage."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path

from src.who_will_viral.mlflow_utilities import setup_mlflow, run_experiment


class TestMlflowUtilities:
    """Tests for mlflow utilities."""
    
    @patch("who_will_viral.mlflow_utilities.mlflow.set_tracking_uri")
    @patch("who_will_viral.mlflow_utilities.mlflow.set_experiment")
    @patch("who_will_viral.mlflow_utilities.Path.mkdir")
    def test_setup_mlflow(self, mock_mkdir, mock_set_exp, mock_set_uri):
        """Test mlflow setup."""
        setup_mlflow()
        mock_set_uri.assert_called_once()
        mock_set_exp.assert_called_once()

    @patch("who_will_viral.mlflow_utilities.mlflow.start_run")
    @patch("who_will_viral.mlflow_utilities.mlflow.set_tags")
    @patch("who_will_viral.mlflow_utilities.mlflow.log_params")
    @patch("who_will_viral.mlflow_utilities.mlflow.log_metrics")
    @patch("who_will_viral.mlflow_utilities.mlflow.sklearn.log_model")
    @patch("who_will_viral.mlflow_utilities.mlflow.log_artifact")
    @patch("who_will_viral.mlflow_utilities.plt.show")
    @patch("who_will_viral.mlflow_utilities.plt.close")
    def test_run_experiment_with_best_params(
        self, mock_close, mock_show, mock_log_artifact,
        mock_log_model, mock_log_metrics, mock_log_params,
        mock_set_tags, mock_start_run
    ):
        """Test run_experiment with a GridSearchCV model."""
        # Create a mock model with best_params_
        model = MagicMock()
        model.best_params_ = {'C': 0.1}
        model.best_score_ = 0.85
        model.predict.return_value = np.array([0, 1, 0, 1])
        model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3], [0.6, 0.4], [0.1, 0.9]])
        
        X_tr = np.array([[1, 2], [3, 4]])
        y_tr = np.array([0, 1])
        X_ev = np.array([[2, 3], [4, 5], [1, 1], [2, 2]])
        y_ev = np.array([0, 1, 0, 1])
        
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock()
        
        metrics, returned_model = run_experiment(
            "Test Model",
            model,
            X_tr, y_tr, X_ev, y_ev
        )
        
        assert metrics is not None
        assert 'accuracy' in metrics
        assert 'f1' in metrics
        assert returned_model is model

    @patch("who_will_viral.mlflow_utilities.mlflow.start_run")
    @patch("who_will_viral.mlflow_utilities.mlflow.set_tags")
    @patch("who_will_viral.mlflow_utilities.mlflow.log_params")
    @patch("who_will_viral.mlflow_utilities.mlflow.log_metrics")
    @patch("who_will_viral.mlflow_utilities.mlflow.sklearn.log_model")
    @patch("who_will_viral.mlflow_utilities.mlflow.log_artifact")
    @patch("who_will_viral.mlflow_utilities.plt.show")
    @patch("who_will_viral.mlflow_utilities.plt.close")
    def test_run_experiment_without_predict_proba(
        self, mock_close, mock_show, mock_log_artifact,
        mock_log_model, mock_log_metrics, mock_log_params,
        mock_set_tags, mock_start_run
    ):
        """Test run_experiment with a model that doesn't have predict_proba."""
        # Create a mock model without predict_proba (like LinearSVC)
        model = MagicMock()
        del model.predict_proba  # Remove predict_proba
        model.best_params_ = {'C': 0.1}
        model.best_score_ = 0.85
        model.predict.return_value = np.array([0, 1, 0, 1])
        
        X_tr = np.array([[1, 2], [3, 4]])
        y_tr = np.array([0, 1])
        X_ev = np.array([[2, 3], [4, 5], [1, 1], [2, 2]])
        y_ev = np.array([0, 1, 0, 1])
        
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock()
        
        metrics, returned_model = run_experiment(
            "Test Model No Proba",
            model,
            X_tr, y_tr, X_ev, y_ev
        )
        
        assert metrics is not None
        assert metrics.get('roc_auc') is None

    @patch("who_will_viral.mlflow_utilities.mlflow.start_run")
    @patch("who_will_viral.mlflow_utilities.mlflow.set_tags")
    @patch("who_will_viral.mlflow_utilities.mlflow.log_params")
    @patch("who_will_viral.mlflow_utilities.mlflow.log_metrics")
    @patch("who_will_viral.mlflow_utilities.mlflow.sklearn.log_model")
    @patch("who_will_viral.mlflow_utilities.mlflow.log_artifact")
    @patch("who_will_viral.mlflow_utilities.plt.show")
    @patch("who_will_viral.mlflow_utilities.plt.close")
    def test_run_experiment_skip_fit(
        self, mock_close, mock_show, mock_log_artifact,
        mock_log_model, mock_log_metrics, mock_log_params,
        mock_set_tags, mock_start_run
    ):
        """Test run_experiment with skip_fit=True."""
        model = MagicMock()
        # Remove best_params_ to test the params path
        del model.best_params_
        model.predict.return_value = np.array([0, 1, 0, 1])
        model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3], [0.6, 0.4], [0.1, 0.9]])
        
        X_tr = np.array([[1, 2], [3, 4]])
        y_tr = np.array([0, 1])
        X_ev = np.array([[2, 3], [4, 5], [1, 1], [2, 2]])
        y_ev = np.array([0, 1, 0, 1])
        params = {'C': 0.5}
        
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock()
        
        metrics, returned_model = run_experiment(
            "Test Model Skip Fit",
            model,
            X_tr, y_tr, X_ev, y_ev,
            params=params,
            skip_fit=True
        )
        
        # Verify fit was not called
        model.fit.assert_not_called()
        assert metrics is not None
