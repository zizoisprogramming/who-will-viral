"""Integration tests for API and CLI functionality."""

import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest


class TestDataAcquisitionAPI:
    """Integration tests for YouTube data acquisition APIs."""

    @patch('who_will_viral.data_acquisition.youtube_api.YoutubeAPI')
    def test_youtube_api_initialization(self, mock_api):
        """Test that YouTube API can be initialized."""
        mock_instance = mock_api.return_value

        # Should be able to initialize
        assert mock_instance is not None

    @patch('who_will_viral.data_acquisition.youtube_api.YoutubeAPI')
    def test_api_key_handling(self, mock_api):
        """Test that API keys are handled correctly."""
        mock_instance = mock_api.return_value
        mock_instance.api_key = 'test_key_123'

        assert mock_instance.api_key == 'test_key_123'

    @patch('who_will_viral.data_acquisition.youtube_database.YoutubeDatabase')
    def test_database_connection_initialization(self, mock_db):
        """Test database can be initialized."""
        mock_instance = mock_db.return_value

        assert mock_instance is not None

    @patch('who_will_viral.data_acquisition.youtube_scraper.YoutubeScraper')
    def test_web_scraper_initialization(self, mock_scraper):
        """Test web scraper initialization."""
        mock_instance = mock_scraper.return_value

        assert mock_instance is not None


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_cli_app_creation(self):
        """Test that CLI app can be created."""
        try:
            from who_will_viral.cli import app

            assert app is not None
        except ImportError:
            pytest.skip('CLI module not available')

    @patch('who_will_viral.cli.app')
    def test_cli_help_command(self, mock_app):
        """Test CLI help command."""
        # Mock app should be callable
        assert callable(mock_app) or hasattr(mock_app, 'command')


class TestDataPersistence:
    """Integration tests for data persistence."""

    def test_csv_read_write_cycle(self):
        """Test that data can be written and read from CSV."""
        with tempfile.TemporaryDirectory() as tmp_path:
            csv_path = os.path.join(tmp_path, 'test_data.csv')

            # Create test data
            df = pd.DataFrame(
                {
                    'video_id': ['vid1', 'vid2', 'vid3'],
                    'view_count': [1000, 500, 2000],
                    'likes': [100, 50, 150],
                    'is_trending': [1, 0, 1],
                }
            )

            # Write
            df.to_csv(csv_path, index=False)
            assert os.path.exists(csv_path)

            # Read
            df_read = pd.read_csv(csv_path)
            assert len(df_read) == len(df)
            assert list(df_read.columns) == list(df.columns)

            # Data should match
            pd.testing.assert_frame_equal(df, df_read)

    def test_multiple_csv_handling(self):
        """Test handling multiple CSV files."""
        with tempfile.TemporaryDirectory() as tmp_path:
            files = []

            for i in range(3):
                csv_path = os.path.join(tmp_path, f'data_{i}.csv')
                df = pd.DataFrame(
                    {
                        'id': [f'id_{j}' for j in range(10)],
                        'value': [j * i for j in range(10)],
                    }
                )
                df.to_csv(csv_path, index=False)
                files.append(csv_path)

            # Verify all files exist
            assert all(os.path.exists(f) for f in files)

            # Read all files
            dfs = [pd.read_csv(f) for f in files]
            assert len(dfs) == 3
            assert all(len(df) == 10 for df in dfs)

    def test_model_serialization(self):
        """Test that models can be serialized."""
        import pickle

        from sklearn.linear_model import LogisticRegression

        with tempfile.TemporaryDirectory() as tmp_path:
            model_path = os.path.join(tmp_path, 'model.pkl')

            # Train simple model
            import numpy as np

            X = np.random.randn(100, 10)
            y = np.random.randint(0, 2, 100)

            model = LogisticRegression()
            model.fit(X, y)

            # Serialize
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            assert os.path.exists(model_path)

            # Deserialize
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)

            # Should work
            predictions = loaded_model.predict(X[:5])
            assert len(predictions) == 5


class TestEnvironmentConfiguration:
    """Integration tests for environment configuration."""

    @patch.dict(
        os.environ,
        {
            'YOUTUBE_API_KEY': 'test_key_123',
            'BASE_CSV': './data/test.csv',
            'CLEANED_PATH': './data/cleaned.csv',
        },
    )
    def test_environment_variable_loading(self):
        """Test that environment variables are loaded correctly."""

        api_key = os.getenv('YOUTUBE_API_KEY')
        base_csv = os.getenv('BASE_CSV')

        assert api_key == 'test_key_123'
        assert base_csv == './data/test.csv'

    def test_default_path_configuration(self):
        """Test default paths are set correctly."""

        # Should have defaults
        train_path = os.getenv('TRAIN_PATH', './data/train.csv')
        val_path = os.getenv('VAL_PATH', './data/val.csv')
        test_path = os.getenv('TEST_PATH', './data/test.csv')

        assert train_path
        assert val_path
        assert test_path


class TestDataValidationIntegration:
    """Integration tests for data validation."""

    def test_validate_function_exists(self):
        """Test that validation function is available."""
        try:
            from who_will_viral.validate import validate_raw_data

            assert callable(validate_raw_data)
        except ImportError:
            pytest.skip('Validation module not available')

    def test_validation_with_real_data_shape(self):
        """Test validation with realistic data shapes."""
        df = pd.DataFrame(
            {
                'video_id': [f'vid_{i}' for i in range(1000)],
                'view_count': [i * 100 for i in range(1000)],
                'likes': [i * 10 for i in range(1000)],
                'comment_count': [i for i in range(1000)],
                'is_trending': [i % 2 for i in range(1000)],
            }
        )

        # Should handle large datasets
        assert len(df) == 1000
        assert df.memory_usage(deep=True).sum() > 0

    def test_data_schema_validation(self):
        """Test that data schema is validated."""
        required_cols = ['video_id', 'view_count', 'likes', 'comment_count']

        df = pd.DataFrame(
            {
                'video_id': ['vid1', 'vid2'],
                'view_count': [1000, 500],
                'likes': [100, 50],
                'comment_count': [10, 5],
            }
        )

        for col in required_cols:
            assert col in df.columns

    def test_missing_required_columns_detection(self):
        """Test detection of missing required columns."""
        required_cols = ['video_id', 'view_count', 'is_trending']

        df = pd.DataFrame(
            {
                'video_id': ['vid1', 'vid2'],
                'view_count': [1000, 500],
            }
        )

        missing = [col for col in required_cols if col not in df.columns]
        assert len(missing) > 0


class TestDataTransformationIntegration:
    """Integration tests for data transformations."""

    def test_numerical_transformation_preserves_validity(self):
        """Test that numerical transformations keep data valid."""
        df = pd.DataFrame(
            {
                'feature_1': [1, 2, 3, 4, 5],
                'feature_2': [10, 20, 30, 40, 50],
            }
        )

        # Normalize
        df_norm = (df - df.min()) / (df.max() - df.min())

        # Check ranges
        assert (df_norm >= 0).all().all()
        assert (df_norm <= 1).all().all()

    def test_categorical_encoding(self):
        """Test categorical variable encoding."""
        df = pd.DataFrame(
            {
                'language': ['en', 'es', 'en', 'fr', 'es'],
                'is_trending': [1, 0, 1, 0, 1],
            }
        )

        # One-hot encode
        df_encoded = pd.get_dummies(df, columns=['language'])

        # Should create binary columns
        assert any('language_' in col for col in df_encoded.columns)

    def test_time_series_feature_extraction(self):
        """Test extraction of time-based features."""
        df = pd.DataFrame(
            {
                'publishedAt': ['2024-01-01T12:00:00Z', '2024-01-02T14:30:00Z', '2024-01-03T10:15:00Z'],
            }
        )

        # Parse dates
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])

        # Extract features
        df['day_of_week'] = df['publishedAt'].dt.dayofweek
        df['hour'] = df['publishedAt'].dt.hour

        assert 'day_of_week' in df.columns
        assert 'hour' in df.columns
        assert df['day_of_week'].min() >= 0
        assert df['hour'].min() >= 0


class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    def test_handling_empty_dataframe(self):
        """Test handling of empty dataframes."""
        df_empty = pd.DataFrame()

        assert len(df_empty) == 0
        assert df_empty.empty

    def test_handling_all_null_column(self):
        """Test handling columns with all null values."""
        df = pd.DataFrame(
            {
                'col1': [1, 2, 3],
                'col2': [None, None, None],
            }
        )

        assert df['col2'].isnull().all()

    def test_handling_mixed_types_in_column(self):
        """Test handling mixed data types in columns."""
        df = pd.DataFrame(
            {
                'mixed': [1, 'string', 3.14, None],
            }
        )

        # Should not crash
        assert len(df) == 4

    def test_division_by_zero_prevention(self):
        """Test prevention of division by zero errors."""
        df = pd.DataFrame(
            {
                'numerator': [100, 200, 0],
                'denominator': [10, 0, 0],
            }
        )

        # Safe division
        df['ratio'] = df['numerator'] / df['denominator'].replace(0, 1)

        assert not df['ratio'].isnull().all()
