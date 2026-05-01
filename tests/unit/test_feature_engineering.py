import numpy as np
import pandas as pd

from who_will_viral.feature_engineering.feature_scaling import FeatureScaling
from who_will_viral.feature_engineering.feature_selection import COLS_TO_DROP, FeatureSelection


def test_feature_selection(tmp_path, mocker):
	"""Test that FeatureSelection drops cols, groups rare, and runs PCA."""

	df_data = {
		'video_id': [f'v{i}' for i in range(40)],
		'categoryId': [22] * 39 + [999],  # 999 is rare
		'is_trending': [1] * 10 + [0] * 30,
		'lang_base': ['en'] * 40,
		'lang_region': ['US'] * 40,
		'description_emb_0': [0.1] * 40,
	}

	for col in COLS_TO_DROP:
		df_data[col] = [0] * 40

	df = pd.DataFrame(df_data)

	mocker.patch('who_will_viral.feature_engineering.feature_selection.PCA.fit')
	mocker.patch(
		'who_will_viral.feature_engineering.feature_selection.PCA.transform',
		side_effect=lambda X: np.ones((len(X), 100)),
	)

	mock_rfecv_class = mocker.patch('who_will_viral.feature_engineering.feature_selection.RFECV')
	mock_rfecv_instance = mock_rfecv_class.return_value

	def mock_fit(X, y):
		mock_rfecv_instance.support_ = np.array([True] * X.shape[1])

	mock_rfecv_instance.fit.side_effect = mock_fit

	train_path = str(tmp_path / 'train.csv')
	val_path = str(tmp_path / 'val.csv')
	test_path = str(tmp_path / 'test.csv')

	selector = FeatureSelection(df, train_path, val_path, test_path)
	selector.run()

	assert 'title' not in selector.df_train.columns
	assert 'is_trending' in selector.df_train.columns
	assert len(selector.df_train) > 0  # Ensure the splits actually contain data


def test_feature_scaling(tmp_path, mocker):
	"""Test that RobustScaler is applied to the correct columns."""
	df = pd.DataFrame({'view_count': [10, 100, 1000], 'is_trending': [0, 1, 0], 'has_cards': [1, 0, 1]})

	train_path = tmp_path / 'train.csv'
	df.to_csv(train_path, index=False)
	df.to_csv(tmp_path / 'val.csv', index=False)
	df.to_csv(tmp_path / 'test.csv', index=False)

	mocker.patch('os.getenv', return_value=str(tmp_path / 'scaled.csv'))

	scaler = FeatureScaling(train_path, tmp_path / 'val.csv', tmp_path / 'test.csv')
	scaler.run()

	scaled_train = pd.read_csv(tmp_path / 'scaled.csv')
	assert 'view_count' in scaled_train.columns
	assert scaled_train['is_trending'].tolist() == [0, 1, 0]
