import os

import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import RobustScaler

load_dotenv()

ALREADY_PROCESSED = [
	'lang_base',
	'lang_region',
	'like_to_view_ratio',
	'comment_to_view_ratio',
	'has_cards',
]


class FeatureScaling:
	def __init__(self, train_path, val_path, test_path):
		self.train_path = train_path
		self.val_path = val_path
		self.test_path = test_path

		self.scaled_train_path = os.getenv('SCALED_TRAIN_PATH')
		self.scaled_val_path = os.getenv('SCALED_VAL_PATH')
		self.scaled_test_path = os.getenv('SCALED_TEST_PATH')

	def run(self):

		df_train = pd.read_csv(self.train_path)
		df_val = pd.read_csv(self.val_path)
		df_test = pd.read_csv(self.test_path)

		COLS_TO_SCALE = [
			c
			for c in df_train.columns
			if c not in ALREADY_PROCESSED + [col for col in df_train.columns if 'pca' in col] and c != 'is_trending'
		]

		scaler = RobustScaler()
		df_train[COLS_TO_SCALE] = scaler.fit_transform(df_train[COLS_TO_SCALE])
		df_val[COLS_TO_SCALE] = scaler.transform(df_val[COLS_TO_SCALE])
		df_test[COLS_TO_SCALE] = scaler.transform(df_test[COLS_TO_SCALE])

		df_train.to_csv(self.scaled_train_path, index=False)
		df_val.to_csv(self.scaled_val_path, index=False)
		df_test.to_csv(self.scaled_test_path, index=False)