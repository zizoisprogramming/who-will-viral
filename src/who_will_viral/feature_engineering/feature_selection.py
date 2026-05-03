import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

COLS_TO_DROP = [
	'video_id',
	'title',
	'publishedAt',
	'channelId',
	'tags',
	'likes',
	'comment_count',
	'description',
	'duration',
	'tags_joined',
	'defaultLanguage',
	'supports_miniplayer',
	'playability_status',
	'definition',
	'dimension',
	'projection',
	'channelTitle',
	'has_paid_promotion',
	'card_count',
]

CATEGORY_ID_COUNT = 500


class FeatureSelection:
	def __init__(self, df, train_path, val_path, test_path):
		self.df = df
		self.train_path = train_path
		self.val_path = val_path
		self.test_path = test_path

	def _drop_features(self):
		self.df.drop(COLS_TO_DROP, axis=1, inplace=True)

	def _group_rare(self):

		counts = self.df['categoryId'].value_counts()
		rare = counts[counts < CATEGORY_ID_COUNT].index

		self.df['categoryId'] = self.df['categoryId'].replace(rare, 29)

	def _split_data(self):
		self.df = self.df.reset_index(drop=True)

		temp, test = train_test_split(self.df, test_size=0.2, random_state=42)
		train, val = train_test_split(temp, test_size=0.25, random_state=42)

		self.df_train = train.reset_index(drop=True)
		self.df_val = val.reset_index(drop=True)
		self.df_test = test.reset_index(drop=True)

	def _encode_features(self):
		freq_base = self.df_train['lang_base'].value_counts(normalize=True)
		freq_region = self.df_train['lang_region'].value_counts(normalize=True)

		self.df_train['lang_base'] = self.df_train['lang_base'].map(freq_base)
		self.df_val['lang_base'] = self.df_val['lang_base'].map(freq_base)
		self.df_test['lang_base'] = self.df_test['lang_base'].map(freq_base)

		self.df_train['lang_region'] = self.df_train['lang_region'].map(freq_region)
		self.df_val['lang_region'] = self.df_val['lang_region'].map(freq_region)
		self.df_test['lang_region'] = self.df_test['lang_region'].map(freq_region)

	def _transform(self):
		TO_TRANSFORM = [col for col in self.df_train.columns if 'emb' in col and 'embeddable' not in col]

		n_components = 100

		pca = PCA(n_components=n_components)
		_ = pca.fit(self.df_train[TO_TRANSFORM])

		pca_cols = [f'pca_{i}' for i in range(n_components)]

		TO_TRANSFORM_vals_train = pca.transform(self.df_train[TO_TRANSFORM])
		TO_TRANSFORM_vals_val = pca.transform(self.df_val[TO_TRANSFORM])
		TO_TRANSFORM_vals_test = pca.transform(self.df_test[TO_TRANSFORM])

		self.df_train = pd.concat(
			[
				self.df_train.drop(columns=TO_TRANSFORM),
				pd.DataFrame(TO_TRANSFORM_vals_train, columns=pca_cols, index=self.df_train.index),
			],
			axis=1,
		)
		self.df_val = pd.concat(
			[
				self.df_val.drop(columns=TO_TRANSFORM),
				pd.DataFrame(TO_TRANSFORM_vals_val, columns=pca_cols, index=self.df_val.index),
			],
			axis=1,
		)
		self.df_test = pd.concat(
			[
				self.df_test.drop(columns=TO_TRANSFORM),
				pd.DataFrame(TO_TRANSFORM_vals_test, columns=pca_cols, index=self.df_test.index),
			],
			axis=1,
		)

	def _select_features(self):
		ones = self.df_train[self.df_train['is_trending'] == 1]
		zeros = self.df_train[self.df_train['is_trending'] == 0].sample(n=len(ones), random_state=42)

		sample = pd.concat([ones, zeros]).sample(frac=1, random_state=42)
		dt_cf = DecisionTreeClassifier(random_state=42)
		rfecv = RFECV(estimator=dt_cf, step=1, cv=5, scoring='f1', n_jobs=-1)
		rfecv.fit(sample.drop(columns=['is_trending']), sample['is_trending'])

		selected_cols = self.df_train.drop(columns=['is_trending']).columns[rfecv.support_].tolist()
		print(f'Selected {len(selected_cols)} features')
		selected_cols.append('is_trending')

		self.df_train = self.df_train[selected_cols]
		self.df_val = self.df_val[selected_cols]
		self.df_test = self.df_test[selected_cols]

		self.df_train.to_csv(self.train_path, index=False)
		self.df_val.to_csv(self.val_path, index=False)
		self.df_test.to_csv(self.test_path, index=False)

	def run(self):
		self._drop_features()
		self._group_rare()
		self._split_data()
		self._encode_features()
		self._transform()
		self._select_features()
