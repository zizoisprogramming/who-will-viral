"""Complete preprocessing pipeline for deployment (replicates training pipeline).

This module performs all preprocessing steps in sequence:
1. Feature extraction (embeddings, calculations, etc.)
2. Frequency encoding
3. PCA transformation
4. Feature scaling
5. Feature selection
"""

import os
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Configuration
EMBEDDINGS_COLS = ['tags_joined', 'description', 'title']
EMOJIS_COLS = ['description', 'title']
RATIOS = {'likes': 'like_to_view_ratio', 'comments': 'comment_to_view_ratio'}
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

ALREADY_PROCESSED = [
	'lang_base',
	'lang_region',
	'like_to_view_ratio',
	'comment_to_view_ratio',
	'has_cards',
]


class DeploymentPreprocessor:
	"""Complete preprocessing pipeline for inference."""

	def __init__(
		self,
		embedding_model_name: str = 'all-MiniLM-L6-v2',
		pca_path: str = None,
		scaler_path: str = None,
		rfecv_path: str = None,
		lang_base_freqs_path: str = None,
		lang_region_freqs_path: str = None,
		hf_token: str = None,
	):
		"""Initialize preprocessor with required artifacts.

		Args:
		    embedding_model_name: Name of sentence transformer model
		    pca_path: Path to fitted PCA model ({{PCA_PATH_PLACEHOLDER}})
		    scaler_path: Path to fitted RobustScaler ({{SCALER_PATH_PLACEHOLDER}})
		    rfecv_path: Path to fitted RFECV selector ({{RFECV_PATH_PLACEHOLDER}})
		    lang_base_freqs_path: Path to lang_base frequency dict ({{LANG_BASE_FREQS_PATH_PLACEHOLDER}})
		    lang_region_freqs_path: Path to lang_region frequency dict ({{LANG_REGION_FREQS_PATH_PLACEHOLDER}})
		    hf_token: Hugging Face token for embedding model
		"""
		self.pca_path = pca_path or os.getenv('PCA_PATH', '{{PCA_PATH_PLACEHOLDER}}')
		self.scaler_path = scaler_path or os.getenv('SCALER_PATH', '{{SCALER_PATH_PLACEHOLDER}}')
		self.rfecv_path = rfecv_path or os.getenv('RFECV_PATH', '{{RFECV_PATH_PLACEHOLDER}}')
		self.lang_base_freqs_path = lang_base_freqs_path or os.getenv(
			'LANG_BASE_FREQS_PATH', '{{LANG_BASE_FREQS_PATH_PLACEHOLDER}}'
		)
		self.lang_region_freqs_path = lang_region_freqs_path or os.getenv(
			'LANG_REGION_FREQS_PATH', '{{LANG_REGION_FREQS_PATH_PLACEHOLDER}}'
		)
		self.hf_token = hf_token or os.getenv('HF_TOKEN')

		# Load models
		self.embedding_model = SentenceTransformer(embedding_model_name, token=self.hf_token)
		self.pca = self._load_artifact(self.pca_path, 'PCA')
		self.scaler = self._load_artifact(self.scaler_path, 'Scaler')
		self.rfecv = self._load_artifact(self.rfecv_path, 'RFECV')
		self.lang_base_freqs = self._load_artifact(self.lang_base_freqs_path, 'Lang Base Frequencies')
		self.lang_region_freqs = self._load_artifact(self.lang_region_freqs_path, 'Lang Region Frequencies')

		self.selected_features = None
		if self.rfecv is not None:
			self.selected_features = self.rfecv.get_feature_names_out().tolist()

	@staticmethod
	def _load_artifact(path: str, name: str):
		"""Load artifact from disk with joblib/pickle fallback."""
		if not path or '{{' in path:  # Skip if placeholder
			return None
		if not Path(path).exists():
			print(f'Warning: {name} not found at {path}')
			return None
		try:
			return joblib.load(path)
		except Exception:
			try:
				with open(path, 'rb') as f:
					return pickle.load(f)
			except Exception as e:
				print(f'Error loading {name}: {e}')
				return None

	def preprocess(
		self,
		views: int,
		likes: int,
		comments: int,
		title: str = '',
		description: str = '',
		tags_joined: str = '',
		lang_base: str = 'en',
		lang_region: str = 'US',
		has_cards: int = 0,
		publish_hour: int = 12,
		publish_dayofweek: int = 3,
		duration_seconds: int = 0,
		categoryId: int = 28,
		title_length: int = 0,
		description_length: int = 0,
		tag_count: int = 0,
		title_has_caps_ratio: float = 0.0,
		has_chapter: int = 0,
	) -> pd.DataFrame:
		"""Preprocess features for prediction.

		This replicates the full training pipeline:
		1. Calculate ratios
		2. Generate embeddings
		3. Apply PCA
		4. Encode language features
		5. Scale features
		6. Select features

		Args:
		    views, likes, comments: Video metrics
		    title, description, tags_joined: Text fields for embedding
		    lang_base, lang_region: Language codes
		    has_cards, publish_hour, publish_dayofweek: Video metadata
		    duration_seconds: Video duration
		    categoryId: YouTube category
		    title_length, description_length, tag_count, title_has_caps_ratio, has_chapter: Computed features

		Returns:
		    Preprocessed feature dataframe ready for model prediction
		"""
		# Step 1: Create initial dataframe
		data = {
			'view_count': views,
			'likes': likes,
			'comment_count': comments,
			'title': title,
			'description': description,
			'tags_joined': tags_joined,
			'lang_base': lang_base,
			'lang_region': lang_region,
			'has_cards': has_cards,
			'publish_hour': publish_hour,
			'publish_dayofweek': publish_dayofweek,
			'duration_seconds': duration_seconds,
			'categoryId': categoryId,
			'title_length': title_length,
			'description_length': description_length,
			'tag_count': tag_count,
			'title_has_caps_ratio': title_has_caps_ratio,
			'has_chapter': has_chapter,
		}
		df = pd.DataFrame([data])

		# Step 2: Calculate ratios
		if df['view_count'].iloc[0] > 0:
			df['like_to_view_ratio'] = df['likes'] / df['view_count']
			df['comment_to_view_ratio'] = df['comment_count'] / df['view_count']
		else:
			df['like_to_view_ratio'] = 0
			df['comment_to_view_ratio'] = 0

		# Step 3: Generate embeddings
		df = self._get_embeddings(df)

		# Step 4: Encode language features using frequency mapping
		df = self._encode_language_features(df)

		# Step 5: Apply PCA transformation (replaces embedding columns with PCA components)
		df = self._apply_pca(df)

		# Step 6: Scale features
		df = self._scale_features(df)

		# Step 7: Select features (and reorder)
		df = self._select_features(df)

		return df

	def _get_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Generate sentence embeddings for text columns."""
		if self.embedding_model is None:
			print('Warning: Embedding model not loaded, skipping embeddings')
			for col in EMBEDDINGS_COLS:
				for i in range(384):
					df[f'{col}_emb_{i}'] = 0.0
			return df

		for col in EMBEDDINGS_COLS:
			if col not in df.columns:
				df[col] = ''
			text = df[col].iloc[0] if len(df) > 0 else ''
			text = text if text else 'no text'

			embeddings = self.embedding_model.encode([text], show_progress_bar=False)
			embedding_df = pd.DataFrame(embeddings, columns=[f'{col}_emb_{i}' for i in range(embeddings.shape[1])])
			df = pd.concat([df.reset_index(drop=True), embedding_df], axis=1)

		return df

	def _encode_language_features(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Encode language features using frequency mapping."""
		if self.lang_base_freqs is not None and 'lang_base' in df.columns:
			df['lang_base'] = df['lang_base'].map(self.lang_base_freqs).fillna(0.0)

		if self.lang_region_freqs is not None and 'lang_region' in df.columns:
			df['lang_region'] = df['lang_region'].map(self.lang_region_freqs).fillna(0.0)

		return df

	def _apply_pca(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Apply PCA transformation to embedding columns."""
		if self.pca is None:
			print('Warning: PCA model not loaded, skipping PCA')
			return df

		# Find embedding columns
		emb_cols = [col for col in df.columns if 'emb' in col and 'embeddable' not in col]

		if not emb_cols:
			print('Warning: No embedding columns found')
			return df

		# Apply PCA
		pca_vals = self.pca.transform(df[emb_cols])
		pca_cols = [f'pca_{i}' for i in range(pca_vals.shape[1])]

		# Replace embedding columns with PCA columns
		df = df.drop(columns=emb_cols)
		pca_df = pd.DataFrame(pca_vals, columns=pca_cols, index=df.index)
		df = pd.concat([df, pca_df], axis=1)

		return df

	def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Apply RobustScaler to numerical features."""
		if self.scaler is None:
			print('Warning: Scaler not loaded, skipping scaling')
			return df

		# Identify columns to scale (exclude language, ratios, pca, flags, categorical)
		cols_to_scale = [
			c
			for c in df.columns
			if c not in ALREADY_PROCESSED
			and not any(x in c for x in ['pca', 'emb', 'is_trending', 'categoryId'])
			and c not in ['lang_base', 'lang_region']
		]

		if cols_to_scale and hasattr(self.scaler, 'transform'):
			df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])

		return df

	def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Select and reorder features based on RFECV."""
		if self.selected_features is None:
			print('Warning: Selected features not available, returning all features')
			return df

		# Filter to selected features only
		available_features = [f for f in self.selected_features if f in df.columns]
		return df[available_features]
