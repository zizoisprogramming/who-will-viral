"""Model loader for predictions with full preprocessing pipeline.

This module loads the trained model and all preprocessing artifacts (embeddings,
PCA, scaler, RFECV) and handles inference end-to-end.
"""

import os
import pickle
from pathlib import Path

import joblib
from dotenv import load_dotenv

from who_will_viral.deployment_preprocessor import DeploymentPreprocessor

load_dotenv()


class ModelLoader:
	"""Load trained model and perform complete inference pipeline."""

	def __init__(self, model_path: str = None):
		"""Initialize model loader with all preprocessing artifacts.

		Args:
		    model_path: Path to trained model file. Uses MODEL_PATH env var if not provided.

		Environment variables (all required for full preprocessing):
		    MODEL_PATH: Path to trained classifier
		    PCA_PATH: Path to fitted PCA model
		    SCALER_PATH: Path to fitted RobustScaler
		    RFECV_PATH: Path to fitted RFECV feature selector
		    LANG_BASE_FREQS_PATH: Path to language base frequency dict
		    LANG_REGION_FREQS_PATH: Path to language region frequency dict
		    HF_TOKEN: Hugging Face token for embedding model
		"""
		self.model_path = model_path or os.getenv('MODEL_PATH', '{{MODEL_PATH_PLACEHOLDER}}')
		self.model = None
		self.preprocessor = None
		self.load()

	def load(self):
		"""Load model and initialize preprocessor."""
		# Load model
		if not Path(self.model_path).exists():
			raise FileNotFoundError(f'Model file not found: {self.model_path}')

		try:
			self.model = joblib.load(self.model_path)
		except Exception:
			with open(self.model_path, 'rb') as f:
				self.model = pickle.load(f)

		# Initialize preprocessor (loads PCA, scaler, RFECV, etc.)
		self.preprocessor = DeploymentPreprocessor()

	def predict(self, **feature_kwargs) -> dict:
		"""Make prediction from user input features.

		Args:
		    **feature_kwargs: Feature values as keyword arguments:
		        - views, likes, comments (metrics)
		        - title, description, tags_joined (text)
		        - lang_base, lang_region (language)
		        - has_cards, publish_hour, publish_dayofweek (metadata)
		        - duration_seconds, categoryId (video info)
		        - title_length, description_length, tag_count, title_has_caps_ratio, has_chapter (computed)

		Returns:
		    Dictionary with prediction, trend boolean, and probability
		"""
		if self.model is None:
			raise RuntimeError('Model not loaded. Check MODEL_PATH environment variable.')

		if self.preprocessor is None:
			raise RuntimeError('Preprocessor not initialized.')

		# Preprocess features (runs full pipeline)
		df = self.preprocessor.preprocess(**feature_kwargs)

		# Make prediction
		prediction = self.model.predict(df)[0]
		probability = None

		if hasattr(self.model, 'predict_proba'):
			proba = self.model.predict_proba(df)[0]
			probability = float(proba[1])

		return {
			'prediction': int(prediction),
			'will_trend': bool(prediction),
			'probability': probability,
		}

	def get_feature_names(self) -> list:
		"""Get list of expected feature names from model.

		Returns:
		    List of feature names
		"""
		if hasattr(self.model, 'feature_names_in_'):
			return list(self.model.feature_names_in_)
		return []
