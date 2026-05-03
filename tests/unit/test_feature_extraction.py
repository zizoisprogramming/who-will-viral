from unittest.mock import patch

import numpy as np
import pandas as pd

from who_will_viral.feature_engineering.feature_extraction import FeatureExtraction


@patch('who_will_viral.feature_engineering.feature_extraction.SentenceTransformer')
def test_get_duration_seconds(mock_model):
	"""Test parsing ISO 8601 duration strings."""
	extractor = FeatureExtraction()

	assert extractor._get_duration_seconds('PT1M') == 60
	assert extractor._get_duration_seconds('PT1H1M1S') == 3661
	assert extractor._get_duration_seconds('P1DT2H') == 93600
	assert extractor._get_duration_seconds('invalid') == 0
	assert extractor._get_duration_seconds(None) == 0


@patch('who_will_viral.feature_engineering.feature_extraction.SentenceTransformer')
def test_count_emojis(mock_model):
	"""Test emoji counting in title and description."""
	extractor = FeatureExtraction()
	df = pd.DataFrame({'title': ['Hello 🌍!', 'No emojis here'], 'description': ['Wow 🔥✨', 'Just text']})

	result = extractor._count_emojis(df)
	assert result['title_emoji_count'].tolist() == [1, 0]
	assert result['description_emoji_count'].tolist() == [2, 0]


@patch('who_will_viral.feature_engineering.feature_extraction.SentenceTransformer')
def test_feature_extraction_run(mock_model, tmp_path, mocker):
	"""Test the entire extraction pipeline top-to-bottom."""
	mocker.patch('os.getenv', return_value=str(tmp_path / 'extracted.csv'))

	mock_instance = mock_model.return_value
	mock_instance.encode.return_value = np.array([[0.1, 0.2], [0.1, 0.2]])

	df = pd.DataFrame(
		{
			'video_id': ['v1', 'v2'],
			'likes': [100, 200],
			'comment_count': [10, 20],
			'view_count': [1000, 2000],
			'tags': ["['tag1', 'tag2']", 'tag3,tag4'],
			'title': ['Title 1', 'Title 2 😊'],
			'description': ['Desc 1', 'Desc 2'],
			'card_count': [1, 0],
			'chapter_count': [0, 5],
			'publishedAt': ['2024-01-01T12:00:00Z', '2024-01-02T15:30:00Z'],
			'duration': ['PT1M', 'PT2M'],
			'defaultLanguage': ['en-US', 'es-ES'],
		}
	)

	extractor = FeatureExtraction()
	result_df = extractor.run(df)

	assert 'like_to_view_ratio' in result_df.columns
	assert 'publish_hour' in result_df.columns
	assert 'lang_base' in result_df.columns
	assert 'tags_joined_emb_0' in result_df.columns
	assert result_df['has_cards'].tolist() == [1, 0]
	assert result_df['has_chapter'].tolist() == [0, 1]
