import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df():
	"""A small mock dataframe simulating raw YouTube data."""
	return pd.DataFrame(
		{
			'video_id': ['vid1', 'vid2', 'vid3', 'vid1'],
			'view_count': [1000, 500, 2000, 1000],
			'likes': [100, 600, 150, 100],
			'comment_count': [10, 5, 20, 10],
			'comments_disabled': ['False', 'False', 'True', 'False'],
			'tags': ["['tag1', 'tag2']", 'tag3, tag4', None, "['tag1', 'tag2']"],
			'defaultLanguage': ['en-US', 'es-ES', 'invalid-code', 'en-US'],
			'duration': ['PT1H5M', 'PT30S', 'P1DT2H', 'PT1H5M'],
			'title': ['Title 1 😊', 'Title 2', 'Title 3', 'Title 1 😊'],
			'description': ['Desc 1', None, 'Desc 3', 'Desc 1'],
		}
	)
