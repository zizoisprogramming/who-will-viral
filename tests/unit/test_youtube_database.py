import logging

import pandas as pd

from who_will_viral.data_acquisition.youtube_database import YoutubeDatabase


def test_youtube_database_run(tmp_path):
	"""Test loading the base CSV dataset."""
	csv_path = tmp_path / 'dummy_base.csv'
	pd.DataFrame({'video_id': ['v1', 'v2'], 'likes': [10, 20]}).to_csv(csv_path, index=False)

	db = YoutubeDatabase(logger=logging.getLogger('dummy'), path=str(csv_path))
	df = db.run()

	assert len(df) == 2
	assert 'video_id' in df.columns
