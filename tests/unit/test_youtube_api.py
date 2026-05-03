import logging
from unittest.mock import MagicMock, patch

import pandas as pd

from who_will_viral.data_acquisition.youtube_api import YoutubeAPI


def test_items_to_dataframe():
    """Test that nested JSON from YouTube API flattens correctly."""
    api = YoutubeAPI(api_key='dummy', base_csv='dummy', logger=logging.getLogger())

    raw_items = [
        {
            'id': 'vid1',
            'snippet': {
                'title': 'My Video',
                'thumbnails': {'default': {'url': 'http://img.png'}, 'high': {'url': 'http://img-high.png'}},
            },
            'statistics': {'viewCount': '100'},
        }
    ]

    df = api.items_to_dataframe(raw_items)

    assert 'viewCount' in df.columns
    assert 'title' in df.columns
    assert 'thumbnail_link' in df.columns
    assert df['thumbnail_link'].iloc[0] == 'http://img.png'
    assert 'snippet.thumbnails.high.url' not in df.columns


def test_align_columns():
    api = YoutubeAPI(api_key='dummy', base_csv='dummy', logger=logging.getLogger())

    df_to_fix = pd.DataFrame({'A': [1], 'B': [2]})
    df_reference = pd.DataFrame({'B': [3], 'C': [4]})  # Requires 'C', drops 'A'

    aligned = api.align_columns(df_to_fix, df_reference)

    assert 'C' in aligned.columns
    assert 'A' not in aligned.columns
    assert aligned['C'].isna().all()  # Missing column should be filled with NaN


def test_add_pipeline_metadata():
    api = YoutubeAPI(api_key='dummy', base_csv='dummy', logger=logging.getLogger())

    df = pd.DataFrame({'video_id': ['vid1']})
    res = api.add_pipeline_metadata(df, is_trending=True, today='2026-04-10')

    assert res['is_trending'].iloc[0]
    assert res['trending_date'].iloc[0] == '2026-04-10'


def test_get_video_details_batched(mocker):
    """Test that the API correctly splits a large list of IDs into batches."""
    api = YoutubeAPI(api_key='dummy', base_csv='dummy', logger=logging.getLogger('dummy'))

    mock_youtube = MagicMock()
    api._youtube = mock_youtube

    mock_request = MagicMock()
    mock_request.execute.return_value = {'items': [{'id': 'vid1'}, {'id': 'vid2'}]}
    mock_youtube.videos().list.return_value = mock_request

    mocker.patch.object(api._rate, 'wait_if_needed')

    video_ids = ['v1', 'v2', 'v3']
    res = api.get_video_details_batched(video_ids, batch_size=2, delay=0)

    assert len(res) == 4
    assert mock_youtube.videos().list.call_count == 2


@patch('who_will_viral.data_acquisition.youtube_api.YoutubeAPI.get_trending_videos')
def test_fetch_trending(mock_get_trending, tmp_path):
    """Test that fetching trending videos properly writes and reads backup files."""
    api = YoutubeAPI(api_key='dummy', base_csv='dummy', logger=logging.getLogger('dummy'), backup_dir=str(tmp_path))

    mock_get_trending.return_value = [
        {'id': 'vid1', 'statistics': {'viewCount': '100'}, 'snippet': {'title': 'Trending Video'}}
    ]

    df_first = api._fetch_trending()
    assert 'is_trending' in df_first.columns
    assert mock_get_trending.call_count == 1
    assert (tmp_path / 'trending_videos_backup.json').exists()

    df_second = api._fetch_trending()
    assert len(df_second) == 1
    assert mock_get_trending.call_count == 1
