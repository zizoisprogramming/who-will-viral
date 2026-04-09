import pandas as pd
from unittest.mock import patch
from who_will_viral.acquire import YoutubePipeline

@patch("who_will_viral.acquire.YoutubeAPI")
@patch("who_will_viral.acquire.YoutubeScraper")
@patch("who_will_viral.acquire.YoutubeDatabase")
def test_pipeline_run(MockDB, MockScraper, MockAPI, mocker, tmp_path):
    
    def mock_getenv(key, default=None):
        if "DIR" in key: return str(tmp_path / "backup_dir")
        if "CSV" in key: return str(tmp_path / f"{key}.csv")
        return "dummy_token"
        
    mocker.patch("os.getenv", side_effect=mock_getenv)
    
    db_instance = MockDB.return_value
    db_instance.run.return_value = pd.DataFrame({"video_id": ["vid1", "vid2"]})
    
    api_instance = MockAPI.return_value
    api_instance.run.return_value = pd.DataFrame({
        "video_id": ["vid1", "vid2"], 
        "comments_disabled_x": [True, False]
    })
    
    scraper_instance = MockScraper.return_value
    scraper_instance.scrape_videos.return_value = pd.DataFrame({
        "video_id": ["vid1", "vid2"], 
        "comments_disabled_y": [False, True] 
    })
    
    pipeline = YoutubePipeline()
    final_df = pipeline.run()
    
    assert "comments_disabled" in final_df.columns
    assert "comments_disabled_x" not in final_df.columns
    assert "comments_disabled_y" not in final_df.columns
    assert len(final_df) == 2