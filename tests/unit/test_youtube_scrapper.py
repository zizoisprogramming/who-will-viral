import logging
from unittest.mock import MagicMock, patch

from bs4 import BeautifulSoup

from who_will_viral.data_acquisition.youtube_scraper import YoutubeScraper


def test_extract_playability():
    """Test extracting playability status from raw player data JSON."""
    scraper = YoutubeScraper()

    mock_player_data_valid = {
        "playabilityStatus": {"status": "OK"},
        "microformat": {"playerMicroformatRenderer": {"isFamilySafe": True}}
    }

    mock_player_data_missing = {}

    res_valid = scraper._extract_playability(mock_player_data_valid)
    assert res_valid["status"] == "OK"
    assert res_valid["supports_miniplayer"] is True

    res_missing = scraper._extract_playability(mock_player_data_missing)
    assert res_missing["status"] == "UNKNOWN"
    assert res_missing["supports_miniplayer"] is False


def test_check_robots_txt(mocker):
    mock_session = MagicMock()
    dummy_logger = logging.getLogger("dummy")

    scraper = YoutubeScraper(session=mock_session, logger=dummy_logger)

    mock_rp = mocker.patch("who_will_viral.data_acquisition.youtube_scraper.RobotFileParser")
    mock_rp_instance = mock_rp.return_value
    mock_rp_instance.can_fetch.return_value = True

    url = "https://www.youtube.com/watch?v=12345"

    is_allowed = scraper.check_robots_txt(url)
    assert is_allowed is True
    mock_rp_instance.read.assert_called_once()

    is_allowed_cached = scraper.check_robots_txt("https://www.youtube.com/watch?v=67890")
    assert is_allowed_cached is True
    assert mock_rp_instance.read.call_count == 1


def test_extract_json():
    """Test that the scraper can pull nested JSON out of raw YouTube HTML."""
    scraper = YoutubeScraper(logger=logging.getLogger("dummy"))

    html = """
    <html><body>
        <script>var ytInitialPlayerResponse = {"status": "OK", "is_awesome": true};</script>
    </body></html>
    """
    soup = BeautifulSoup(html, "html.parser")

    res = scraper._extract_json(soup, "ytInitialPlayerResponse")
    assert res == {"status": "OK", "is_awesome": True}

def test_save_and_load_progress(tmp_path):
    """Test writing scraper progress to disk and reading it back."""
    scraper = YoutubeScraper(output_dir=str(tmp_path), logger=logging.getLogger("dummy"))

    dummy_data = [{"video_id": "v1", "chapters": 5}, {"video_id": "v2", "chapters": 2}]

    scraper.save_progress(dummy_data, "test_prog.txt")

    loaded_data = scraper.load_progress("test_prog.txt")

    assert len(loaded_data) == 2
    assert loaded_data[0]["video_id"] == "v1"

def test_extract_cards():
    scraper = YoutubeScraper(logger=logging.getLogger("dummy"))

    valid_data = {
        "cards": {"cardCollectionRenderer": {"cards": [
            {"cardRenderer": {"startCardActiveMs": "1000",
            "teaser": {"simpleCardTeaserRenderer":
            {"message": {"simpleText": "Check this out"}}}}}
        ]}}
    }
    res_valid = scraper._extract_cards(valid_data)
    assert res_valid["card_count"] == 1
    assert res_valid["card_items"][0]["start_ms"] == "1000"

    res_invalid = scraper._extract_cards({})
    assert res_invalid["card_count"] == 0

def test_extract_verified():
    scraper = YoutubeScraper(logger=logging.getLogger("dummy"))

    # Path 1: Valid Data
    valid_data = {
        "contents": {"twoColumnWatchNextResults": {"results": {"results": {"contents": [
            {},
            {
            "videoSecondaryInfoRenderer":
            {
            "owner": {"videoOwnerRenderer":
            {"badges": [
            {"metadataBadgeRenderer":
            {"accessibilityData": {"label": "Verified"},
             "style": "BADGE_STYLE_TYPE_VERIFIED",
             "icon": {"iconType": "CHECK_CIRCLE_THICK"}}}
            ]}}}}
        ]}}}}
    }
    res_valid = scraper._extract_verified(valid_data)
    assert res_valid["is_verified"] is True
    assert "Verified" in res_valid["badge_labels"]

    res_invalid = scraper._extract_verified({})
    assert res_invalid["is_verified"] is False

def test_extract_comments_disabled():
    scraper = YoutubeScraper(logger=logging.getLogger("dummy"))

    valid_data = {
        "contents": {"twoColumnWatchNextResults": {"results": {"results": {"contents": [
            {"itemSectionRenderer": {"contents": [{"messageRenderer": {}}]}}
        ]}}}}
    }
    assert scraper._extract_comments_disabled(valid_data) is True

    assert scraper._extract_comments_disabled({}) is False


@patch.object(YoutubeScraper, "load_progress")
@patch.object(YoutubeScraper, "_scrape_one_safe")
@patch.object(YoutubeScraper, "export_data")
def test_scrape_videos_threading(mock_export, mock_scrape_one, mock_load, mocker):
    """Test the ThreadPoolExecutor and batch saving logic."""
    mock_load.return_value = [{"video_id": "vid_done", "data": 1}]

    mock_scrape_one.return_value = {"video_id": "vid_new", "data": 2}

    scraper = YoutubeScraper(logger=logging.getLogger("dummy"))

    scraper.SAVE_INTERVAL = 1

    mocker.patch.object(scraper, "save_progress")

    video_ids = ["vid_done", "vid_new", "vid_fail"]

    def side_effect(vid):
        if vid == "vid_fail":
            return None
        return {"video_id": vid, "data": 2}
    mock_scrape_one.side_effect = side_effect

    df = scraper.scrape_videos(video_ids)


    assert len(df) == 2
    assert "vid_done" in df["video_id"].values
    assert "vid_new" in df["video_id"].values
    assert "vid_fail" not in df["video_id"].values

    mock_export.assert_called_once()

def test_scraper_chapter_parsing():

    scraper = YoutubeScraper(logger=logging.getLogger("dummy"))

    assert scraper._timestamp_to_seconds("1:30") == 90
    assert scraper._timestamp_to_seconds("10:05") == 605

    dummy_player_data = {
        "videoDetails": {
            "shortDescription": "Welcome to the video!\n0:00 Intro\n1:30 Middle Part\n10:05 Outro\nThanks for watching."
        }
    }
    chapters = scraper._chapters_from_description(dummy_player_data)

    assert len(chapters) == 3
    assert chapters[0]["title"] == "Intro"
    assert chapters[0]["start_seconds"] == 0
    assert chapters[1]["title"] == "Middle Part"
    assert chapters[1]["start_seconds"] == 90
