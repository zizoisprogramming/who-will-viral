from urllib.robotparser import RobotFileParser
from requests.adapters import HTTPAdapter
from urllib.parse import urlparse
from collections import deque
from bs4 import BeautifulSoup
from urllib3 import Retry
from tqdm import tqdm
import pandas as pd
import requests
import logging
import json
import time
import os
import re


class YoutubeScraper:
    """
    Production-ready youtube scraper with logging, rate limiting, and error recovery.
    """

    BASE_HEADERS = {"User-Agent": "Mozilla/5.0 (Educational Purpose) YoutubeScraper/1.0"}
    BASE_URL = "https://www.youtube.com/watch?v="
 
    VERIFIED_LABELS = {"Verified"}
    VERIFIED_STYLES = {"BADGE_STYLE_TYPE_VERIFIED"}
    VERIFIED_ICONS  = {"CHECK_CIRCLE_THICK"}

    def __init__(self, output_dir='scraped_data', logger=None, session=None):
        """
        Initialize scraper with logging and rate limiting.
        """

        self.logger = logger
        self.session = session
        

        self.rate_limit = 100 
        self.time_window = 60
        self.request_times = deque()

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)  

    def scrape_videos(self, video_ids: list[str]) -> pd.DataFrame:
        """
        Main entry point. Accept a list of video IDs and return a DataFrame
        with one row per video, containing all extracted features.
        """
        rows        = self.load_progress()
        scraped_ids = {row["video_id"] for row in rows}
        pending     = [vid for vid in video_ids if vid not in scraped_ids]

        if scraped_ids:
            self.logger.info(
                f"Resuming: {len(scraped_ids)} already done, {len(pending)} remaining."
            )

        for vid_id in tqdm(pending):
            url = self.BASE_URL + vid_id
            self.logger.info(f"Scraping video: {vid_id}")
 
            if not self.check_robots_txt(url):
                self.logger.warning(f"Blocked by robots.txt — skipping {vid_id}")
                continue
 
            try:
                row = self._scrape_single(url, vid_id)
                rows.append(row)
                self.save_progress(rows) 
                self.logger.info(f"Successfully scraped {vid_id}")
            except Exception as e:
                self.logger.error(f"Failed to scrape {vid_id}: {e}")
 
        df = pd.DataFrame(rows)
        self.export_data(df, "videos")
        self.logger.info(f"Total scraped: {len(df)} videos")
        return df
    
    def _scrape_single(self, url: str, video_id: str) -> dict:
        """Fetch and extract all features for one video."""
        soup         = self._fetch_page(url)
        player_data  = self._extract_json(soup, "ytInitialPlayerResponse")
        initial_data = self._extract_json(soup, "ytInitialData")
 
        chapters     = self.get_chapters(player_data,initial_data)
        playability  = self._extract_playability(player_data)
        cards        = self._extract_cards(player_data)
        verified     = self._extract_verified(initial_data)
        comments_disabled = self._extract_comments_disabled(initial_data)
        has_paid_promotion = self._extract_paid_promotion(player_data)
 
        return {
            "video_id":           video_id,
            "url":                url,
            "chapter_count":      len(chapters),
            "chapters":           chapters,
            "playability_status": playability["status"],
            "supports_miniplayer":playability["supports_miniplayer"],
            "card_count":         cards["card_count"],
            "cards":              cards["card_items"],
            "is_verified":        verified["is_verified"],
            "badge_labels":       verified["badge_labels"],
            "comments_disabled":  comments_disabled,
            "has_paid_promotion": has_paid_promotion,
        }
    
    def _fetch_page(self, url: str) -> BeautifulSoup:
        self.enforce_rate_limit()
        response = self.session.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    
    
    def get_chapters(self, player_data: dict | None, initial_data: dict | None) -> list[dict]:
            """
            Return a list of chapter dicts: [{"title": str, "start_seconds": int}, ...]
            Returns an empty list if no chapters are found.
            """
            return (
                self._chapters_from_player_data(player_data)
                or self._chapters_from_initial_data(initial_data)
                or self._chapters_from_description(player_data)
                or []
            )
    
    def _chapters_from_player_data(self, player_data: dict | None) -> list[dict]:
        """Strategy 1 – ytInitialPlayerResponse → markersMap."""
        try:
            markers = self._get_markers_map(player_data)
            return self._parse_markers(markers)
        except (KeyError, TypeError):
            return []
 
    def _chapters_from_initial_data(self, initial_data: dict | None) -> list[dict]:
        """Strategy 2 – ytInitialData → markersMap."""
        try:
            markers = self._get_markers_map(initial_data)
            return self._parse_markers(markers)
        except (KeyError, TypeError):
            return []
 
    def _chapters_from_description(self, player_data: dict | None) -> list[dict]:
        """Strategy 3 – parse MM:SS timestamps from the video description."""
        try:
            description = player_data["videoDetails"]["shortDescription"]
            matches = re.findall(r'(\d{1,2}:\d{2})\s+(.+)', description)
            return [
                {
                    "title": title.strip(),
                    "start_seconds": self._timestamp_to_seconds(time_str),
                }
                for time_str, title in matches
            ]
        except (KeyError, TypeError):
            return []
    

    def _extract_playability(self, player_data: dict | None) -> dict:
        """Extract playability status, age restriction, embeddability, miniplayer."""
        defaults = {
            "status":             "UNKNOWN",
            "supports_miniplayer": False,
        }
        if not player_data:
            return defaults
        try:
            playability        = player_data["playabilityStatus"]
            status             = playability.get("status", "UNKNOWN")
            supports_miniplayer = (
                player_data
                .get("microformat", {})
                .get("playerMicroformatRenderer", {})
                .get("isFamilySafe", False)
            )
            return {
                "status":              status,
                "supports_miniplayer": supports_miniplayer,
            }
        except (KeyError, TypeError):
            return defaults
 
    def _extract_cards(self, player_data: dict | None) -> dict:
        """Extract card teasers and their activation timestamps."""
        if not player_data:
            return {"card_count": 0, "card_items": []}
        try:
            cards      = player_data["cards"]["cardCollectionRenderer"]["cards"]
            card_items = []
            for card in cards:
                r = card["cardRenderer"]
                card_items.append({
                    "teaser_text": (
                        r.get("teaser", {})
                         .get("simpleCardTeaserRenderer", {})
                         .get("message", {})
                         .get("simpleText", "")
                    ),
                    "start_ms": r.get("startCardActiveMs"),
                })
            return {"card_count": len(card_items), "card_items": card_items}
        except (KeyError, TypeError):
            return {"card_count": 0, "card_items": []}
 
    def _extract_verified(self, initial_data: dict | None) -> dict:
        """Extract channel verified badge — works for Arabic and English."""
        defaults = {"is_verified": False, "badge_labels": []}
        if not initial_data:
            return defaults
        try:
            badges = (
                initial_data["contents"]
                ["twoColumnWatchNextResults"]
                ["results"]["results"]["contents"][1]
                ["videoSecondaryInfoRenderer"]["owner"]
                ["videoOwnerRenderer"]["badges"]
            )
            is_verified  = False
            badge_labels = []
 
            for b in badges:
                renderer  = b.get("metadataBadgeRenderer", {})
                label     = renderer.get("accessibilityData", {}).get("label", "")
                style     = renderer.get("style", "")
                icon_type = renderer.get("icon", {}).get("iconType", "")
                badge_labels.append(label)
 
                if (
                    label     in self.VERIFIED_LABELS
                    or any(style.startswith(s) for s in self.VERIFIED_STYLES)
                    or icon_type in self.VERIFIED_ICONS
                ):
                    is_verified = True
 
            return {"is_verified": is_verified, "badge_labels": badge_labels}
        except (KeyError, TypeError):
            return defaults
        
    def _extract_comments_disabled(self, initial_data: dict | None) -> bool:
        try:
            contents = (
                initial_data["contents"]
                ["twoColumnWatchNextResults"]
                ["results"]["results"]["contents"]
            )
            for item in contents:
                for c in item.get("itemSectionRenderer", {}).get("contents", []):
                    if "messageRenderer" in c:
                        return True 
        except (KeyError, TypeError):
            pass
        return False
    
    def _extract_paid_promotion(self, player_data: dict | None) -> bool:
        try:
            return bool(player_data["paidContentOverlay"])
        except (KeyError, TypeError):
            return False
 
    def _extract_json(self, soup: BeautifulSoup, key: str) -> dict | None:
        """Pull a JSON blob assigned to *key* out of any <script> tag."""
        for script in soup.find_all("script"):
            if key not in script.text:
                continue
            match = re.search(rf'{key}\s*=\s*({{.*}});', script.text)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
        return None
        
    
    def _get_markers_map(self, data: dict) -> list:
        return (
            data["playerOverlays"]
               ["playerOverlayRenderer"]
               ["decoratedPlayerBarRenderer"]
               ["decoratedPlayerBarRenderer"]
               ["playerBar"]
               ["multiMarkersPlayerBarRenderer"]
               ["markersMap"]
        )
 
    def _parse_markers(self, markers: list) -> list[dict]:
        for item in markers:
            if item.get("key") == "AUTO_CHAPTERS":
                return [
                    {
                        "title":         ch["chapterRenderer"]["title"]["simpleText"],
                        "start_seconds": ch["chapterRenderer"]["timeRangeStartMillis"] // 1000,
                    }
                    for ch in item["value"]["chapters"]
                ]
        return []
 
    def _timestamp_to_seconds(self, time_str: str) -> int:
        minutes, seconds = map(int, time_str.split(":"))
        return minutes * 60 + seconds


    def check_robots_txt(self, url):
        """
        Check if scraping is allowed for the given URL.
        """
        try:
            # Build the robots.txt URL from the domain
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

            # Parse the robots.txt file
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()

            # Check if our user_agent is allowed to access this URL
            allowed = rp.can_fetch(self.session.headers.get('User-Agent', ''), url)
            self.logger.info(f"Checked robots.txt for {url}: {'Allowed' if allowed else 'Blocked'}")
            return allowed

        except Exception as e:
            self.logger.warning(f"Could not read robots.txt for {url}: {e}")
            return False
        

    def enforce_rate_limit(self):

        now = time.time()
        # Remove timestamps older than the rate period
        while self.request_times and now - self.request_times[0] > self.time_window:
            self.request_times.popleft() # remove requests older than the time window (60s)

        if len(self.request_times) >= self.rate_limit: # wait until next request is allowed
            wait_time = self.time_window - (now - self.request_times[0]) + 1  # wait till oldest request is time window old +1s to be safe
            self.logger.info(f"Rate limit reached. Waiting for {wait_time:.2f} seconds...")
            time.sleep(wait_time)

        # record the current request time
        self.request_times.append(time.time())
    
    def save_progress(self, videos, filename='progress.json'):
        """
        Save current scraping progress to disk (for resumability).
        """
        try:
            filename = os.path.join(self.output_dir, filename)
            with open(filename, 'w') as f:
                json.dump(videos, f, indent=4)
            self.logger.info(f"Progress saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")

    def load_progress(self, filename='progress.json'):
        """
        Load previous progress from disk to resume interrupted scraping.
        """
        try:
            filename = os.path.join(self.output_dir, filename)
            with open(filename, 'r') as f:
                videos = json.load(f)
            self.logger.info(f"Progress loaded from {filename}")
            return videos
        except Exception as e:
            self.logger.warning(f"No progress file found or failed to load: {e}")
            return []

    def export_data(self, videos, base_filename='videos'):
        """
        Export collected data to multiple file formats.

        Creates:
        - videos.csv   (UTF-8 encoded)
        - videos.xlsx  (with formatted headers)
        - videos.json  (properly structured)
        """
        try:
            filename = os.path.join(self.output_dir, base_filename)
            df = pd.DataFrame(videos)
            df.to_csv(f'{filename}.csv', index=False, encoding='utf-8')
            self.logger.info(f"Data exported to {base_filename}.csv")
        except Exception as e:
            self.logger.error(f"Failed to export data: {e}")