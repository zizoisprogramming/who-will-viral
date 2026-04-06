from urllib.robotparser import RobotFileParser
from requests.adapters import HTTPAdapter
from dotenv import load_dotenv
from urllib3 import Retry
import pandas as pd
import requests
import logging
import os

from pipelines.data_acquisition.youtube_scraper import YoutubeScraper
from pipelines.data_acquisition.youtube_database import YoutubeDatabase
from pipelines.data_acquisition.youtube_api import YoutubeAPI 


class YoutubePipeline:
    BASE_HEADERS = {"User-Agent": "Mozilla/5.0 (Educational Purpose) YoutubeScraper/1.0"}


    def __init__(self):
        success = load_dotenv()
        if not success:
            raise Exception("Failed to load .env file")
        api_key = os.getenv('YOUTUBE_API_KEY')
        print("API Key Loaded:", api_key)
        logging.basicConfig(
            filename='acquisition.log',
            filemode='a',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
        )
        self.logger = logging.getLogger(__name__)

        self.session = requests.Session()
        self.session.headers.update(self.BASE_HEADERS)
        self.base_csv = os.getenv("BASE_CSV", "./data/youtube/youtube.csv")
        self.output_csv = os.getenv("OUTPUT_CSV", "./data/youtube/youtube_enriched.csv")
        self.backup_dir = os.getenv("BACKUP_DIR", "./data/youtube/backup")
        print(self.base_csv, self.output_csv, self.backup_dir)

        retry_strategy = Retry(
            total=3, 
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504], 
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.today      = pd.Timestamp.now().strftime("%Y-%m-%d")
        self.youtube_api = YoutubeAPI(
            api_key = api_key, logger=self.logger,
            base_csv=self.base_csv, backup_dir=self.backup_dir,
            today=self.today
        )
        self.youtube_scraper = YoutubeScraper(logger=self.logger, session=self.session,output_dir=self.backup_dir)
        self.youtube_database = YoutubeDatabase(logger=self.logger, path=self.base_csv)

    def run(self) -> pd.DataFrame:
        """Execute the full pipeline and return the final DataFrame."""
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs("/".join(self.output_csv.split("/")[:-1]), exist_ok=True) 

        # 1. Base dataset
        base_df = self.youtube_database.run()

        # 2. Enrich with YouTube API
        final_df = self.youtube_api.run(base_df)  
        final_df.to_csv("api.csv", index=False)

        # 3. Scrape videos
        video_ids = final_df['video_id'].tolist()   
        scrapped_df = self.youtube_scraper.scrape_videos(video_ids)

        # 4. Merge all data
        final_df = final_df.merge(scrapped_df, on='video_id', how='left')
        self.logger.info("Merged scraped data. Final shape: %s", final_df.shape)

        # 5. Write to CSV
        final_df.to_csv(self.output_csv, index=False)

        return final_df
    

pipeline = YoutubePipeline()
pipeline.run()