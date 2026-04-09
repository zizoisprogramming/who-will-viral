from googleapiclient.discovery import build
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import logging
import time
import json
import os
import tqdm


class YoutubeDatabase:
    def __init__(self, logger=None, path='youtube_data.csv'):
        self.logger = logger
        self.path = path

    def run(self) -> pd.DataFrame:
        self.logger.info("Loading base CSV: %s", self.path)
        df = pd.read_csv(self.path)
        self.logger.info("  shape: %s", df.shape)
        return df

