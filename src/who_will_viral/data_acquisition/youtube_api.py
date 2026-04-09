"""
YouTube Dataset Pipeline
========================
Fetches trending and non-trending YouTube video data, merges with an
existing CSV dataset, and exports a unified final CSV.

Usage
-----
    python youtube_pipeline.py \
        --csv data/youtube.csv \
        --output data/final_youtube_dataset.csv \
        [--api-key YOUR_KEY]   # or set YOUTUBE_KEY in .env

Environment
-----------
    YOUTUBE_KEY  – YouTube Data API v3 key (loaded from .env if present)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from googleapiclient.discovery import build


class RateLimiter:
    """Sliding-window rate limiter.

    Keeps track of request timestamps in the last ``time_window`` seconds
    and blocks (sleeps) when the quota is exhausted.

    Parameters
    ----------
    max_requests:
        Maximum number of requests allowed inside ``time_window``.
    time_window:
        Window length in seconds (default: 3 600 = 1 hour).
    """

    def __init__(self, max_requests: int = 10000, time_window: int = 3_600, logger = None) -> None:
        self.max_requests = max_requests
        self.time_window = time_window
        self._timestamps: list[float] = []
        self.logger = logger

    def wait_if_needed(self) -> None:
        """Block until a request slot is available."""
        now = time.monotonic()

        # Evict timestamps older than the window
        cutoff = now - self.time_window
        self._timestamps = [t for t in self._timestamps if t > cutoff]

        if len(self._timestamps) >= self.max_requests:
            sleep_secs = self.time_window - (now - self._timestamps[0])
            if sleep_secs > 0:
                self.logger.warning("Rate limit reached – sleeping %.1f s", sleep_secs)
                time.sleep(sleep_secs)
            self._timestamps = []

        self._timestamps.append(time.monotonic())



# Fields shared by the "video details" and "trending" endpoints
video_fields = (
    "items("
    "id,"
    "snippet(title,publishedAt,channelId,description,channelTitle,"
    "tags,defaultLanguage,categoryId,thumbnails),"
    "statistics(viewCount,favoriteCount,commentCount,likeCount),"
    "status(embeddable,madeForKids),"
    "contentDetails(duration,dimension,definition,caption,"
    "licensedContent,contentRating,projection,regionRestriction)"
    ")"
)

plain_video_fields = (
    "items("
    "id,"
    "snippet(defaultLanguage),"
    "statistics(favoriteCount),"
    "status(embeddable,madeForKids),"
    "contentDetails(duration,dimension,definition,caption,"
    "licensedContent,contentRating,projection,regionRestriction)"
    ")"
)

# Fields returned by the search endpoint (no statistics/contentDetails)
search_fields = (
    "items("
    "id,"
    "snippet(title,publishedAt,channelId,description,channelTitle,thumbnails)"
    ")"
)

# Column rename map applied to every DataFrame produced by this pipeline
rename_cols: dict[str, str] = {
    "id":           "video_id",
    "likeCount":    "likes",
    "viewCount":    "view_count",
    "commentCount": "comment_count",
}

GENRES: list[str] = [
    "cooking", "AI", "coding", "history", "gym", "speed", "vlog",
    "challenge", "reaction", "music", "gaming", "education", "comedy",
    "news", "sports", "beauty", "entertainment", "nonprofits",
    "howto", "science",
]


class YoutubeAPI:
    """Thin wrapper around the YouTube Data API v3.

    Parameters
    ----------
    api_key:
        Your YouTube Data API v3 key.
    rate_limiter:
        Optional :class:`RateLimiter` instance.  A default 100-req/hour
        limiter is created when *None* is passed.
    logger:
        Python logger to use.  Falls back to the module-level logger.
    """

    def __init__(
        self,
        api_key: str,
        base_csv: str ,
        logger: logging.Logger | None = None,
        backup_dir = None,
        today = None
    ) -> None:
        self.api_key = api_key
        self._rate = RateLimiter(logger=logger)
        self.logger = logger
        self._youtube = build("youtube", "v3", developerKey=self.api_key)
        self.genres = GENRES
        self.backup_dir = backup_dir
        self.base_csv = base_csv
        self.today = today

    def _execute(self, request: Any) -> dict:
        """Execute a request, honouring the rate limiter."""
        self._rate.wait_if_needed()
        return request.execute()


    def _enrich_base(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """Fetch additional fields for every unique video in the base CSV."""
        all_ids = base_df["video_id"].unique().tolist()
        self.logger.info("Enriching %d unique video IDs from base CSV", len(all_ids))

        backup = self.backup_dir + "/youtube_api_backup.json"
        if os.path.exists(backup):
            self.logger.info("Loading video details from backup file")
            items = self.load_json(backup)
        else:
            self.logger.info("Fetching video details for %d IDs", len(all_ids))
            items = self.get_video_details_batched(all_ids, full_details=False)
            self.save_json(items, backup)

        api_df = self.items_to_dataframe(items)

        api_df.rename(columns=rename_cols, inplace=True)

        merged = base_df.merge(api_df, on="video_id", how="left")
        self.logger.info("Enriched base DataFrame shape: %s", merged.shape)

        return merged

    def _fetch_trending(self) -> pd.DataFrame:
        backup = self.backup_dir + "/trending_videos_backup.json"
        if os.path.exists(backup):
            items = self.load_json(backup)
        else:
            items = self.get_trending_videos()
            self.save_json(items, backup)

        df = self.items_to_dataframe(items)
        self.logger.info("Adding trending metadata")
        df = self.add_pipeline_metadata(df, is_trending=True, today=self.today)
        df.rename(columns=rename_cols, inplace=True)
        self.logger.info("Trending DataFrame shape: %s", df.shape)
        return df

    def _fetch_non_trending(
        self,
        exclude_ids: set[str],
    ) -> pd.DataFrame:
        search_backup = self.backup_dir + "/non_trending_search_backup.json"
        detail_backup = self.backup_dir + "/non_trending_videos_backup.json"

        # --- search phase ---
        if os.path.exists(search_backup):
            search_items = self.load_json(search_backup)
        else:
            search_items = []
            for genre in GENRES:
                items = self.get_videos_by_genre(genre)
                search_items.extend(items)
                time.sleep(0.1)
            self.save_json(search_items, search_backup)

        all_ids = [
            item.get("id", {}).get("videoId")
            for item in search_items
        ]
        new_ids = list(dict.fromkeys(
            vid for vid in all_ids if vid and vid not in exclude_ids
        ))
        self.logger.info("%d new non-trending IDs after de-duplication", len(new_ids))

        # --- detail phase ---
        if os.path.exists(detail_backup):
            detail_items = self.load_json(detail_backup)
        else:
            detail_items = self.get_video_details_batched(new_ids)
            self.save_json(detail_items, detail_backup)

        df = self.items_to_dataframe(detail_items)
        df = self.add_pipeline_metadata(df, is_trending=False, today=self.today)
        df.rename(columns=rename_cols, inplace=True)
        self.logger.info("Non-trending DataFrame shape: %s", df.shape)
        return df


    def get_video_details(self, video_ids: list[str], full_details: bool = True) -> list[dict]:
        """Fetch full metadata for up to 50 video IDs in one call.

        Parameters
        ----------
        video_ids:
            List of YouTube video IDs (max 50 per call).

        Returns
        -------
        list[dict]
            Raw ``items`` from the API response.
        """
        if not video_ids:
            return []
        self.logger.info("Fetching %d video details", len(video_ids))
        request = self._youtube.videos().list(
            part="statistics,contentDetails,snippet,status",
            id=",".join(video_ids),
            fields=video_fields if full_details else plain_video_fields,
        )
        response = self._execute(request)
        return response.get("items", [])

    def get_video_details_batched(
        self,
        video_ids: list[str],
        batch_size: int = 50,
        delay: float = 0.05,
        full_details: bool = True,
    ) -> list[dict]:
        """Fetch details for an arbitrary number of video IDs in batches.

        Parameters
        ----------
        video_ids:
            All video IDs to fetch.
        batch_size:
            How many IDs to request per API call (max 50).
        delay:
            Seconds to wait between batches (on top of the rate limiter).

        Returns
        -------
        list[dict]
            Flattened list of all API ``items``.
        """
        results: list[dict] = []
        for i in range(0, len(video_ids), batch_size):
            batch = video_ids[i : i + batch_size]
            items = self.get_video_details(batch, full_details=full_details)
            results.extend(items)
            if delay:
                time.sleep(delay)
        self.logger.info("Fetched details for %d videos", len(results))
        return results

    def get_trending_videos(self, max_results: int = 50) -> list[dict]:
        """Return the current *mostPopular* chart.

        Note: the ``maxResults`` parameter is capped at 50 by the API.
        The notebook requested 1 000, which silently returns 50.

        Parameters
        ----------
        max_results:
            Number of results to request (1–50).
        """
        request = self._youtube.videos().list(
            part="statistics,contentDetails,snippet,status",
            chart="mostPopular",
            maxResults=min(max_results, 50),
            fields=video_fields,
        )
        response = self._execute(request)
        items = response.get("items", [])
        self.logger.info("Fetched %d trending videos", len(items))
        return items

    def get_videos_by_genre(
        self,
        genre: str,
        max_results: int = 50,
        published_after: str | None = None,
    ) -> list[dict]:
        """Search for recent videos matching *genre*.

        Parameters
        ----------
        genre:
            Free-text query (e.g. ``"cooking"``).
        max_results:
            Results per request (1–50).
        published_after:
            ISO-8601 datetime string, e.g. ``"2024-01-01T00:00:00Z"``.
            Defaults to 24 hours ago.
        """
        if published_after is None:
            published_after = (
                datetime.now(UTC) - timedelta(days=1)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")

        request = self._youtube.search().list(
            part="snippet",
            type="video",
            maxResults=min(max_results, 50),
            order="date",
            publishedAfter=published_after,
            q=genre,
            fields=search_fields,
        )
        response = self._execute(request)
        return response.get("items", [])

    def _flatten_column_names(self, columns: list[str]) -> list[str]:
        """Collapse ``'section.field'`` → ``'field'`` for two-level dotted names."""
        return [
            col.split(".")[1] if col.count(".") == 1 else col
            for col in columns
        ]


    def _extract_thumbnail_url(self, columns: list[str]) -> list[str]:
        """Rename the default-thumbnail URL column and drop other thumbnail cols."""
        new_cols = []
        for col in columns:
            if col == "snippet.thumbnails.default.url":
                new_cols.append("thumbnail_link")
            elif "thumbnails" in col:
                new_cols.append(None)   # mark for removal
            elif col.count(".") == 1:
                print(f"Warning: unexpected dotted column '{col}' – keeping full name")
                new_cols.append(col.split(".")[1])
            else:
                new_cols.append(col)
        return new_cols


    def items_to_dataframe(self, items: list[dict]) -> pd.DataFrame:
        """Normalise a list of raw API items into a tidy DataFrame.

        * Flattens nested JSON with :func:`pd.json_normalize`.
        * Renames dotted column names to their leaf name.
        * Extracts the default thumbnail URL as ``thumbnail_link``.
        * Drops all other thumbnail columns.

        Parameters
        ----------
        items:
            Raw list from any of the ``get_*`` methods.

        Returns
        -------
        pd.DataFrame
        """
        if not items:
            return pd.DataFrame()

        df = pd.json_normalize(items)

        # Resolve column names
        new_names = self._extract_thumbnail_url(df.columns.tolist())
        drop_mask = [name is None for name in new_names]
        drop_cols = [col for col, drop in zip(df.columns, drop_mask, strict=True) if drop]
        df.drop(columns=drop_cols, inplace=True)
        df.columns = [name for name in new_names if name is not None]
        print("DataFrame columns after thumbnail extraction:", df.columns.tolist())
        return df


    def add_pipeline_metadata(
        self,
        df: pd.DataFrame,
        is_trending: bool,
        today: str | None = None,
    ) -> pd.DataFrame:
        """Attach ``is_trending``, ``trending_date``, and ``comments_disabled``."""
        df = df.copy()
        today = today or pd.Timestamp.now().strftime("%Y-%m-%d")
        df["is_trending"]      = is_trending
        df["trending_date"]    = today
        # df["comments_disabled"] = np.nan
        return df


    def align_columns(self, df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
        """Add missing columns (as NaN) so *df* matches *reference*'s columns."""
        for col in reference.columns:
            if col not in df.columns:
                df[col] = np.nan
        return df[reference.columns]   # same column order


    def save_json(self, data: Any, path: str) -> None:
        os.makedirs('/'.join(path.split("/")[:-1]), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=4)
        self.logger.info("Saved JSON → %s", path)


    def load_json(self, path: str) -> Any:
        with open(path) as fh:
            data = json.load(fh)
        self.logger.info("Loaded JSON ← %s", path)
        return data

    def run(self, base_df):
        big_df = self._enrich_base(base_df)

        # 2. Trending
        trending_df = self._fetch_trending()
        trending_df = self.align_columns(trending_df, big_df)
        with open("trending.json", "w") as f:
            json.dump(trending_df.columns.tolist(), f, indent=4)

        # 3. Non-trending  (exclude IDs already in big_df + trending_df)
        known_ids = set(big_df["video_id"].dropna()) | set(trending_df["video_id"].dropna())
        non_trending_df = self._fetch_non_trending(exclude_ids=known_ids)
        non_trending_df = self.align_columns(non_trending_df, big_df)
        with open("non_trending.json", "w") as f:
            json.dump(non_trending_df.columns.tolist(), f, indent=4)


        # Sanity check – all DataFrames must have the same columns
        assert set(big_df.columns) == set(trending_df.columns) == set(non_trending_df.columns), (
            "Column mismatch before concat!"
        )

        # 4. Combine
        final_df = pd.concat([big_df, trending_df, non_trending_df], ignore_index=True)
        self.logger.info("Final DataFrame shape: %s", final_df.shape)

        return final_df
