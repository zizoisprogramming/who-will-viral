import ast
import os
import re

import emoji
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


RATIOS = {
    "likes" : "like_to_view_ratio",
    "comment_count" : "comment_to_view_ratio"
}
EMBEDDINGS_COLS = ["tags_joined", "description", "title"]
EMOJIS_COLS = ["description", "title"]
class FeatureExtraction:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2", token=HF_TOKEN)

    # Functions used
    def _count_emojis(self, df):
        for col in EMOJIS_COLS:
            df[f"{col}_emoji_count"] = df[col].apply(
                lambda x: len(emoji.emoji_list(x)) if isinstance(x, str) else 0
            )
        return df

    def _get_duration_seconds(self, x):
        if not isinstance(x, str):
            return 0

        match = re.match(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$", x)
        if not match:
            return 0

        days    = int(match.group(1) or 0)
        hours   = int(match.group(2) or 0)
        minutes = int(match.group(3) or 0)
        seconds = int(match.group(4) or 0)

        return (days * 86400) + (hours * 3600) + (minutes * 60) + seconds

    def _parse_tags(self, val):
        if isinstance(val, list):
            return val
        try:
            return ast.literal_eval(val)
        except Exception:
            return []

    def _feature_interactions(self, df):
        for col, new_col in RATIOS.items():
            df[new_col] = df[col] / df["view_count"]
        return df

    def _apply_functions(self, df):
        df["tag_count"] = df["tags"].apply(lambda x: len(x) \
                        if isinstance(x, list) else len(x.split(",")) \
                        if isinstance(x, str) else 0)
        df["title_length"] = df["title"].apply(lambda x: len(x))
        df["description_length"] = df["description"].apply(lambda x: len(x) if isinstance(x, str) else 0)
        df["title_has_caps_ratio"] = df["title"].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x))
        df["tags"] = df["tags"].apply(self._parse_tags)
        df["tags_joined"] = df["tags"].apply(lambda tags: " ".join(tags))
        df['has_cards'] = (df['card_count'] > 0).astype(int)
        df['has_chapter'] = (df['chapter_count'] > 0).astype(int)
        return df

    def _time_features(self, df):
        df["publish_hour"] = pd.to_datetime(df["publishedAt"]).dt.hour
        df["publish_dayofweek"] = pd.to_datetime(df["publishedAt"]).dt.dayofweek
        df["duration_seconds"] = df["duration"].apply(self._get_duration_seconds)
        return df

    def _region_features(self, df):
        df['lang_base'] = df['defaultLanguage'].str.split('-').str[0]
        df['lang_region'] = df['defaultLanguage'].str.split('-').str[1]
        df['lang_region'] = df['lang_region'].fillna('NO')
        threshold = 0.002

        freq = df['lang_base'].value_counts(normalize=True)
        rare = freq[freq < threshold].index

        df['lang_base'] = df['lang_base'].replace(rare, 'other')

        threshold = 0.005

        freq = df['lang_region'].value_counts(normalize=True)
        rare = freq[freq < threshold].index

        df['lang_region'] = df['lang_region'].replace(rare, 'other')
        return df

    def _get_best_embeddings(self, df):
        if os.getenv("CI"):
            for col in EMBEDDINGS_COLS:
                for i in range(768):
                    df[f"{col}_emb_{i}"] = 0.0
            return df
        for col in EMBEDDINGS_COLS:
            for_embedding = df[col].replace("", "no text")
            embeddings = self.model.encode(for_embedding.tolist(), show_progress_bar=True, batch_size=512)
            embeddings_df = pd.DataFrame(embeddings, columns=[f"{col}_emb_{i}" for i in range(embeddings.shape[1])])
            df = pd.concat([df.reset_index(drop=True), embeddings_df], axis=1)
        return df


    def run(self, df):
        df = self._feature_interactions(df)
        df = self._apply_functions(df)
        df = self._time_features(df)
        df = self._count_emojis(df)
        df = self._region_features(df)
        df = self._get_best_embeddings(df)
        df.to_csv(os.getenv("EXTRACTED_PATH"), index=False)
        return df
