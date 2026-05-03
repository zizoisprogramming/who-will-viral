import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.FileHandler('logs/cleaning.log'), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


YOUTUBE_EXTRA_LANGS = {
    'yue',
    'yue-hk',
    'bh',
    'bho',
    'mai',
    'sat',
    'bgc',
    'chr',
    'mni',
    'vro',
    'ase',
    'mo',
    'bi',
    'und',
    'zxx',
    'sdp',
}

COLUMNS_TO_DROP = [
    'thumbnail_link',
    'chapters',
    'cards',
    'badge_labels',
    'contentDetails.contentRating.ytRating',
    'contentDetails.regionRestriction.allowed',
    'contentDetails.regionRestriction.blocked',
    'trending_date',
    'favoriteCount',
]

INT_COLUMNS = ['view_count', 'likes', 'categoryId', 'comment_count', 'card_count', 'is_trending', 'chapter_count']
BOOL_COLUMNS = [
    'embeddable',
    'madeForKids',
    'supports_miniplayer',
    'is_verified',
    'has_paid_promotion',
    'comments_disabled',
]
LOG_COLUMNS = ['view_count', 'likes', 'comment_count']
CAP_COLUMNS = ['view_count', 'likes', 'comment_count', 'chapter_count']


@dataclass
class DecisionEntry:
    step: str
    rule: str
    records_affected: int
    action: str
    rationale: str


@dataclass
class DecisionLog:
    entries: list = field(default_factory=list)
    initial_shape: tuple = None
    final_shape: tuple = None

    def record(self, step, rule, records_affected, action, rationale):
        """Record a cleaning decision."""
        entry = DecisionEntry(step, rule, records_affected, action, rationale)
        self.entries.append(entry)
        logger.info(f'[{step}] {rule} -> {records_affected} rows | {action}')

    def summary(self):
        """Print a formatted decision log table."""
        total = sum(e.records_affected for e in self.entries)
        _ = (total / self.initial_shape[0] * 100) if self.initial_shape else 0

        print('\n' + '═' * 80)
        print('  CLEANING DECISION LOG')
        print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print('═' * 80)
        print(f'  {"Step":<20} {"Rule":<35} {"Affected":>9}  {"Action":<25} {"Rationale"}')
        print('─' * 80)
        for e in self.entries:
            print(f'  {e.step:<20} {e.rule:<35} {e.records_affected:>9,}  {e.action:<25} {e.rationale}')
        print('─' * 80)
        if self.initial_shape and self.final_shape:
            print(f'\n  Raw shape:     {self.initial_shape[0]:,} rows × {self.initial_shape[1]} cols')
            print(f'  Cleaned shape: {self.final_shape[0]:,} rows × {self.final_shape[1]} cols')
        print('═' * 80 + '\n')


class CleaningPipeline:
    """
    A reusable, modular data cleaning pipeline.
    Add steps with .add_step() and run with .fit_transform().
    Every step is logged automatically.
    """

    def __init__(self):
        self.steps: list[tuple[str, Callable, dict]] = []
        self.log = DecisionLog()

    def add_step(self, name: str, func: Callable, **kwargs):
        """Register a cleaning step. Returns self for chaining."""
        self.steps.append((name, func, kwargs))
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all registered steps on df and return cleaned result."""
        self.log.initial_shape = df.shape
        result = df.copy()

        for name, func, kwargs in self.steps:
            before = len(result)
            logger.info(f'Running step: {name}')
            result = func(result, log=self.log, **kwargs)
            after = len(result)
            if before != after:
                logger.info(f'  |-> Rows: {before:,} -> {after:,} (removed {before - after:,})')

        self.log.final_shape = result.shape
        return result


def extract_hl_list_from_file(file_path):
    """Load valid language codes from a YouTube hl_list JSON file."""
    with open(file_path, encoding='utf-8') as f:
        json_data = json.load(f)
    return {item['snippet']['hl'].split('-')[0].lower() for item in json_data.get('items', [])}


def process_tags(x):
    """Normalize tags to a list regardless of input type."""
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        x = x.strip().strip('[]')
        return [tag.strip() for tag in x.split(',') if tag.strip()]
    return []


def remove_duplicates(df, log: DecisionLog = None):
    """Remove duplicate rows and duplicate video IDs."""
    before = len(df)
    df = df.drop_duplicates().drop_duplicates(subset=['video_id'])
    affected = before - len(df)
    if log:
        log.record('Accuracy', 'Duplicate rows / video_id', affected, 'Drop', 'Exact duplicates or repeated video IDs')
    return df


def filter_invalid_rows(df, log: DecisionLog = None):
    """Remove rows with logically inconsistent values."""
    before = len(df)

    likes = pd.to_numeric(df['likes'], errors='coerce').fillna(0)
    view_count = pd.to_numeric(df['view_count'], errors='coerce').fillna(0)
    comment_count = pd.to_numeric(df['comment_count'], errors='coerce')
    comments_disabled = df['comments_disabled'].apply(lambda x: str(x).lower() == 'true')

    likes_mask = likes <= view_count
    df = df[likes_mask]

    # Re-compute after likes filter
    comment_count = pd.to_numeric(df['comment_count'], errors='coerce')
    comments_disabled = df['comments_disabled'].apply(lambda x: str(x).lower() == 'true')

    valid_comments = (
        comment_count.isna() | ((comment_count >= 0) & ~comments_disabled) | ((comment_count == 0) & comments_disabled)
    )
    df = df[valid_comments]

    affected = before - len(df)
    if log:
        log.record(
            'Consistency',
            'likes > view_count / invalid comment state',
            affected,
            'Reject -> drop',
            'Business rule violation',
        )
    return df


def drop_columns(df, log: DecisionLog = None, columns=None):
    """Drop irrelevant columns."""
    cols = [c for c in (columns or COLUMNS_TO_DROP) if c in df.columns]
    df = df.drop(columns=cols)
    if log:
        log.record(
            'Relevance', f'Drop {len(cols)} unused columns', 0, f'Dropped: {len(cols)} cols', 'Not needed for analysis'
        )
    return df


def normalize_tags(df, log: DecisionLog = None):
    """Normalize tags column to lists."""
    df['tags'] = df['tags'].apply(process_tags)
    if log:
        log.record(
            'Consistency', 'tags -> list normalization', len(df['tags']), 'Coerce to list', 'Standardize tag format'
        )
    return df


def fix_description(df, log: DecisionLog = None):
    """Fill null descriptions with empty string."""
    affected = df['description'].isna().sum()
    df['description'] = df['description'].fillna('').astype(str)
    if log:
        log.record(
            'Completeness',
            'description is null',
            int(affected),
            'Fill -> empty string',
            'Null description treated as no description',
        )
    return df


def fix_comment_count(df, log: DecisionLog = None):
    """Fill missing comment_count with 0 where comments are disabled."""
    mask = df['comment_count'].isna() & (df['comments_disabled'].apply(lambda x: str(x).lower()) == 'true')
    affected = int(mask.sum())
    df.loc[mask, 'comment_count'] = 0
    if log:
        log.record(
            'Completeness', 'comment_count null + disabled', affected, 'Fill -> 0', 'Disabled comments implies 0 count'
        )
    return df


def drop_nulls(df, log: DecisionLog = None):
    """Drop remaining rows with any null values."""
    before = len(df)
    df = df.dropna()
    affected = before - len(df)
    if log:
        log.record('Completeness', 'Remaining null values', affected, 'Drop rows', 'Cannot impute remaining nulls')
    return df


def cast_types(df, log: DecisionLog = None):
    """Cast columns to their correct data types."""
    for col in INT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    for col in BOOL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x).lower() == 'true')
    if log:
        log.record('Type Casting', 'Int + Bool columns', 0, 'Cast dtypes', 'Ensure correct types for analysis')
    return df


def clean_default_language(df, log: DecisionLog = None, hl_file_path='data/youtube/hl_list.json'):
    """Replace invalid defaultLanguage values with 'unknown'."""
    if 'defaultLanguage' not in df.columns:
        return df
    hl_set = extract_hl_list_from_file(hl_file_path) | YOUTUBE_EXTRA_LANGS
    series = df['defaultLanguage'].dropna()
    invalid_mask = ~series.str.split('-').str[0].str.lower().isin(hl_set)
    affected = int(invalid_mask.sum())
    df.loc[invalid_mask.index[invalid_mask], 'defaultLanguage'] = 'unknown'
    if log:
        log.record(
            'Consistency', 'defaultLanguage invalid code', affected, "Replace -> 'unknown'", 'Not in YouTube i18n list'
        )
    return df


def apply_log_transformation(df, log: DecisionLog = None, columns=None, base='natural'):
    """Apply log(x+1) transformation. Clips negatives to 0 first."""
    columns = columns or LOG_COLUMNS
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].clip(lower=0)
        if base == 'natural':
            df[col] = np.log1p(df[col])
        elif base == 'log2':
            df[col] = np.log2(df[col] + 1)
        elif base == 'log10':
            df[col] = np.log10(df[col] + 1)
    if log:
        log.record('Transformation', f'log({base}) on {columns}', len(df), 'log1p transform', 'Reduce skewness')
    return df


def cap_outliers(
    df, log: DecisionLog = None, columns=None, method='iqr', iqr_multiplier=1.5, z_threshold=3, upper_bound=20
):
    """Cap outliers using IQR or Z-score."""
    columns = columns or CAP_COLUMNS
    total_capped = 0
    for col in columns:
        if col not in df.columns:
            continue
        if col == 'chapter_count':
            lower_cap = 0
            upper_cap = upper_bound
        elif method == 'iqr':
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_cap = Q1 - iqr_multiplier * IQR
            upper_cap = Q3 + iqr_multiplier * IQR
        elif method == 'zscore':
            mean, std = df[col].mean(), df[col].std()
            lower_cap = mean - z_threshold * std
            upper_cap = mean + z_threshold * std
        else:
            continue
        n_capped = int(((df[col] < lower_cap) | (df[col] > upper_cap)).sum())
        total_capped += n_capped
        df[col] = df[col].clip(lower=lower_cap, upper=upper_cap)
    if log:
        log.record(
            'Outliers',
            f'Cap via {method} on {len(columns)} cols',
            total_capped,
            'Clip to bounds',
            f'Outlier treatment using {method}',
        )
    return df


def build_youtube_pipeline(hl_file_path='data/youtube/hl_list.json') -> CleaningPipeline:
    """Build and return the YouTube cleaning pipeline."""
    pipeline = CleaningPipeline()
    (
        pipeline.add_step('Remove Duplicates', remove_duplicates)
        .add_step('Filter Invalid Rows', filter_invalid_rows)
        .add_step('Drop Columns', drop_columns, columns=COLUMNS_TO_DROP)
        .add_step('Normalize Tags', normalize_tags)
        .add_step('Fix Description', fix_description)
        .add_step('Fix Comment Count', fix_comment_count)
        .add_step('Drop Nulls', drop_nulls)
        .add_step('Cast Types', cast_types)
        .add_step('Clean Language', clean_default_language, hl_file_path=hl_file_path)
        .add_step('Log Transformation', apply_log_transformation, columns=LOG_COLUMNS, base='natural')
        .add_step('Cap Outliers', cap_outliers, columns=CAP_COLUMNS, method='zscore')
    )
    return pipeline


if __name__ == '__main__':
    df = pd.read_csv(os.getenv('MERGED_PATH'))

    pipeline = build_youtube_pipeline()
    cleaned_df = pipeline.fit_transform(df)
    pipeline.log.summary()

    cleaned_df.to_csv(os.getenv('CLEANED_PATH'), index=False)
    print('Saved -> data/youtube/clean_dataset.csv')
