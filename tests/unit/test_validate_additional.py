"""Additional tests for validate module to improve coverage."""

import json

import pandas as pd
import pytest

from src.who_will_viral.validate import extract_hl_list_from_file, normalize_lang, quick_summary


@pytest.fixture
def sample_hl_list_file(tmp_path):
    """Create a sample hl_list.json file."""
    data = {
        'items': [
            {'snippet': {'hl': 'en-US'}},
            {'snippet': {'hl': 'es-ES'}},
            {'snippet': {'hl': 'en-GB'}},
            {'snippet': {'hl': 'fr-FR'}},
        ]
    }
    file_path = tmp_path / 'hl_list.json'
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    return file_path


@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing."""
    return pd.DataFrame(
        {
            'video_id': ['vid1', 'vid2', 'vid3'],
            'title': ['Title 1', 'Title 2', 'Title 3'],
            'view_count': [1000, 2000, 3000],
            'likes': [100, 200, 300],
            'comment_count': [10, 20, 30],
            'favoriteCount': [5, 10, 15],
            'is_trending': [1, 0, 1],
            'publishedAt': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'channelId': ['ch1', 'ch2', 'ch3'],
            'categoryId': [10, 20, 30],
            'duration': ['PT10M', 'PT20M', 'PT30M'],
            'defaultLanguage': ['en-US', 'es-ES', 'fr-FR'],
        }
    )


def test_extract_hl_list_from_file(sample_hl_list_file):
    """Test extracting hl list from file."""
    result = extract_hl_list_from_file(str(sample_hl_list_file))

    assert 'en' in result
    assert 'es' in result
    assert 'fr' in result
    assert len(result) == 3


def test_extract_hl_list_from_file_empty(tmp_path):
    """Test extracting hl list from empty file."""
    data = {'items': []}
    file_path = tmp_path / 'empty_hl_list.json'
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    result = extract_hl_list_from_file(str(file_path))
    assert result == []


def test_normalize_lang_with_region():
    """Test normalizing language code with region."""
    result = normalize_lang('en-US')
    assert result == 'en'


def test_normalize_lang_already_normalized():
    """Test normalizing already normalized language code."""
    result = normalize_lang('en')
    assert result == 'en'


def test_normalize_lang_none():
    """Test normalizing None language code."""
    result = normalize_lang(None)
    assert result is None


def test_normalize_lang_empty_string():
    """Test normalizing empty string."""
    result = normalize_lang('')
    assert result == ''


def test_quick_summary_basic(sample_df, capsys):
    """Test quick_summary function with basic dataframe."""
    quick_summary(sample_df)
    captured = capsys.readouterr()

    assert 'DATASET QUICK SUMMARY' in captured.out
    assert 'Rows' in captured.out
    assert 'Columns' in captured.out
    assert '3' in captured.out  # 3 rows


def test_quick_summary_with_missing_values(capsys):
    """Test quick_summary with missing values."""
    df = pd.DataFrame({'col1': [1, 2, None], 'col2': [None, 'b', 'c'], 'col3': [10, 20, 30], 'is_trending': [0, 1, 0]})

    quick_summary(df)
    captured = capsys.readouterr()

    assert 'DATASET QUICK SUMMARY' in captured.out
    assert 'Missing Values' in captured.out


def test_quick_summary_without_is_trending(capsys):
    """Test quick_summary with dataframe without is_trending column."""
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

    quick_summary(df)
    captured = capsys.readouterr()

    assert 'DATASET QUICK SUMMARY' in captured.out
    assert 'Rows' in captured.out
