from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from who_will_viral.validate import DataValidator, quick_summary, summarize_all


def test_validate_schema():
    validator = DataValidator()
    df = pd.DataFrame({"view_count": [100, 200], "title": ["A", "B"]})

    res_pass = validator.validate_schema(df, ["view_count", "title"], {"view_count": "int64"})
    assert res_pass["passed"] is True

    res_fail = validator.validate_schema(df, ["view_count", "title", "missing_col"], {})
    assert res_fail["passed"] is False

def test_validate_cross_column_rules():
    validator = DataValidator()
    df = pd.DataFrame({
        "view_count": [10, 100, 1000],
        "likes": [100, 50, 500],
        "comments_disabled": [False, True, False],
        "comment_count": [5, 10, 50]
    })

    res = validator.validate_cross_column_rules(df)
    assert res["passed"] is False
    assert len(res["issues"]) == 2

def test_validate_no_blank_strings():
    validator = DataValidator()
    df = pd.DataFrame({
        "title": ["Good Title", "   ", "Another"], # Row 1 is blank
        "description": ["Text", "Text", "Text"]
    })

    res = validator.validate_no_blank_strings(df, ["title", "description"])
    assert res["passed"] is False
    assert "blank-string" in res["issues"][0]

def test_validate_outliers_zscore():
    validator = DataValidator()
    df = pd.DataFrame({"view_count": [1, 2, 1, 2, 1000, 1, 2]})

    res = validator.validate_outliers_zscore(df, ["view_count"], threshold=2.0)
    assert res["passed"] is False

def test_validate_class_imbalance():
    validator = DataValidator()
    df = pd.DataFrame({"is_trending": [0]*9 + [1]})

    res_fail = validator.validate_class_imbalance(df, "is_trending", threshold=0.80)
    assert res_fail["passed"] is False

    res_pass = validator.validate_class_imbalance(df, "is_trending", threshold=0.95)
    assert res_pass["passed"] is True

def test_validate_date_order():
    validator = DataValidator()

    df = pd.DataFrame({
        "publishedAt": ["2024-01-01T00:00:00Z", "2024-01-05T00:00:00Z"],
        "trending_date": ["2024-01-02", "2024-01-03"]
    })

    res = validator.validate_date_order(df, "publishedAt", "trending_date")
    assert res["passed"] is False
    assert len(res["issues"]) == 1

def test_validate_count_matches_list():
    validator = DataValidator()

    df = pd.DataFrame({
        "chapter_count": [2, 1],
        "chapters": ["[{'t': 1}, {'t': 2}]", "[]"]
    })

    res = validator.validate_count_matches_list(df, "chapter_count", "chapters")
    assert res["passed"] is False

def test_validate_category_dominance():
    validator = DataValidator()
    df = pd.DataFrame({"categoryId": [1]*9 + [2]})

    res_fail = validator.validate_category_dominance(df, "categoryId", max_share=0.8)
    assert res_fail["passed"] is False

    res_pass = validator.validate_category_dominance(df, "categoryId", max_share=0.95)
    assert res_pass["passed"] is True

def test_quick_summary(capsys):
    """Test that the quick_summary function prints the expected dataset stats."""

    df = pd.DataFrame({
        "is_trending": [1, 0, 1],
        "view_count": [1000, 2000, np.nan],
        "title": ["A", "B", "C"]
    })

    quick_summary(df)

    captured = capsys.readouterr()

    assert "DATASET QUICK SUMMARY" in captured.out
    assert "Rows    : 3" in captured.out
    assert "Trending Distribution:" in captured.out
    assert "view_count" in captured.out # Should appear in missing values

def test_summarize_all(capsys):
    """Test the massive final scorecard printer."""
    mock_gx_results = MagicMock()
    mock_gx_results.results = [
        MagicMock(success=True),
        MagicMock(success=False)
    ]

    mock_pd_summary = {
        "total": 10, "passed": 8, "failed": 2, "success_rate": 80.0,
        "details": [{"dimension": "Accuracy", "passed": True}]
    }

    summarize_all(mock_gx_results, mock_pd_summary)

    captured = capsys.readouterr()
    assert "FINAL VALIDATION SCORECARD" in captured.out
    assert "Great Expectations" in captured.out
    assert "Per-Dimension Breakdown" in captured.out


def test_validate_statistics():

    validator = DataValidator()

    df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [2, 4, 6, 8, 10],
        "C": [1, 1, 1, 1, 100],
        "D": [5, 5, 5, 5, 5]
    })

    res_corr = validator.validate_correlation(df, ["A", "B"], corr_threshold=0.9)
    assert res_corr["passed"] is False

    res_skew = validator.validate_skew(df, ["C"], skew_threshold=1.0)
    assert res_skew["passed"] is False

    res_zero = validator.validate_non_zero_variance(df, ["D"])
    assert res_zero["passed"] is False
