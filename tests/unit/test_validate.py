from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from who_will_viral.validate import DataValidator, quick_summary, summarize_all


@pytest.fixture
def base_df():
	"""Minimal valid DataFrame used across many tests."""
	return pd.DataFrame(
		{
			'video_id': ['v1', 'v2', 'v3'],
			'view_count': [1000, 500, 2000],
			'likes': [100, 50, 200],
			'comment_count': [10, 5, 20],
			'comments_disabled': ['False', 'False', 'False'],
			'description': ['desc1', None, 'desc3'],
			'tags': ["['a','b']", 'x,y', np.nan],
			'is_trending': [1, 0, 1],
		}
	)


@pytest.fixture
def validator():
	return DataValidator()


def test_validate_schema():
	validator = DataValidator()
	df = pd.DataFrame({'view_count': [100, 200], 'title': ['A', 'B']})

	res_pass = validator.validate_schema(df, ['view_count', 'title'], {'view_count': 'int64'})
	assert res_pass['passed'] is True

	res_fail = validator.validate_schema(df, ['view_count', 'title', 'missing_col'], {})
	assert res_fail['passed'] is False


def test_validate_cross_column_rules():
	validator = DataValidator()
	df = pd.DataFrame(
		{
			'view_count': [10, 100, 1000],
			'likes': [100, 50, 500],
			'comments_disabled': [False, True, False],
			'comment_count': [5, 10, 50],
		}
	)

	res = validator.validate_cross_column_rules(df)
	assert res['passed'] is False
	assert len(res['issues']) == 2


def test_validate_no_blank_strings():
	validator = DataValidator()
	df = pd.DataFrame(
		{
			'title': ['Good Title', '   ', 'Another'],  # Row 1 is blank
			'description': ['Text', 'Text', 'Text'],
		}
	)

	res = validator.validate_no_blank_strings(df, ['title', 'description'])
	assert res['passed'] is False
	assert 'blank-string' in res['issues'][0]


def test_validate_outliers_zscore():
	validator = DataValidator()
	df = pd.DataFrame({'view_count': [1, 2, 1, 2, 1000, 1, 2]})

	res = validator.validate_outliers_zscore(df, ['view_count'], threshold=2.0)
	assert res['passed'] is False


def test_validate_class_imbalance():
	validator = DataValidator()
	df = pd.DataFrame({'is_trending': [0] * 9 + [1]})

	res_fail = validator.validate_class_imbalance(df, 'is_trending', threshold=0.80)
	assert res_fail['passed'] is False

	res_pass = validator.validate_class_imbalance(df, 'is_trending', threshold=0.95)
	assert res_pass['passed'] is True


def test_validate_date_order():
	validator = DataValidator()

	df = pd.DataFrame(
		{'publishedAt': ['2024-01-01T00:00:00Z', '2024-01-05T00:00:00Z'], 'trending_date': ['2024-01-02', '2024-01-03']}
	)

	res = validator.validate_date_order(df, 'publishedAt', 'trending_date')
	assert res['passed'] is False
	assert len(res['issues']) == 1


def test_validate_count_matches_list():
	validator = DataValidator()

	df = pd.DataFrame({'chapter_count': [2, 1], 'chapters': ["[{'t': 1}, {'t': 2}]", '[]']})

	res = validator.validate_count_matches_list(df, 'chapter_count', 'chapters')
	assert res['passed'] is False


def test_validate_category_dominance():
	validator = DataValidator()
	df = pd.DataFrame({'categoryId': [1] * 9 + [2]})

	res_fail = validator.validate_category_dominance(df, 'categoryId', max_share=0.8)
	assert res_fail['passed'] is False

	res_pass = validator.validate_category_dominance(df, 'categoryId', max_share=0.95)
	assert res_pass['passed'] is True


def test_quick_summary(capsys):
	"""Test that the quick_summary function prints the expected dataset stats."""

	df = pd.DataFrame({'is_trending': [1, 0, 1], 'view_count': [1000, 2000, np.nan], 'title': ['A', 'B', 'C']})

	quick_summary(df)

	captured = capsys.readouterr()

	assert 'DATASET QUICK SUMMARY' in captured.out
	assert 'Rows    : 3' in captured.out
	assert 'Trending Distribution:' in captured.out
	assert 'view_count' in captured.out  # Should appear in missing values


def test_summarize_all(capsys):
	"""Test the massive final scorecard printer."""
	mock_gx_results = MagicMock()
	mock_gx_results.results = [MagicMock(success=True), MagicMock(success=False)]

	mock_pd_summary = {
		'total': 10,
		'passed': 8,
		'failed': 2,
		'success_rate': 80.0,
		'details': [{'dimension': 'Accuracy', 'passed': True}],
	}

	summarize_all(mock_gx_results, mock_pd_summary)

	captured = capsys.readouterr()
	assert 'FINAL VALIDATION SCORECARD' in captured.out
	assert 'Great Expectations' in captured.out
	assert 'Per-Dimension Breakdown' in captured.out


def test_validate_statistics():

	validator = DataValidator()

	df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10], 'C': [1, 1, 1, 1, 100], 'D': [5, 5, 5, 5, 5]})

	res_corr = validator.validate_correlation(df, ['A', 'B'], corr_threshold=0.9)
	assert res_corr['passed'] is False

	res_skew = validator.validate_skew(df, ['C'], skew_threshold=1.0)
	assert res_skew['passed'] is False

	res_zero = validator.validate_non_zero_variance(df, ['D'])
	assert res_zero['passed'] is False


class TestValidateSchema:
	def test_passes_with_correct_columns_and_types(self, validator):
		df = pd.DataFrame({'view_count': pd.array([1, 2], dtype='int64'), 'title': ['A', 'B']})
		result = validator.validate_schema(df, ['view_count', 'title'], {'view_count': 'int64'})
		assert result['passed'] is True

	def test_fails_on_missing_column(self, validator):
		df = pd.DataFrame({'view_count': [1]})
		result = validator.validate_schema(df, ['view_count', 'missing'], {})
		assert result['passed'] is False
		assert any('Missing columns' in i for i in result['issues'])

	def test_fails_on_wrong_dtype(self, validator):
		df = pd.DataFrame({'view_count': ['str_value']})
		result = validator.validate_schema(df, ['view_count'], {'view_count': 'int64'})
		assert result['passed'] is False

	def test_extra_columns_noted_but_does_not_fail(self, validator):
		df = pd.DataFrame({'view_count': [1], 'extra': [2]})
		result = validator.validate_schema(df, ['view_count'], {})
		# extra columns should appear in issues as info, but not fail by themselves
		assert any('Extra columns' in i for i in result['issues'])


class TestValidateCrossColumnRules:
	def test_is_trending_without_trending_date_fails(self, validator):
		df = pd.DataFrame(
			{
				'is_trending': [1],
				'trending_date': [np.nan],
				'comments_disabled': [False],
				'comment_count': [0],
				'likes': [10],
				'view_count': [100],
			}
		)
		result = validator.validate_cross_column_rules(df)
		assert result['passed'] is False
		assert any('trending_date is null' in i for i in result['issues'])

	def test_passes_all_rules(self, validator):
		df = pd.DataFrame(
			{
				'is_trending': [1],
				'trending_date': ['2024-01-01'],
				'comments_disabled': [True],
				'comment_count': [0],
				'likes': [10],
				'view_count': [100],
			}
		)
		result = validator.validate_cross_column_rules(df)
		assert result['passed'] is True

	def test_age_restricted_with_comments_enabled_fails(self, validator):
		df = pd.DataFrame(
			{
				'contentDetails.contentRating.ytRating': ['ytAgeRestricted'],
				'comments_disabled': [False],
			}
		)
		result = validator.validate_cross_column_rules(df)
		assert result['passed'] is False


class TestValidateNoBlankStrings:
	def test_detects_whitespace_only(self, validator):
		df = pd.DataFrame({'title': ['Good', '   ', 'Also good']})
		result = validator.validate_no_blank_strings(df, ['title'])
		assert result['passed'] is False
		assert any('blank-string' in i for i in result['issues'])

	def test_passes_when_no_blanks(self, validator):
		df = pd.DataFrame({'title': ['A', 'B', 'C']})
		result = validator.validate_no_blank_strings(df, ['title'])
		assert result['passed'] is True

	def test_skips_missing_column(self, validator):
		df = pd.DataFrame({'other': ['A']})
		result = validator.validate_no_blank_strings(df, ['title'])
		assert result['passed'] is True  # column absent → nothing to flag


class TestValidateNoFutureDates:
	def test_future_date_fails(self, validator):
		df = pd.DataFrame({'publishedAt': ['2099-01-01T00:00:00Z']})
		result = validator.validate_no_future_dates(df, ['publishedAt'])
		assert result['passed'] is False

	def test_past_date_passes(self, validator):
		df = pd.DataFrame({'publishedAt': ['2020-01-01T00:00:00Z']})
		result = validator.validate_no_future_dates(df, ['publishedAt'])
		assert result['passed'] is True

	def test_skips_missing_column(self, validator):
		df = pd.DataFrame({'other': ['2020-01-01']})
		result = validator.validate_no_future_dates(df, ['nonexistent'])
		assert result['passed'] is True


class TestValidateDateOrder:
	def test_correct_order_passes(self, validator):
		df = pd.DataFrame(
			{
				'publishedAt': ['2024-01-01T00:00:00Z'],
				'trending_date': ['2024-01-05'],
			}
		)
		result = validator.validate_date_order(df, 'publishedAt', 'trending_date')
		assert result['passed'] is True

	def test_reversed_order_fails(self, validator):
		df = pd.DataFrame(
			{
				'publishedAt': ['2024-06-01T00:00:00Z'],
				'trending_date': ['2024-01-01'],
			}
		)
		result = validator.validate_date_order(df, 'publishedAt', 'trending_date')
		assert result['passed'] is False

	def test_missing_column_noted(self, validator):
		df = pd.DataFrame({'publishedAt': ['2024-01-01']})
		result = validator.validate_date_order(df, 'publishedAt', 'nonexistent')
		assert any('not found' in i for i in result['issues'])


class TestValidateOutliers:
	def test_iqr_flags_extremes(self, validator):
		df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 10000]})
		result = validator.validate_outliers_iqr(df, ['x'], multiplier=1.5)
		print(result)
		assert result['passed'] is False

	def test_iqr_zero_iqr_column_skipped(self, validator):
		"""All identical values → IQR=0, should be skipped gracefully."""
		df = pd.DataFrame({'x': [5, 5, 5, 5, 5]})
		result = validator.validate_outliers_iqr(df, ['x'])
		assert result['passed'] is True
		assert any('IQR=0' in i for i in result['info'])

	def test_zscore_passes_normal_data(self, validator):
		df = pd.DataFrame({'x': [1, 2, 3, 2, 1, 2, 3]})
		result = validator.validate_outliers_zscore(df, ['x'], threshold=3.0)
		assert result['passed'] is True

	def test_zscore_zero_std_column_skipped(self, validator):
		df = pd.DataFrame({'x': [7, 7, 7, 7]})
		result = validator.validate_outliers_zscore(df, ['x'], threshold=2.0)
		assert result['passed'] is True  # no crash, skipped


class TestValidateClassImbalance:
	def test_non_binary_values_fail(self, validator):
		df = pd.DataFrame({'is_trending': [0, 1, 2]})
		result = validator.validate_class_imbalance(df, 'is_trending')
		assert result['passed'] is False
		assert any('non-binary' in i for i in result['issues'])

	def test_balanced_passes(self, validator):
		df = pd.DataFrame({'is_trending': [0, 1] * 50})
		result = validator.validate_class_imbalance(df, 'is_trending', threshold=0.9)
		assert result['passed'] is True

	def test_missing_column_handled(self, validator):
		df = pd.DataFrame({'other': [1, 2]})
		result = validator.validate_class_imbalance(df, 'is_trending')
		assert any('not found' in i for i in result['issues'])


class TestValidateCategoryDominance:
	def test_dominant_category_fails(self, validator):
		df = pd.DataFrame({'cat': [1] * 95 + [2] * 5})
		result = validator.validate_category_dominance(df, 'cat', max_share=0.80)
		assert result['passed'] is False

	def test_spread_categories_pass(self, validator):
		df = pd.DataFrame({'cat': [1, 2, 3, 4] * 25})
		result = validator.validate_category_dominance(df, 'cat', max_share=0.80)
		assert result['passed'] is True

	def test_missing_column_handled(self, validator):
		df = pd.DataFrame({'other': [1]})
		result = validator.validate_category_dominance(df, 'missing_col')
		assert any('not found' in i for i in result['issues'])


class TestValidateCountMatchesList:
	def test_match_passes(self, validator):
		df = pd.DataFrame(
			{
				'chapter_count': [2],
				'chapters': ["[{'t': 1}, {'t': 2}]"],
			}
		)
		result = validator.validate_count_matches_list(df, 'chapter_count', 'chapters')
		assert result['passed'] is True

	def test_mismatch_fails(self, validator):
		df = pd.DataFrame(
			{
				'chapter_count': [3],
				'chapters': ["[{'t': 1}]"],
			}
		)
		result = validator.validate_count_matches_list(df, 'chapter_count', 'chapters')
		assert result['passed'] is False

	def test_empty_list_with_zero_count_passes(self, validator):
		df = pd.DataFrame(
			{
				'chapter_count': [0],
				'chapters': ['[]'],
			}
		)
		result = validator.validate_count_matches_list(df, 'chapter_count', 'chapters')
		assert result['passed'] is True

	def test_missing_columns_handled(self, validator):
		df = pd.DataFrame({'other': [1]})
		result = validator.validate_count_matches_list(df, 'chapter_count', 'chapters')
		assert any('not found' in i for i in result['issues'])


class TestValidateCorrelation:
	def test_high_correlation_fails(self, validator):
		x = list(range(10))
		df = pd.DataFrame({'A': x, 'B': x})
		result = validator.validate_correlation(df, ['A', 'B'], corr_threshold=0.9)
		assert result['passed'] is False

	def test_low_correlation_passes(self, validator):
		df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 1, 4, 2, 3]})
		result = validator.validate_correlation(df, ['A', 'B'], corr_threshold=0.9)
		assert result['passed'] is True

	def test_spearman_method(self, validator):
		x = list(range(10))
		df = pd.DataFrame({'A': x, 'B': x})
		result = validator.validate_correlation(df, ['A', 'B'], corr_threshold=0.9, method='spearman')
		assert result['passed'] is False

	def test_missing_column_skipped(self, validator):
		df = pd.DataFrame({'A': [1, 2, 3]})
		result = validator.validate_correlation(df, ['A', 'nonexistent'], corr_threshold=0.9)
		assert result['passed'] is True  # nothing to compare


class TestValidateSkew:
	def test_high_skew_fails(self, validator):
		df = pd.DataFrame({'x': [1] * 99 + [10000]})
		result = validator.validate_skew(df, ['x'], skew_threshold=1.0)
		assert result['passed'] is False

	def test_low_skew_passes(self, validator):
		df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
		result = validator.validate_skew(df, ['x'], skew_threshold=5.0)
		assert result['passed'] is True

	def test_missing_column_skipped(self, validator):
		df = pd.DataFrame({'other': [1, 2, 3]})
		result = validator.validate_skew(df, ['nonexistent'])
		assert result['passed'] is True


class TestValidateNonZeroVariance:
	def test_constant_column_fails(self, validator):
		df = pd.DataFrame({'x': [7, 7, 7, 7]})
		result = validator.validate_non_zero_variance(df, ['x'])
		assert result['passed'] is False

	def test_varied_column_passes(self, validator):
		df = pd.DataFrame({'x': [1, 2, 3, 4]})
		result = validator.validate_non_zero_variance(df, ['x'])
		assert result['passed'] is True


class TestGenerateReport:
	def test_report_counts_correctly(self, validator):
		df_pass = pd.DataFrame({'title': ['Good']})
		df_fail = pd.DataFrame({'title': ['   ']})

		validator.validate_no_blank_strings(df_pass, ['title'])  # PASS
		validator.validate_no_blank_strings(df_fail, ['title'])  # FAIL

		summary = validator.generate_report()
		assert summary['total'] == 2
		assert summary['passed'] == 1
		assert summary['failed'] == 1
		assert summary['success_rate'] == 50.0

	def test_report_prints_scorecard(self, validator, capsys):
		validator.validate_no_blank_strings(pd.DataFrame({'t': ['a']}), ['t'])
		validator.generate_report()
		out = capsys.readouterr().out
		assert 'PANDAS VALIDATION REPORT' in out
		assert 'Per-Dimension Breakdown' in out


class TestQuickSummary:
	def test_prints_expected_sections(self, capsys):
		df = pd.DataFrame(
			{
				'is_trending': [0, 1],
				'view_count': [100, 200],
			}
		)
		quick_summary(df)
		out = capsys.readouterr().out
		assert 'DATASET QUICK SUMMARY' in out
		assert 'Rows' in out
		assert 'Trending Distribution' in out

	def test_no_crash_without_is_trending(self, capsys):
		df = pd.DataFrame({'a': [1, 2, 3]})
		quick_summary(df)  # should not raise
		out = capsys.readouterr().out
		assert 'DATASET QUICK SUMMARY' in out

	def test_missing_values_section_when_no_nulls(self, capsys):
		df = pd.DataFrame({'a': [1, 2], 'is_trending': [0, 1]})
		quick_summary(df)
		out = capsys.readouterr().out
		assert 'None' in out or 'Missing' in out
