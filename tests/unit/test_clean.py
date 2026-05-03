import numpy as np
import pandas as pd
import pytest

from who_will_viral.clean import (
    CleaningPipeline,
    DecisionLog,
    apply_log_transformation,
    cap_outliers,
    cast_types,
    drop_columns,
    drop_nulls,
    filter_invalid_rows,
    fix_comment_count,
    fix_description,
    normalize_tags,
    process_tags,
    remove_duplicates,
)


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


class TestDecisionLog:
    def test_record_adds_entry(self):
        log = DecisionLog()
        log.initial_shape = (10, 5)
        log.record('Step A', 'Some rule', 3, 'Drop', 'reason')
        assert len(log.entries) == 1
        e = log.entries[0]
        assert e.step == 'Step A'
        assert e.records_affected == 3

    def test_summary_with_no_shape(self, capsys):
        """summary() should not crash when initial_shape is not set."""
        log = DecisionLog()
        log.record('S', 'R', 0, 'A', 'rat')
        log.summary()  # should not raise
        out = capsys.readouterr().out
        assert 'CLEANING DECISION LOG' in out

    def test_summary_shows_shape_diff(self, capsys):
        log = DecisionLog()
        log.initial_shape = (100, 10)
        log.final_shape = (80, 10)
        log.record('S', 'R', 20, 'Drop', 'test')
        log.summary()
        out = capsys.readouterr().out
        assert '100' in out
        assert '80' in out


def test_remove_duplicates(sample_raw_df):
    """Test that duplicate rows and duplicate video_ids are dropped."""
    df_clean = remove_duplicates(sample_raw_df)
    assert len(df_clean) == 3
    assert df_clean['video_id'].value_counts().max() == 1


def test_filter_invalid_rows(sample_raw_df):
    """Test that videos with likes > views or invalid comment counts are dropped."""
    df_clean = filter_invalid_rows(sample_raw_df)

    valid_vids = df_clean['video_id'].tolist()
    assert 'vid2' not in valid_vids
    assert 'vid3' not in valid_vids
    assert 'vid1' in valid_vids


def test_process_tags():
    """Test that tags are correctly parsed into lists."""
    assert process_tags("['apple', 'banana']") == ["'apple'", "'banana'"]
    assert process_tags('apple, orange') == ['apple', 'orange']
    assert process_tags(np.nan) == []
    assert process_tags(['already', 'a', 'list']) == ['already', 'a', 'list']


def test_fix_description(sample_raw_df):
    """Test that null descriptions become empty strings."""
    df_clean = fix_description(sample_raw_df)
    assert df_clean.loc[1, 'description'] == ''
    assert isinstance(df_clean.loc[1, 'description'], str)


def test_apply_log_transformation():
    df = pd.DataFrame({'view_count': [0, np.expm1(1), np.expm1(2)]})
    res = apply_log_transformation(df, columns=['view_count'])

    assert np.isclose(res['view_count'].iloc[0], 0.0)
    assert np.isclose(res['view_count'].iloc[1], 1.0)
    assert np.isclose(res['view_count'].iloc[2], 2.0)


def test_cap_outliers():
    # Using Z-score capping
    df = pd.DataFrame({'view_count': [-1000, 1, 2, 3, 4, 1000]})
    res = cap_outliers(df, columns=['view_count'], method='zscore', z_threshold=1)

    # 1000 and -1000 should be capped to the bounds
    assert res['view_count'].max() < 1000
    assert res['view_count'].min() > -1000


def test_drop_nulls():
    """Test dropping rows with any NaNs."""
    df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [1, 2, 3]})
    res = drop_nulls(df)
    assert len(res) == 2


def test_cast_types():
    """Test converting string numbers/booleans to actual types."""
    df = pd.DataFrame({'view_count': ['10', '20', 'invalid'], 'embeddable': ['True', 'False', 'true']})
    res = cast_types(df)

    assert str(res['view_count'].dtype) == 'Int64'
    assert res['embeddable'].tolist() == [True, False, True]


def test_cleaning_pipeline_and_log(capsys):
    """Test the orchestrator loop and the decision log printing."""
    pipeline = CleaningPipeline()

    def dummy_step(df, log, **kwargs):
        log.record('Test Step', 'Rule 1', 1, 'Drop', 'Testing the logger')
        return df.iloc[:1]  # Simulates dropping one row

    pipeline.add_step('Dummy Step', dummy_step)

    df = pd.DataFrame({'A': [1, 2]})
    res = pipeline.fit_transform(df)

    assert len(res) == 1
    assert len(pipeline.log.entries) == 1

    pipeline.log.summary()
    captured = capsys.readouterr()

    assert 'CLEANING DECISION LOG' in captured.out
    assert 'Test Step' in captured.out
    assert 'Cleaned shape:' in captured.out


def test_fix_comment_count_and_drop():

    df1 = pd.DataFrame({'comment_count': [np.nan, np.nan, 5], 'comments_disabled': ['True', 'False', 'False']})
    res_comments = fix_comment_count(df1)

    assert res_comments['comment_count'].iloc[0] == 0
    assert pd.isna(res_comments['comment_count'].iloc[1])

    df2 = pd.DataFrame({'keep_me': [1], 'drop_me': [2], 'also_drop': [3]})
    res_drop = drop_columns(df2, columns=['drop_me', 'also_drop'])

    assert 'keep_me' in res_drop.columns
    assert 'drop_me' not in res_drop.columns


class TestProcessTags:
    def test_string_with_brackets(self):
        result = process_tags("['rock', 'pop']")
        assert "'rock'" in result
        assert "'pop'" in result

    def test_comma_separated(self):
        assert process_tags('a, b, c') == ['a', 'b', 'c']

    def test_nan_returns_empty(self):
        assert process_tags(np.nan) == []

    def test_already_list_passthrough(self):
        assert process_tags(['x', 'y']) == ['x', 'y']

    def test_empty_string_returns_empty(self):
        # An empty string after strip should yield []
        assert process_tags('') == []


class TestNormalizeTags:
    def test_applies_to_every_row(self, base_df):
        result = normalize_tags(base_df.copy())
        for val in result['tags']:
            assert isinstance(val, list)

    def test_logs_entry(self, base_df):
        log = DecisionLog()
        log.initial_shape = base_df.shape
        normalize_tags(base_df.copy(), log=log)
        assert len(log.entries) == 1


class TestRemoveDuplicates:
    def test_exact_duplicate_rows_removed(self):
        df = pd.DataFrame(
            {
                'video_id': ['a', 'a', 'b'],
                'value': [1, 1, 2],
            }
        )
        result = remove_duplicates(df)
        assert len(result) == 2

    def test_duplicate_video_id_removed(self):
        df = pd.DataFrame(
            {
                'video_id': ['a', 'a', 'b'],
                'value': [1, 9, 2],  # different rows, same video_id
            }
        )
        result = remove_duplicates(df)
        assert len(result) == 2

    def test_no_duplicates_unchanged(self):
        df = pd.DataFrame({'video_id': ['a', 'b', 'c'], 'v': [1, 2, 3]})
        assert len(remove_duplicates(df)) == 3


class TestFilterInvalidRows:
    def test_keeps_valid_rows(self, base_df):
        result = filter_invalid_rows(base_df)
        assert 'v1' in result['video_id'].tolist()

    def test_removes_likes_gt_views(self):
        df = pd.DataFrame(
            {
                'video_id': ['bad'],
                'likes': [999],
                'view_count': [1],
                'comment_count': [0],
                'comments_disabled': ['False'],
            }
        )
        result = filter_invalid_rows(df)
        assert len(result) == 0

    def test_removes_nonzero_comments_when_disabled(self):
        df = pd.DataFrame(
            {
                'video_id': ['bad'],
                'likes': [10],
                'view_count': [1000],
                'comment_count': [5],  # disabled but nonzero
                'comments_disabled': ['True'],
            }
        )
        result = filter_invalid_rows(df)
        assert len(result) == 0

    def test_allows_zero_comments_when_disabled(self):
        df = pd.DataFrame(
            {
                'video_id': ['ok'],
                'likes': [10],
                'view_count': [1000],
                'comment_count': [0],
                'comments_disabled': ['True'],
            }
        )
        result = filter_invalid_rows(df)
        assert len(result) == 1

    def test_allows_null_comment_count(self):
        df = pd.DataFrame(
            {
                'video_id': ['ok'],
                'likes': [10],
                'view_count': [1000],
                'comment_count': [np.nan],
                'comments_disabled': ['False'],
            }
        )
        result = filter_invalid_rows(df)
        assert len(result) == 1


class TestFixDescription:
    def test_null_becomes_empty_string(self, base_df):
        result = fix_description(base_df.copy())
        assert result.loc[1, 'description'] == ''

    def test_existing_values_preserved(self, base_df):
        result = fix_description(base_df.copy())
        assert result.loc[0, 'description'] == 'desc1'

    def test_logs_correctly(self, base_df):
        log = DecisionLog()
        log.initial_shape = base_df.shape
        fix_description(base_df.copy(), log=log)
        assert log.entries[0].records_affected == 1


class TestFixCommentCount:
    def test_fills_zero_when_disabled(self):
        df = pd.DataFrame(
            {
                'comment_count': [np.nan],
                'comments_disabled': ['True'],
            }
        )
        result = fix_comment_count(df)
        assert result['comment_count'].iloc[0] == 0

    def test_does_not_fill_when_not_disabled(self):
        df = pd.DataFrame(
            {
                'comment_count': [np.nan],
                'comments_disabled': ['False'],
            }
        )
        result = fix_comment_count(df)
        assert pd.isna(result['comment_count'].iloc[0])

    def test_does_not_overwrite_existing_value(self):
        df = pd.DataFrame(
            {
                'comment_count': [10.0],
                'comments_disabled': ['True'],
            }
        )
        result = fix_comment_count(df)
        assert result['comment_count'].iloc[0] == 10.0


class TestDropColumns:
    def test_drops_specified_columns(self):
        df = pd.DataFrame({'keep': [1], 'gone': [2], 'also_gone': [3]})
        result = drop_columns(df, columns=['gone', 'also_gone'])
        assert list(result.columns) == ['keep']

    def test_ignores_missing_columns(self):
        df = pd.DataFrame({'keep': [1]})
        result = drop_columns(df, columns=['nonexistent'])
        assert 'keep' in result.columns

    def test_logs_entry(self):
        log = DecisionLog()
        df = pd.DataFrame({'a': [1], 'b': [2]})
        drop_columns(df, log=log, columns=['b'])
        assert len(log.entries) == 1


class TestDropNulls:
    def test_removes_rows_with_any_null(self):
        df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
        result = drop_nulls(df)
        assert len(result) == 1

    def test_no_nulls_unchanged(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        assert len(drop_nulls(df)) == 2


class TestCastTypes:
    def test_int_columns_cast(self):
        df = pd.DataFrame({'view_count': ['100', '200', 'bad']})
        result = cast_types(df)
        assert str(result['view_count'].dtype) == 'Int64'

    def test_bool_columns_cast(self):
        df = pd.DataFrame({'embeddable': ['True', 'false', 'TRUE']})
        result = cast_types(df)
        assert result['embeddable'].tolist() == [True, False, True]

    def test_missing_columns_skipped(self):
        """cast_types should not crash when columns are absent."""
        df = pd.DataFrame({'some_other_col': [1, 2]})
        result = cast_types(df)  # should not raise
        assert 'some_other_col' in result.columns


class TestApplyLogTransformation:
    def test_zero_maps_to_zero(self):
        df = pd.DataFrame({'view_count': [0.0]})
        result = apply_log_transformation(df, columns=['view_count'])
        assert np.isclose(result['view_count'].iloc[0], 0.0)

    def test_negative_clipped_to_zero_then_log(self):
        df = pd.DataFrame({'view_count': [-5.0]})
        result = apply_log_transformation(df, columns=['view_count'])
        assert result['view_count'].iloc[0] == 0.0

    def test_log2_base(self):
        df = pd.DataFrame({'view_count': [1.0]})  # log2(2) = 1
        result = apply_log_transformation(df, columns=['view_count'], base='log2')
        assert np.isclose(result['view_count'].iloc[0], 1.0)

    def test_log10_base(self):
        df = pd.DataFrame({'view_count': [9.0]})  # log10(10) = 1
        result = apply_log_transformation(df, columns=['view_count'], base='log10')
        assert np.isclose(result['view_count'].iloc[0], 1.0)

    def test_skips_missing_column(self):
        df = pd.DataFrame({'other': [1.0]})
        result = apply_log_transformation(df, columns=['view_count'])
        assert 'view_count' not in result.columns


class TestCapOutliers:
    def test_iqr_method(self):
        df = pd.DataFrame({'view_count': [1, 2, 3, 4, 5, 1000]})
        result = cap_outliers(df, columns=['view_count'], method='iqr')
        assert result['view_count'].max() < 1000

    def test_zscore_method(self):
        df = pd.DataFrame({'view_count': [1, 2, 2, 2, 2, 9999]})
        result = cap_outliers(df, columns=['view_count'], method='zscore', z_threshold=2)
        assert result['view_count'].max() < 9999

    def test_chapter_count_capped_at_upper_bound(self):
        df = pd.DataFrame({'chapter_count': [0, 5, 100]})
        result = cap_outliers(df, columns=['chapter_count'], upper_bound=20)
        assert result['chapter_count'].max() <= 20

    def test_unknown_method_skips(self):
        df = pd.DataFrame({'view_count': [1, 2, 9999]})
        result = cap_outliers(df, columns=['view_count'], method='unknown_method')
        assert result['view_count'].max() == 9999  # untouched


class TestCleaningPipeline:
    def test_chaining_returns_self(self):
        pipeline = CleaningPipeline()
        result = pipeline.add_step('A', lambda df, log, **kw: df)
        assert result is pipeline

    def test_multiple_steps_execute_in_order(self):
        order = []

        def step_one(df, log, **kw):
            order.append(1)
            return df

        def step_two(df, log, **kw):
            order.append(2)
            return df

        pipeline = CleaningPipeline()
        pipeline.add_step('One', step_one).add_step('Two', step_two)
        pipeline.fit_transform(pd.DataFrame({'A': [1]}))
        assert order == [1, 2]

    def test_initial_and_final_shapes_recorded(self):
        pipeline = CleaningPipeline()
        pipeline.add_step('Drop one', lambda df, log, **kw: df.iloc[:-1])
        df = pd.DataFrame({'A': [1, 2, 3]})
        pipeline.fit_transform(df)
        assert pipeline.log.initial_shape == (3, 1)
        assert pipeline.log.final_shape == (2, 1)

    def test_kwargs_forwarded_to_step(self):
        received = {}

        def step(df, log, magic=None, **kw):
            received['magic'] = magic
            return df

        pipeline = CleaningPipeline()
        pipeline.add_step('Kwarg Step', step, magic=42)
        pipeline.fit_transform(pd.DataFrame({'A': [1]}))
        assert received['magic'] == 42
