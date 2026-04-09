import numpy as np
import pandas as pd
from who_will_viral.clean import (
    remove_duplicates,
    filter_invalid_rows,
    process_tags,
    fix_description,
    apply_log_transformation,
    cap_outliers,
    drop_nulls, 
    cast_types,
    fix_comment_count, 
    drop_columns,
    CleaningPipeline,
    DecisionLog
)

def test_remove_duplicates(sample_raw_df):
    """Test that duplicate rows and duplicate video_ids are dropped."""
    df_clean = remove_duplicates(sample_raw_df)
    assert len(df_clean) == 3
    assert df_clean["video_id"].value_counts().max() == 1

def test_filter_invalid_rows(sample_raw_df):
    """Test that videos with likes > views or invalid comment counts are dropped."""
    df_clean = filter_invalid_rows(sample_raw_df)
    
    valid_vids = df_clean["video_id"].tolist()
    assert "vid2" not in valid_vids
    assert "vid3" not in valid_vids
    assert "vid1" in valid_vids

def test_process_tags():
    """Test that tags are correctly parsed into lists."""
    assert process_tags("['apple', 'banana']") == ["'apple'", "'banana'"]
    assert process_tags("apple, orange") == ["apple", "orange"]
    assert process_tags(np.nan) == []
    assert process_tags(["already", "a", "list"]) == ["already", "a", "list"]

def test_fix_description(sample_raw_df):
    """Test that null descriptions become empty strings."""
    df_clean = fix_description(sample_raw_df)
    assert df_clean.loc[1, "description"] == ""
    assert isinstance(df_clean.loc[1, "description"], str)


def test_apply_log_transformation():
    df = pd.DataFrame({"view_count": [0, np.expm1(1), np.expm1(2)]})
    res = apply_log_transformation(df, columns=["view_count"])
    
    assert np.isclose(res["view_count"].iloc[0], 0.0)
    assert np.isclose(res["view_count"].iloc[1], 1.0)
    assert np.isclose(res["view_count"].iloc[2], 2.0)

def test_cap_outliers():
    # Using Z-score capping
    df = pd.DataFrame({"view_count": [-1000, 1, 2, 3, 4, 1000]})
    res = cap_outliers(df, columns=["view_count"], method="zscore", z_threshold=1)
    
    # 1000 and -1000 should be capped to the bounds
    assert res["view_count"].max() < 1000
    assert res["view_count"].min() > -1000


def test_drop_nulls():
    """Test dropping rows with any NaNs."""
    df = pd.DataFrame({"A": [1, np.nan, 3], "B": [1, 2, 3]})
    res = drop_nulls(df)
    assert len(res) == 2

def test_cast_types():
    """Test converting string numbers/booleans to actual types."""
    df = pd.DataFrame({
        "view_count": ["10", "20", "invalid"], 
        "embeddable": ["True", "False", "true"]
    })
    res = cast_types(df)
    
    assert str(res["view_count"].dtype) == "Int64"
    assert res["embeddable"].tolist() == [True, False, True]


def test_cleaning_pipeline_and_log(capsys):
    """Test the orchestrator loop and the decision log printing."""
    pipeline = CleaningPipeline()
    
    def dummy_step(df, log, **kwargs):
        log.record("Test Step", "Rule 1", 1, "Drop", "Testing the logger")
        return df.iloc[:1] # Simulates dropping one row
    
    pipeline.add_step("Dummy Step", dummy_step)
    
    df = pd.DataFrame({"A": [1, 2]})
    res = pipeline.fit_transform(df)
    
    assert len(res) == 1
    assert len(pipeline.log.entries) == 1
    
    pipeline.log.summary()
    captured = capsys.readouterr()
    
    assert "CLEANING DECISION LOG" in captured.out
    assert "Test Step" in captured.out
    assert "Cleaned shape:" in captured.out

def test_fix_comment_count_and_drop():

    df1 = pd.DataFrame({
        "comment_count": [np.nan, np.nan, 5],
        "comments_disabled": ["True", "False", "False"]
    })
    res_comments = fix_comment_count(df1)
    
    assert res_comments["comment_count"].iloc[0] == 0  
    assert pd.isna(res_comments["comment_count"].iloc[1]) 

    df2 = pd.DataFrame({"keep_me": [1], "drop_me": [2], "also_drop": [3]})
    res_drop = drop_columns(df2, columns=["drop_me", "also_drop"])
    
    assert "keep_me" in res_drop.columns
    assert "drop_me" not in res_drop.columns