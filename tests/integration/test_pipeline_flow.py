from unittest.mock import patch

import numpy as np

from who_will_viral.clean import build_youtube_pipeline
from who_will_viral.feature_engineering.feature_extraction import FeatureExtraction


@patch("who_will_viral.feature_engineering.feature_extraction.SentenceTransformer")
def test_clean_to_features_handoff(mock_model, sample_raw_df, tmp_path, mocker):
    # Mock environment variables so it writes to a temporary test folder
    mocker.patch("os.getenv", side_effect=lambda k, d=None: str(tmp_path / "dummy.csv") if "PATH" in k else "dummy")
    mocker.patch("who_will_viral.clean.extract_hl_list_from_file", return_value={"en", "es"})

    # FIX: Return a NumPy array so .shape works
    mock_model_instance = mock_model.return_value
    mock_model_instance.encode.return_value = np.array([[0.1, 0.2]])

    cleaning_pipeline = build_youtube_pipeline(hl_file_path="dummy.json")
    cleaned_df = cleaning_pipeline.fit_transform(sample_raw_df)

    # Add dummy columns that FeatureExtraction expects
    cleaned_df["view_count"] = [1000]
    cleaned_df["card_count"] = [0]
    cleaned_df["chapter_count"] = [0]
    cleaned_df["publishedAt"] = ["2024-01-01T12:00:00Z"]

    extractor = FeatureExtraction()
    final_df = extractor.run(cleaned_df)

    assert "like_to_view_ratio" in final_df.columns
    assert "tags_joined_emb_0" in final_df.columns
