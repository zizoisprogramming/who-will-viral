import pandas as pd
from unittest.mock import patch
from who_will_viral.feature_engineering.feature_scaling import FeatureScaling

def test_feature_scaling_run(tmp_path, mocker):
    """Test that FeatureScaling scales numeric columns but ignores protected ones."""
    mocker.patch("os.getenv", side_effect=lambda k, d=None: str(tmp_path / f"{k}.csv"))

    train_in = tmp_path / "train_in.csv"
    val_in = tmp_path / "val_in.csv"
    test_in = tmp_path / "test_in.csv"

    df = pd.DataFrame({
        "view_count": [10, 100, 1000],  
        "is_trending": [0, 1, 0],       
        "lang_base": [1, 2, 3],         
        "pca_0": [0.1, 0.2, 0.3]        
    })
    
    df.to_csv(train_in, index=False)
    df.to_csv(val_in, index=False)
    df.to_csv(test_in, index=False)

    scaler = FeatureScaling(str(train_in), str(val_in), str(test_in))
    scaler.run()

    out_train = pd.read_csv(tmp_path / "SCALED_TRAIN_PATH.csv")
    
    assert "view_count" in out_train.columns
    assert out_train["is_trending"].tolist() == [0, 1, 0]
    assert out_train["view_count"].iloc[0] != 10