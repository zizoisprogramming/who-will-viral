import pandas as pd
from sklearn.model_selection import train_test_split
from who_will_viral.feature_engineering.feature_extraction import FeatureExtraction
from who_will_viral.feature_engineering.feature_scaling import FeatureScaling
from who_will_viral.feature_engineering.feature_selection import FeatureSelection



class FeatureEngineering:
    def __init__(self):
        self.cleaned_path = "/Users/ziadsamer/Documents/who-will-viral/data/youtube/cleaned_dataset.csv"
        self.train_path = "/Users/ziadsamer/Documents/who-will-viral/data/youtube/train.csv"
        self.val_path = "/Users/ziadsamer/Documents/who-will-viral/data/youtube/val.csv"
        self.test_path = "/Users/ziadsamer/Documents/who-will-viral/data/youtube/test.csv"

        self.df = pd.read_csv(self.cleaned_path, keep_default_na=False)


    def run(self):

        feature_extraction = FeatureExtraction()
        extracted_df = feature_extraction.run(self.df)

        feature_selection = FeatureSelection(extracted_df, self.train_path, self.val_path, self.test_path)
        feature_selection.run()

        feature_scaling = FeatureScaling(self.train_path, self.val_path, self.test_path)
        feature_scaling.run()

print("Running feature engineering...")
FeatureEngineering().run()