import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from ..data_extractor import DataExtractor

# -- LOAD DATASET FOR P3 --
data_extractor = DataExtractor()
p3_df = data_extractor.extract_csv(
    temporality=False,
    question=4,
    feature="", # empty string to get all features
    labels="depression"
)

print(data_extractor.validate_feature(feature="gaze", question=4, temporality=False))
print(data_extractor.validate_question(question=4))
print(p3_df.head())

p3_df.to_csv("p3_depression.csv", index=False)


