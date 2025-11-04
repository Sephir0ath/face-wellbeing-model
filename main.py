from src.data_extractor import DataExtractor
from src.models import Models
from sklearn.model_selection import train_test_split
from src.menu import *

import pandas as pd


def split_data(dataframe: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.Series]:
    y = dataframe[label]
    X = dataframe.drop(columns=label)
    return X, y


def get_ID(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return dataframe.drop(columns=["ID"]), dataframe["ID"]


def main():
    # Extract csv data for temporality, question and feature
    extractor = DataExtractor()

    question = select_question()
    temporality = select_temporality()
    feature = select_feature(temporary=temporality)
    label = select_label()

    dataframe, labels_name = extractor.extract_csv(
        temporality=temporality, question=question, feature=feature, labels=label
    )

    # Create train and test
    X, y = split_data(dataframe=dataframe, label=labels_name)
    X, X_id = get_ID(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create model
    model = Models()

    y_predict = model.fit_and_predict_models(
        X_train=X_train, y_train=y_train, X_test=X_test
    )

    f1_score_result = model.f1_scores_macro(y_predict, y_test)

    for f1_score in f1_score_result:
        print(f"F1_SCORE of {f1_score} model: {f1_score_result[f1_score]}\n")


if __name__ == "__main__":
    main()
