from src.data_extractor import DataExtractor
from src.models import Models
from src.feature_selection import *
from src.smote import *
from src.dimensionality_reduction import *
from src.structure import *
from src.menu import *

from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

import pandas as pd
import warnings

# Hide warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def split_data(dataframe: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.Series]:
    y = dataframe[label]
    X = dataframe.drop(columns=label)
    return X, y


def get_ID(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return dataframe.drop(columns=["ID"]), dataframe["ID"]


def return_mode(
    mode: str, X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    if not mode:
        return None

    match mode:
        case "selection feature":
            apply = FeatureSelection()
            X = apply.apply(X, y)
        case "PCA":
            apply = PCA()
            X = apply.apply(X)
        case "t-SNE":
            apply = t_SNE()
            X = apply.apply(X)
        case "SMOTE":
            apply = Smote()
            X, y = apply.apply(X_train=X, y_train=y)

    return X, y


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

    # Mode (Feature Selection, Dimensionality Reduction, Over Sampling)
    mode = select_mode()

    if mode != "SMOTE" and mode is not None:
        X, y = return_mode(mode, X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Over Sampling
    if mode == "SMOTE":
        X_train, y_train = return_mode(mode, X_train, y_train)

    # Create model
    model = Models()
    model_selected = select_model()

    # MultiModel or Single Model
    if not model_selected:
        y_predict = model.fit_and_predict_models(
            X_train=X_train, y_train=y_train, X_test=X_test
        )
    else:
        y_predict = model.fit_and_predict_single_model(
            X_train=X_train, y_train=y_train, X_test=X_test, model=model_selected
        )

    # F1 SCORE
    f1_score_result = model.f1_scores_macro(y_predict, y_test)

    # Check the F1 SCORE for each model
    for f1_score in f1_score_result:
        print(f"F1_SCORE of {f1_score} model: {f1_score_result[f1_score]}\n")


if __name__ == "__main__":
    main()
