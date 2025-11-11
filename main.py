from src.data_extractor import DataExtractor
from src.models import Models
from src.feature_selection import *
from src.smote import *
from src.dimensionality_reduction import *
from src.structure import *
from src.menu import *
from src.graph import *

from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

import pandas as pd
import warnings
import argparse

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


def extract_data() -> tuple[pd.DataFrame, str | list]:
    extract = DataExtractor()

    question = select_question()
    temporality = select_temporality()
    feature = select_feature(temporary=temporality)
    label = select_label()

    return extract.extract_csv(
        temporality=temporality, question=question, feature=feature, labels=label
    )


def extract_data_with_arguments(
        question: int, temporality: bool, feature: str, label: str 
    ) -> tuple[pd.DataFrame, str | list]:
    
    extract = DataExtractor()

    return extract.extract_csv(
        temporality=temporality, question=question, feature=feature, labels=label
    )


def extract_dataframes_and_series(
    dataframe: pd.DataFrame, label_names: str | list
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X, y = split_data(dataframe=dataframe, label=label_names)
    X, X_id = get_ID(X)
    return X, X_id, y


def config_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    mode = select_mode()

    # Mode (Feature Selection, Dimensionality Reduction, Over Sampling)
    if mode != "SMOTE" and mode is not None:
        X, y = return_mode(mode, X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Over Sampling
    if mode == "SMOTE":
        X_train, y_train = return_mode(mode, X_train, y_train)

    return X_train, X_test, y_train, y_test


def config_data_with_arguments(
    X: pd.DataFrame, y: pd.Series, mode
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    # Mode (Feature Selection, Dimensionality Reduction, Over Sampling)
    if mode != "SMOTE" and mode is not None:
        X, y = return_mode(mode, X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Over Sampling
    if mode == "SMOTE":
        X_train, y_train = return_mode(mode, X_train, y_train)

    return X_train, X_test, y_train, y_test


def execute_model(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> tuple[pd.Series, dict]:
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

    return y_predict, f1_score_result


def execute_model_with_arguments(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> tuple[pd.Series, dict]:
    # Create model
    model = Models(arguments=True)
    model_selected = ""

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

    return y_predict, f1_score_result


def sorted_dict(f1_scores: dict) -> dict:
    balance = {
        k: v
        for k, v in sorted(f1_scores.items(), key=lambda item: item[1], reverse=True)
    }
    return balance


def testing_models():
    # Extract csv data for temporality, question and feature
    dataframe, label_names = extract_data()

    # Split X, X_id and y
    X, X_id, y = extract_dataframes_and_series(
        dataframe=dataframe, label_names=label_names
    )

    # Split train data and test data
    X_train, X_test, y_train, y_test = config_data(X=X, y=y)

    # Execute model
    y_predict, f1_score_results = execute_model(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )

    # Sort dictionary
    f1_score_results = sorted_dict(f1_scores=f1_score_results)

    # Check the F1 SCORE for each model
    for f1_score in f1_score_results:
        print(f"F1_SCORE of {f1_score} model: {f1_score_results[f1_score]}\n")

    input("Press [ENTER] to view the confusion matrix... ")

    for model in y_predict:
        view_confusion_matrix(
            y_pred=y_predict[model], y_test=y_test, label=label_names, model_name=model
        )


def testing_models_with_arguments(args):
    
    feature = "" if args.feature == "All features" else args.feature
    temporality = True if args.temporality == "True" else False
    question = int(args.question)
    label = args.label
    mode = args.mode

    # Extract csv data for temporality, question and feature
    dataframe, label_names = extract_data_with_arguments(feature=feature, temporality=temporality, question=question, label=label)

    # Split X, X_id and y
    X, X_id, y = extract_dataframes_and_series(
        dataframe=dataframe, label_names=label_names
    )

    # Split train data and test data
    X_train, X_test, y_train, y_test = config_data_with_arguments(X=X, y=y, mode=mode)

    # Execute model
    y_predict, f1_score_results = execute_model_with_arguments(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )

    # Sort dictionary
    f1_score_results = sorted_dict(f1_scores=f1_score_results)

    # Check the F1 SCORE for each model
    for f1_score in f1_score_results:
        print(f"F1_SCORE of {f1_score} model: {f1_score_results[f1_score]}")


def main():

    parser = argparse.ArgumentParser()

    # Parser for questions 
    parser.add_argument("--question", type=int, help="Number of question. Example: 1, 2, 3, 4 or 5")
    # Parser for features
    parser.add_argument("--feature", type=str, help="Feature")
    # Parser for label
    parser.add_argument("--label", type=str, help="Name of labels")
    # Parser for temporality
    parser.add_argument("--temporality", type=str, help="Data with temporality or not")
    # Parser for models
    parser.add_argument("--model", type=str, help="Models")
    # Parser for mode
    parser.add_argument("--mode", type=str, help="hi")

    # Read to arguments
    args = parser.parse_args()

    if any(vars(args).values()):
        testing_models_with_arguments(args)
    else:
        testing_models()

        while True:

            response = input(
                "Do you want to continue testing models? [Y]es | [N]o: "
            ).lower()

            if response == "yes" or response == "y":
                testing_models()
            else:
                break


if __name__ == "__main__":
    main()
