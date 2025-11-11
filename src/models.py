from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

MODELS = [
    LogisticRegression,
    KNeighborsClassifier,
    DecisionTreeClassifier,
    RandomForestClassifier,
    SVC,
    LinearSVC,
    GaussianNB,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
]


class Models:
    def __init__(self, arguments: bool = None):
        self.arguments = arguments

    def fit_and_predict_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        models: list = MODELS,
    ) -> dict:

        if not self.arguments:
            print("Model training and predictions... \n\n")
        predict = {}

        for model in models:
            name = model.__name__

            tuning = self.parameter_tuning(name)

            # Add model with tuned parameters
            model_instance = model(**tuning)

            model_instance.fit(X_train, y_train)
            predict[name] = model_instance.predict(X_test)

        return predict

    def f1_scores_macro(self, y_predict: dict, y_test: pd.Series) -> dict:
        f1_score_result = {}
        for y_val in y_predict:
            f1_score_result[y_val] = f1_score(y_test, y_predict[y_val], average="macro")
        return f1_score_result

    def parameter_tuning(self, name: str) -> dict:
        if name == "LogisticRegression":
            tuning = {
                "C": 1.0,
                "max_iter": 5000,
                "class_weight": "balanced",
                "random_state": 42,
            }

        elif name == "KNeighborsClassifier":
            tuning = {"n_neighbors": 5, "weights": "distance"}

        elif name == "DecisionTreeClassifier":
            tuning = {"max_depth": 6, "min_samples_split": 10, "random_state": 42}

        elif name == "RandomForestClassifier":
            tuning = {
                "n_estimators": 200,
                "max_depth": 8,
                "class_weight": "balanced",
                "random_state": 42,
            }

        elif name == "SVC":
            tuning = {
                "C": 1.0,
                "kernel": "rbf",
                "class_weight": "balanced",
                "probability": True,
                "random_state": 42,
            }

        elif name == "LinearSVC":
            tuning = {
                "C": 0.5,
                "class_weight": "balanced",
                "max_iter": 5000,
                "random_state": 42,
            }

        elif name == "GaussianNB":
            tuning = {"var_smoothing": 1e-9}

        elif name == "GradientBoostingClassifier":
            tuning = {
                "n_estimators": 150,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42,
            }

        elif name == "XGBClassifier":
            tuning = {
                "n_estimators": 200,
                "learning_rate": 0.1,
                "max_depth": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "eval_metric": "logloss",
            }
        elif name == "HistGradientBoostingClassifier":
            tuning = {
                "max_iter": 5000,
                "learning_rate": 0.1,
                "max_depth": None,
                "random_state": 42,
            }
        else:
            raise ValueError("Model not recognized for parameter tuning.")

        return tuning

    def fit_and_predict_single_model(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, model
    ) -> dict:
        y_predict = {}

        name = model.__name__
        tuning = self.parameter_tuning(name)

        model_instance = model(**tuning)
        model_instance.fit(X_train, y_train)
        y_predict[name] = model_instance.predict(X_test)

        return y_predict
