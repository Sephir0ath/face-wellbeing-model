from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


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
    def __init__(self, arguments: bool | None = None):
        self.arguments = arguments

    def _build_pipeline(self, model_instance, mode_step=None) -> Pipeline:
        """Se construye sklearn Pipeline.

        mode_step: (name, transformer) como ("select", SelectKBest(...))
                  o ("pca", PCA(...)). 
                  
        """

        # NOTE: ahora mismo para todos los modelos de ML clásicos se está aplicando imputación y escalado.
        steps: list[tuple[str, object]] = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]

        if mode_step is not None:
            name, transformer = mode_step
            steps.append((name, clone(transformer)))

        steps.append(("clf", model_instance))
        return Pipeline(steps=steps)

    def fit_and_predict_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        models: list = MODELS,
        mode_step=None,
    ) -> dict:
        if not self.arguments:
            print("Model training and predictions... \n\n")

        predict: dict[str, np.ndarray] = {}
        for model in models:
            name = model.__name__
            tuning = self.parameter_tuning(name)
            model_instance = model(**tuning)

            pipeline = self._build_pipeline(model_instance=model_instance, mode_step=mode_step)
            pipeline.fit(X_train, y_train)
            predict[name] = pipeline.predict(X_test)

        return predict

    def fit_and_predict_single_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        model,
        mode_step=None,
    ) -> dict:
        name = model.__name__
        tuning = self.parameter_tuning(name)
        model_instance = model(**tuning)

        pipeline = self._build_pipeline(model_instance=model_instance, mode_step=mode_step)
        pipeline.fit(X_train, y_train)
        return {name: pipeline.predict(X_test)}

    def f1_scores_macro(self, y_predict: dict, y_test: pd.Series) -> dict:
        # Keep this name for compatibility; we report macro-F1.
        return {
            model_name: f1_score(y_test, y_pred, average="macro", zero_division=0)
            for model_name, y_pred in y_predict.items()
        }

    def f1_scores_binary(self, y_predict: dict, y_test: pd.Series) -> dict:
        return {
            model_name: f1_score(y_test, y_pred, average="binary", zero_division=0)
            for model_name, y_pred in y_predict.items()
        }

    def parameter_tuning(self, name: str) -> dict:
        if name == "LogisticRegression":
            return {
                "C": 1.0,
                "max_iter": 5000,
                "class_weight": "balanced",
                "random_state": 42,
            }
        if name == "KNeighborsClassifier":
            return {"n_neighbors": 5, "weights": "distance"}
        if name == "DecisionTreeClassifier":
            return {"max_depth": 6, "min_samples_split": 10, "random_state": 42}
        if name == "RandomForestClassifier":
            return {
                "n_estimators": 200,
                "max_depth": 8,
                "class_weight": "balanced",
                "random_state": 42,
            }
        if name == "SVC":
            return {
                "C": 1.0,
                "kernel": "rbf",
                "class_weight": "balanced",
                "probability": True,
                "random_state": 42,
            }
        if name == "LinearSVC":
            return {
                "C": 0.5,
                "class_weight": "balanced",
                "max_iter": 5000,
                "random_state": 42,
            }
        if name == "GaussianNB":
            return {"var_smoothing": 1e-9}
        if name == "GradientBoostingClassifier":
            return {
                "n_estimators": 150,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42,
            }
        if name == "HistGradientBoostingClassifier":
            return {
                "max_iter": 5000,
                "learning_rate": 0.1,
                "max_depth": None,
                "random_state": 42,
            }

        raise ValueError("Model not recognized for parameter tuning.")

    # -----------------------------
    # Deep learning (temporality=True)
    # -----------------------------
    def fit_and_predict_deep_learning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model,
    ) -> tuple[dict, np.ndarray]:
        """Train an LSTM on sliding windows and return predictions + aligned y_test."""

        # Local imports: tensorflow is optional for most workflows
        from tensorflow.keras.initializers import GlorotUniform
        from tensorflow.keras.layers import Dense, LSTM
        from tensorflow.keras.models import Sequential

        y_predict: dict[str, np.ndarray] = {}
        time_steps = 10

        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, time_steps)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test, time_steps)

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train_seq),
            y=y_train_seq,
        )
        class_weight = {0: class_weights[0], 1: class_weights[1]}

        net = Sequential()
        name = "LSTM"
        net.add(
            LSTM(
                64,
                activation="tanh",
                return_sequences=False,
                input_shape=(time_steps, X_train_seq.shape[2]),
                kernel_initializer=GlorotUniform(seed=42),
            )
        )
        net.add(Dense(32, activation="relu", kernel_initializer=GlorotUniform(seed=42)))
        net.add(Dense(1, activation="sigmoid", kernel_initializer=GlorotUniform(seed=42)))
        net.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        net.fit(
            X_train_seq,
            y_train_seq,
            epochs=10,
            batch_size=32,
            validation_split=0.3,
            class_weight=class_weight,
            verbose=0,
        )

        y_predict_base = net.predict(X_test_seq, verbose=0)
        y_predict[name] = (y_predict_base > 0.5).astype(int)
        return y_predict, y_test_seq

    def create_sequences(
        self, X: pd.DataFrame, y: pd.Series, time_steps: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        X_seq: list[np.ndarray] = []
        y_seq: list[int] = []

        for i in range(time_steps, len(X)):
            X_seq.append(X[i - time_steps : i].values)
            y_seq.append(int(y[i]))

        return np.array(X_seq), np.array(y_seq)
