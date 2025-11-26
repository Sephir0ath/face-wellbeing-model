from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np

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
            f1_score_result[y_val] = f1_score(
                y_test, y_predict[y_val], average="weighted"
            )
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

    def fit_and_predict_deep_learning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model,
    ) -> tuple[dict, np.array]:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.initializers import GlorotUniform

        y_predict = {}
        time_steps = 10

        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, time_steps)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test, time_steps)

        # In depression case is more importat 1 that 0, because exist imbalance
        print("Distribución en y_train:", y_train.value_counts(), end="\n")
        print("Distribución en y_test:", y_test.value_counts(), end="\n")
        print(
            "Distribución en y_train_seq:",
            np.unique(y_train_seq, return_counts=True),
            end="\n",
        )

        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_train_seq), y=y_train_seq
        )
        class_weight = {0: class_weights[0], 1: class_weights[1]}
        print("Class weights:", class_weight, end="\n")

        model = Sequential()
        name = "LSTM"

        model.add(
            LSTM(
                64,
                activation="tanh",
                return_sequences=False,
                input_shape=(time_steps, X_train_seq.shape[2]),
                kernel_initializer=GlorotUniform(seed=42),
            )
        )

        # binary classifier
        model.add(
            Dense(32, activation="relu", kernel_initializer=GlorotUniform(seed=42))
        )
        model.add(
            Dense(1, activation="sigmoid", kernel_initializer=GlorotUniform(seed=42))
        )

        # optimizer: adam is the most popular
        # loss: is the error measure, i use binary_crossentropy because we have binary classification
        # metrics: show numbers to understand performance
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        # epoch: how many times the models sees the full dataset. More epoch -> more learning
        model.fit(
            X_train_seq,
            y_train_seq,
            epochs=10,
            batch_size=32,
            validation_split=0.3,
            class_weight=class_weight,
        )

        y_predict_base = model.predict(X_test_seq)

        print("=== ANÁLISIS DE PREDICCIONES ===")
        print(f"Valores mínimos: {y_predict_base.min()}")
        print(f"Valores máximos: {y_predict_base.max()}")
        print(f"Media: {y_predict_base.mean()}")
        print(f"Distribución:")
        print(f"  < 0.1: {np.sum(y_predict_base < 0.1)}")
        print(f"0.1-0.5: {np.sum((y_predict_base >= 0.1) & (y_predict_base < 0.5))}")
        print(f"0.5-0.9: {np.sum((y_predict_base >= 0.5) & (y_predict_base < 0.9))}")
        print(f"  > 0.9: {np.sum(y_predict_base >= 0.9)}")

        # Ver las primeras 10 predicciones
        print("Primeras 10 predicciones raw:")
        print(y_predict_base[:10].flatten())

        y_predict[name] = (y_predict_base > 0.5).astype(int)

        return y_predict, y_test_seq

    def create_sequences(self, X: pd.DataFrame, y: pd.Series, time_steps=10):
        X_seq = []
        y_seq = []

        for i in range(time_steps, len(X)):
            X_seq.append(X[i - time_steps : i].values)
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)
