from __future__ import annotations

import argparse
import warnings
from typing import Any

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.data_extractor import DataExtractor
from src.graph import view_confusion_matrix
from src.menu import (
    select_feature,
    select_label,
    select_mode,
    select_model,
    select_question,
    select_temporality,
)
from src.models import MODELS, Models
from src.smote import Smote


# Hide warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def split_data(dataframe: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.Series]:
    y = dataframe[label]
    X = dataframe.drop(columns=label)
    return X, y


def get_ID(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return dataframe.drop(columns=["ID"]), dataframe["ID"]


def return_mode(mode: str) -> tuple[str, object] | None:
    """Return a (name, transformer) step for the sklearn Pipeline.

    IMPORTANT: do not fit/transform here to avoid data leakage.
    """
    if not mode:
        return None

    match mode:
        case "selection feature":
            return ("select", SelectKBest(score_func=f_classif, k=10))
        case "PCA":
            return ("pca", PCA(n_components=15, random_state=42))
        case "t-SNE":
            # t-SNE is for visualization only
            return None
        case "SMOTE":
            # SMOTE is applied only on the train split (per fold)
            return None
        case _:
            return None


def extract_data() -> tuple[pd.DataFrame, str | list, bool]:
    extractor = DataExtractor()

    question = select_question()
    temporality = select_temporality()
    feature = select_feature(temporary=temporality)
    label = select_label()

    df, label_names = extractor.extract_csv(
        temporality=temporality,
        question=question,
        feature=feature,
        labels=label,
    )
    return df, label_names, temporality


def extract_data_with_arguments(
    *, feature: str, temporality: bool, question: int, label: str
) -> tuple[pd.DataFrame, str | list]:
    extractor = DataExtractor()
    return extractor.extract_csv(
        temporality=temporality,
        question=question,
        feature=feature,
        labels=label,
    )


def extract_dataframes_and_series(
    dataframe: pd.DataFrame, label_names: str | list
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X, y = split_data(dataframe=dataframe, label=label_names)
    X, X_id = get_ID(X)
    return X, X_id, y


def config_data(
    X: pd.DataFrame, y: pd.Series, temporality: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Single hold-out split.

    Kept for backwards compatibility and temporality=True deep learning path.
    """
    _ = temporality
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y if getattr(y, "nunique", lambda: None)() and y.nunique() > 1 else None,
    )
    return X_train, X_test, y_train, y_test


def config_data_with_arguments(
    *, X: pd.DataFrame, y: pd.Series, mode: str | None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Single hold-out split used by the --args execution path."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y if getattr(y, "nunique", lambda: None)() and y.nunique() > 1 else None,
    )

    if mode == "SMOTE":
        apply = Smote()
        X_train, y_train = apply.apply(X_train=X_train, y_train=y_train)

    return X_train, X_test, y_train, y_test


def execute_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    mode_step: tuple[str, object] | None = None,
) -> tuple[dict, dict]:
    """Single hold-out training/eval for tabular models."""
    model = Models()
    model_selected = select_model()

    if not model_selected:
        y_predict = model.fit_and_predict_models(
            X_train=X_train, y_train=y_train, X_test=X_test, mode_step=mode_step
        )
    else:
        y_predict = model.fit_and_predict_single_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            model=model_selected,
            mode_step=mode_step,
        )

    f1_score_result = model.f1_scores_macro(y_predict, y_test)
    return y_predict, f1_score_result


def execute_deep_learning_model(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> tuple[dict, dict, np.ndarray]:
    """Deep learning path (temporality=True).

    NOTE: still uses a single hold-out split.
    """
    model = Models()
    model_selected = select_model(temporality=True)

    if not model_selected:
        y_predict = model.fit_and_predict_models(X_train=X_train, y_train=y_train, X_test=X_test)
        f1_score_result = model.f1_scores_macro(y_predict, y_test)
        y_test_seq = y_test.to_numpy()
        return y_predict, f1_score_result, y_test_seq

    y_predict, y_test_seq = model.fit_and_predict_deep_learning(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=model_selected,
    )
    f1_score_result = model.f1_scores_macro(y_predict, y_test_seq)
    return y_predict, f1_score_result, y_test_seq


def execute_model_with_arguments(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> tuple[dict, dict, dict]:
    """Args path helper (prints train/test macro F1).

    Kept for compatibility with the old CSV export format.
    """
    model = Models(arguments=True)
    model_selected = ""  # empty => evaluate all

    if not model_selected:
        y_predict_train = model.fit_and_predict_models(
            X_train=X_train, y_train=y_train, X_test=X_train
        )
        y_predict_test = model.fit_and_predict_models(
            X_train=X_train, y_train=y_train, X_test=X_test
        )
    else:
        y_predict_train = model.fit_and_predict_single_model(
            X_train=X_train, y_train=y_train, X_test=X_train, model=model_selected
        )
        y_predict_test = model.fit_and_predict_single_model(
            X_train=X_train, y_train=y_train, X_test=X_test, model=model_selected
        )

    f1_score_train = model.f1_scores_macro(y_predict_train, y_train)
    f1_score_test = model.f1_scores_macro(y_predict_test, y_test)
    return y_predict_test, f1_score_train, f1_score_test


def sorted_dict(f1_scores: dict) -> dict:
    return {
        k: v
        for k, v in sorted(f1_scores.items(), key=lambda item: item[1], reverse=True)
    }


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    mode: str | None,
    mode_step: tuple[str, object] | None,
    model_selected: Any,
    *,
    arguments: bool = False,
    tune: bool = False,
    n_iter: int = 25,
    inner_splits: int = 3,
) -> pd.DataFrame:
    """Stratified group cross-validation by participant ID."""

    unique_groups = groups.nunique()
    n_splits = min(5, unique_groups)
    if n_splits < 2:
        raise ValueError("Not enough unique participants for cross-validation")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows: list[dict[str, Any]] = []

    y_true_all: dict[str, list] = {}
    y_pred_all: dict[str, list] = {}

    for fold, (train_index, test_index) in enumerate(
        cv.split(X, y, groups=groups), start=1
    ):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        if mode == "SMOTE":
            apply = Smote()
            X_train, y_train = apply.apply(X_train=X_train, y_train=y_train)

        model = Models(arguments=arguments)

        if not tune:
            if not model_selected:
                preds = model.fit_and_predict_models(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    mode_step=mode_step,
                )
            else:
                preds = model.fit_and_predict_single_model(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    model=model_selected,
                    mode_step=mode_step,
                )
        else:
            # Nested CV: tune hyperparameters inside each outer fold.
            train_groups = groups.iloc[train_index]
            n_inner = min(inner_splits, train_groups.nunique())
            inner_cv = (
                StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=42)
                if n_inner >= 2
                else None
            )

            model_classes = MODELS if not model_selected else [model_selected]
            preds = {}
            for model_class in model_classes:
                pipe = model.build_pipeline_for(model_class, mode_step=mode_step)
                space = model.get_param_distributions(model_class.__name__, mode_step=mode_step)

                if inner_cv is None or not space:
                    # Fallback: train without tuning
                    pipe.fit(X_train, y_train)
                    preds[model_class.__name__] = pipe.predict(X_test)
                    continue

                search = RandomizedSearchCV(
                    estimator=pipe,
                    param_distributions=space,
                    n_iter=n_iter,
                    scoring="f1",  # binary F1
                    cv=inner_cv,
                    n_jobs=-1,
                    random_state=42,
                    refit=True,
                )
                search.fit(X_train, y_train, groups=train_groups)
                preds[model_class.__name__] = search.best_estimator_.predict(X_test)

        for model_name, y_pred in preds.items():
            rows.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                    "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                    "f1_binary": f1_score(
                        y_test, y_pred, average="binary", zero_division=0
                    ),
                }
            )

            y_true_all.setdefault(model_name, []).extend(list(y_test))
            y_pred_all.setdefault(model_name, []).extend(list(y_pred))

    results = pd.DataFrame(rows)
    results.attrs["y_true_all"] = y_true_all
    results.attrs["y_pred_all"] = y_pred_all
    return results


def testing_models() -> None:
    dataframe, label_names, temporality = extract_data()
    X, X_id, y = extract_dataframes_and_series(dataframe=dataframe, label_names=label_names)

    if temporality:
        X_train, X_test, y_train, y_test = config_data(X=X, y=y, temporality=temporality)
        y_predict, f1_score_results, y_test_seq = execute_deep_learning_model(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
        )

        f1_score_results = sorted_dict(f1_scores=f1_score_results)
        for model_name in f1_score_results:
            print(f"F1_SCORE of {model_name} model: {f1_score_results[model_name]}\n")

        input("Press [ENTER] to view the confusion matrix... ")
        for model_name in y_predict:
            view_confusion_matrix(
                y_pred=y_predict[model_name],
                y_test=y_test_seq,
                label=label_names,
                model_name=model_name,
            )
        return

    mode = select_mode()
    mode_step = return_mode(mode) if mode is not None and mode != "SMOTE" else None
    model_selected = select_model()

    cv_results = cross_validate_models(
        X=X,
        y=y,
        groups=X_id,
        mode=mode,
        mode_step=mode_step,
        model_selected=model_selected,
        arguments=False,
        tune=False,
    )

    summary = (
        cv_results.groupby("model")[
            ["accuracy", "balanced_accuracy", "f1_macro", "f1_binary"]
        ]
        .agg(["mean", "std"])
        .sort_values(("f1_binary", "mean"), ascending=False)
    )

    print("Cross-validation summary (mean/std):\n")
    print(summary)

    input("Press [ENTER] to view the confusion matrix... ")
    y_true_all = cv_results.attrs.get("y_true_all", {})
    y_pred_all = cv_results.attrs.get("y_pred_all", {})
    for model_name in y_pred_all:
        view_confusion_matrix(
            y_pred=pd.Series(y_pred_all[model_name]),
            y_test=pd.Series(y_true_all[model_name]),
            label=label_names,
            model_name=model_name,
        )


def testing_models_with_arguments(args: argparse.Namespace) -> None:
    feature = "" if args.feature == "All features" else args.feature
    temporality = True if args.temporality == "True" else False
    question = int(args.question)
    label = args.label
    mode = args.mode

    dataframe, label_names = extract_data_with_arguments(
        feature=feature,
        temporality=temporality,
        question=question,
        label=label,
    )

    X, X_id, y = extract_dataframes_and_series(dataframe=dataframe, label_names=label_names)

    mode_step = return_mode(mode) if mode is not None and mode != "SMOTE" else None
    model_selected = ""  # always evaluate all here

    cv_results = cross_validate_models(
        X=X,
        y=y,
        groups=X_id,
        mode=mode,
        mode_step=mode_step,
        model_selected=model_selected,
        arguments=True,
        tune=bool(getattr(args, "tune", False)),
        n_iter=int(getattr(args, "n_iter", 25)),
        inner_splits=int(getattr(args, "inner_splits", 3)),
    )

    summary = (
        cv_results.groupby("model")[["f1_macro", "f1_binary"]]
        .agg(["mean", "std"])
        .sort_values(("f1_binary", "mean"), ascending=False)
    )

    for model_name, row in summary.iterrows():
        print(
            f"{question},{temporality},{feature if feature else 'all'},{label},{mode},"
            f"{model_name},{row[('f1_binary', 'mean')]},{row[('f1_binary', 'std')]},"
            f"{row[('f1_macro', 'mean')]},{row[('f1_macro', 'std')]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=int, help="Question number (1-5)")
    parser.add_argument("--feature", type=str, help="Feature name")
    parser.add_argument("--label", type=str, help="Label name")
    parser.add_argument("--temporality", type=str, help="True/False")
    parser.add_argument("--model", type=str, help="Model")
    parser.add_argument("--mode", type=str, help="Mode")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable nested RandomizedSearchCV inside each outer fold",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=25,
        help="RandomizedSearchCV n_iter (only if --tune)",
    )
    parser.add_argument(
        "--inner_splits",
        type=int,
        default=3,
        help="Inner CV splits for tuning (only if --tune)",
    )
    args = parser.parse_args()

    if any(vars(args).values()):
        testing_models_with_arguments(args)
        return

    testing_models()
    while True:
        response = input("Do you want to continue testing models? [Y]es | [N]o: ").lower()
        if response in {"yes", "y"}:
            testing_models()
        else:
            break


if __name__ == "__main__":
    main()
