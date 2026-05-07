from src.data_extractor import DataExtractor
from src.models import Models
from src.feature_selection import *
from src.smote import *
from src.structure import *
from src.menu import *
from src.graph import *

from sklearn.model_selection import GroupKFold, train_test_split, StratifiedGroupKFold
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import pandas as pd
import warnings
import questionary

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
    mode: str
) -> tuple[str, object] | None:
    if not mode:
        return None

    # NOTE: cada mode no se aplica directamente, sino que devuelve un (name, transformer) para introducir luego a pipeline (que se arma en models.py)
    match mode:
        case "selection feature":
            return ("select", SelectKBest(score_func=f_classif, k=10))
        case "PCA":
            return ("pca", PCA(n_components=20, random_state=42))
        case "t-SNE":
            # t-SNE solo para visualizacón, no se debería usar para entrenamiento
            return None
        case "SMOTE":
            # SMOTE se aplica solo en train (ver config_data)
            return None

    return None


def extract_data() -> tuple[pd.DataFrame, str | list]:
    extract = DataExtractor()

    question = select_question()
    temporality = select_temporality()
    feature = select_feature(temporary=temporality)
    label = select_label()

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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, tuple | None]:
    mode = select_mode()
    
    # split antes para evitar data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y if getattr(y, "nunique", lambda: None)() and y.nunique() > 1 else None,
    )

    # Mode (Feature Selection, Dimensionality Reduction)
    # no se aplica ninguna transformación aquí, solo se devuelve un step para construir sklearn.Pipeline en Models.
    mode_step = None
    if mode is not None and mode != "SMOTE":
        mode_step = return_mode(mode)

    # Over Sampling
    if mode == "SMOTE":
        apply = Smote()
        X_train, y_train = apply.apply(X_train=X_train, y_train=y_train)

    return X_train, X_test, y_train, y_test, mode_step


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    mode: str | None,
    mode_step: tuple[str, object] | None,
    model_selected,
) -> pd.DataFrame:
    """
        se ejecuta validación cruzada por grupos (GroupKFold) para evaluar modelos, asegurando que datos de un mismo participante 
        no se mezclen entre train y test.
    """

    unique_groups = groups.nunique()
    n_splits = min(5, unique_groups)
    if n_splits < 2:
        raise ValueError("Not enough unique participants for cross-validation")

    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows: list[dict] = []

    y_true_all: dict[str, list] = {}
    y_pred_all: dict[str, list] = {}

    for fold, (train_index, test_index) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        # Over Sampling only on train split
        if mode == "SMOTE":
            apply = Smote()
            X_train, y_train = apply.apply(X_train=X_train, y_train=y_train)

        model = Models()
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

        for model_name, y_pred in preds.items():
            rows.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                    "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                    "f1_binary": f1_score(y_test, y_pred, average="binary", zero_division=0),
                }
            )

            y_true_all.setdefault(model_name, []).extend(list(y_test))
            y_pred_all.setdefault(model_name, []).extend(list(y_pred))

    results = pd.DataFrame(rows)

    # Store for later use (confusion matrices across folds)
    results.attrs["y_true_all"] = y_true_all
    results.attrs["y_pred_all"] = y_pred_all
    return results


def execute_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    mode_step=None,
) -> tuple[pd.Series, dict]:
    # Create model
    model = Models()
    model_selected = select_model()

    # MultiModel or Single Model
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

    # Cross-validation configuration
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
    )

    summary = (
        cv_results.groupby("model")[["accuracy", "balanced_accuracy", "f1_macro", "f1_binary"]]
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


def main():
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
