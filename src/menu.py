from questionary import select

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


def select_question() -> int:
    questions = [str(i + 1) for i in range(5)]

    response = select("Select question:", choices=questions).ask()
    response = int(response)

    return response


def select_temporality() -> bool:

    response = select("Temporary data:", choices=["True", "False"]).ask()
    response = True if response == "True" else False

    return response


def select_feature(temporary: bool) -> str:

    if temporary:
        questions = [
            "gaze",
            "pose",
            "2d_landmarks",
            "3d_landmarks",
            "pdm",
            "AU",
            "eye_lmk",
            "All features",
        ]
    else:
        questions = ["gaze", "pose", "AU_c", "AU_r", "All features"]

    response = select("Select feature:", choices=questions).ask()

    response = "" if response == "All features" else response

    return response


def select_label() -> str:
    # Ignore both, both is a DataFrame, isn't Series.

    # questions = ["depression", "anxiety", "both (ignore this)"]
    questions = ["depression", "anxiety"]

    response = select("Select label:", choices=questions).ask()

    # response = "both" if response == "both (ignore this)" else response

    return response


def select_model():
    models = [
        "all",
        "LogisticRegression",
        "KNeighborsClassifier",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "SVC",
        "LinearSVC",
        "GaussianNB",
        "GradientBoostingClassifier",
        "HistGradientBoostingClassifier",
    ]

    models_dict = {
        "all": "",
        "LogisticRegression": LogisticRegression,
        "KNeighborsClassifier": KNeighborsClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "SVC": SVC,
        "LinearSVC": LinearSVC,
        "GaussianNB": GaussianNB,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
    }

    response = select("Select model:", choices=models).ask()
    response = models_dict[response]

    return response


def select_mode() -> str | None:
    questions = [
        "Selection Feature (Select K Best)",
        "PCA (Principal Component Analysis)",
        "t-SNE (T-distributed Stochastic Neighbor Embedding)",
        "SMOTE (Synthetic Minority Over Sampling Technique)",
        "Pass",
    ]

    response = select("Select mode:", choices=questions).ask()

    match response:
        case "Selection Feature (Select K Best)":
            return "selection feature"
        case "PCA (Principal Component Analysis)":
            return "PCA"
        case "t-SNE (T-distributed Stochastic Neighbor Embedding)":
            return "t-SNE"
        case "SMOTE (Synthetic Minority Over Sampling Technique)":
            return "SMOTE"
        case "Pass":
            return None
        case _:
            ValueError("It is not a valid mode")
