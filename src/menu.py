from questionary import select
from sklearn.exceptions import ConvergenceWarning
import warnings

# Hide warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


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

    response = select("Temporary data:", choices=questions).ask()

    response = "" if response == "All features" else response

    return response


def select_label() -> str:
    questions = ["depression", "anxiety", "both"]

    response = select("Temporary data:", choices=questions).ask()

    return response
