"""
This class extract DataFrames with respective information you need.

"""

from pathlib import Path
import pandas as pd


class DataExtractor:
    """
    How to use?

    First parametter is temporality, if you want temporal data put True, else put False for stats data.
    Second parametter is question number (1 to 5).
    Third parametter is feature name, you can put one of this options: gaze, pose, 2d_landmarks, 3d_landmarks, pdm, AU, eye_lmk.
    If you want all features, just put an empty string "".
    Fourth parametter is labels type, you can put one of this options: depression, anxiety, both.
    

    Example:
        extractor = DataExtractor()
        dataframe = extractor.extract_csv(temporality=True, question=2, feature="gaze", labels="depression")
    """

    route: Path

    def __init__(self):
        # Extract current path + data folder
        self.route = Path.cwd() / "data"

    def extract_csv(
        self, temporality: bool, question: int, feature: str, labels: str
    ) -> pd.DataFrame:
        """
        Extract CSV file based on question number and feature name.

        Args:
            question (int): The question number.
            feature (str): The feature name.

        Returns:
            pd.DataFrame: The extracted DataFrame.
        """

        df = pd.DataFrame()
        df_list = []

        for user in self.route.iterdir():
            # Pass csv labels
            if "labels.csv" in user.name:
                continue

            # Update route with subfolder (frame2frame or stats)
            user_path = user / self.if_temporality(temporality)

            # Update route with question folder
            user_path /= self.validate_question(question)

            # Add feature to route
            user_path /= self.validate_feature(feature=feature, question=question)

            # Final route with feature csv
            temp_df = self.upload_csv(user_path)

            # Add user column
            temp_df["ID"] = user.name 

            # Add df to list
            df_list.append(temp_df)

        # Concatenate all dataframes
        df = pd.concat(df_list, ignore_index=True)

        # Add labels if needed
        df = self.add_labels(df, labels)
        return df

    def if_temporality(self, temporality: bool) -> str:
        """
        Determine the subfolder based on temporality.

        Args:
            temporality (bool): True for temporal data, False for stats data.

        Returns:
            str: Subfolder name.
        """
        return "facial_features" if temporality else "facial_features_stats"

    def validate_question(self, question: int) -> str:
        """
        Validate if the question number is within the valid range and return respective route.

        Args:
            question (int): The question number.

        Returns:
            str: Question route.
        """

        match question:
            case 1:
                return "A1 Lectura de parrafo"
            case 2:
                return "P1 Trabajo de tus sueños"
            case 3:
                return "P2 Evento con influencia"
            case 4:
                return "P3 Consejo a tu yo mas joven"
            case 5:
                return "P4 Orgulloso"
            case _:
                raise ValueError("Invalid question number. Must be between 1 and 5.")

    def validate_feature(self, feature: str, question: int) -> str:
        """
        Validate if the feature name is within the valid options.

        Args:
            feature (str): The feature name.
        Returns:
            str: Validated feature name.
        """
        feature_route = ""

        # Number of question
        if question == 1:
            feature_route += "1_"
        else:
            feature_route += f"{question - 1}_"

        # Add name question
        feature_route += self.validate_question(question)

        # Return all features
        if not feature:
            return feature_route + ".csv"

        # Validate feature name
        match feature:
            case (
                "gaze"
                | "2d_landmarks"
                | "3d_landmarks"
                | "AU"
                | "eye_lmk"
                | "pdm"
                | "pose"
            ):
                return feature_route + "_" + feature + ".csv"
            case _:
                raise ValueError(
                    "Invalid feature name. Must be one of: gaze, pose, 2d landmarks, 3d landmarks, pdm, AU, eye_lmk."
                )

    def upload_csv(self, user_route: Path) -> pd.DataFrame:
        """
        Upload a CSV file for a specific user and feature.

        Args:
            user_route (Path): The path to the user's folder.

        Returns:
            pd.DataFrame: The uploaded DataFrame.
        """
        df = pd.read_csv(user_route)

        # Perform upload operation (e.g., to a database or cloud storage)
        return df

    def add_labels(self, df: pd.DataFrame, labels: str) -> pd.DataFrame:
        # Add labels to the dataframe based on the specified label type.
        match labels:
            case "depression":
                labels = pd.read_csv(self.route / "labels.csv", usecols=["ID", "S_Depresión"])
            case "anxiety":
                labels = pd.read_csv(self.route / "labels.csv", usecols=["ID", "S_Ansiedad"])
            case "both":
                labels = pd.read_csv(self.route / "labels.csv", usecols=["ID", "S_Depresión", "S_Ansiedad"])
            case _:
                raise ValueError(
                    "Invalid labels name. Must be one of: depression, anxiety, both."
                )
        
        df_labels = pd.merge(df, labels, on="ID")
        return df_labels