import pandas as pd
from structure import Structure

class Smote(Structure): 

    def apply_smote(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        return X_resampled, y_resampled