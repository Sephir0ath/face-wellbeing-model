import pandas as pd

class Smote: 
    def __init__(self):
        pass

    def standard_scaler(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    def apply_smote(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        return X_resampled, y_resampled