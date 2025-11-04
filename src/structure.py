import pandas as pd

class Structure: 
    def __init__(self):
        pass

    def standard_scaler(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled