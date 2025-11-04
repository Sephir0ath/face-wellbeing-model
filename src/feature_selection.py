import pandas as pd

class FeatureSelection:
    def __init__(self):
        pass

    def standard_scaler(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    def select_k_best(X: pd.DataFrame, y: pd.Series, n_features: int = 10) -> pd.DataFrame:
        # Feature Selection
        from sklearn.feature_selection import SelectKBest, f_classif
        
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X = selector.fit_transform(X, y)
        return X