import pandas as pd
from .structure import Structure


class FeatureSelection(Structure):

    def apply(
        self, X: pd.DataFrame, y: pd.Series, n_features: int = 10
    ) -> pd.DataFrame:
        # Feature Selection
        from sklearn.feature_selection import SelectKBest, f_classif

        selector = SelectKBest(score_func=f_classif, k=n_features)
        X = selector.fit_transform(X, y)
        return X
