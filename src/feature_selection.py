import pandas as pd
from .structure import Structure


class FeatureSelection(Structure):

    def apply(
            self, X: pd.DataFrame, y: pd.Series, n_features: int = 10  
        ) -> pd.DataFrame:
        
        from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

        # --- Step 1: Remove constant columns (zero variance) ---
        vt = VarianceThreshold(threshold=0)
        X = vt.fit_transform(X)
        X = pd.DataFrame(X)  # Keep it as DataFrame for consistency

        # --- Step 2: Remove rows with missing values (NaN) ---
        X = X.dropna()
        y = y.loc[X.index]

        # --- Step 3: Feature Selection ---
        selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
        X_selected = selector.fit_transform(X, y)

        return pd.DataFrame(X_selected)

