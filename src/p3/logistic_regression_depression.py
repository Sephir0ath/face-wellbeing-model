from __future__ import annotations

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..data_extractor import DataExtractor

"""
    This script builds logistic regression model for question 4 (P3) with depression label, and saves
    it to a pickle file in models/p3/. It also performs cross-validation on the model and saves the results to a csv
    in results/p3/
"""

def load_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load dataset for P3 question
    """
    data_extractor = DataExtractor()
    p3_df = data_extractor.extract_csv(
        temporality=False,
        question=4,
        feature="", # get all features
        labels="depression"
    )

    groups = p3_df["ID"]
    y = p3_df["S_Depresión"]
    x = p3_df.drop(columns=["ID", "S_Depresión"])
    return x, y, groups

def build_pipeline() -> Pipeline:
    """
    Build pipeline for the model, including preprocessing (imputer and scaling) and logistic regression
    """
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")), 
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ]
    )

    return pipeline

def cross_validate(pipeline: Pipeline, x: pd.DataFrame, y: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """
    Perform cross-validation on the dataset using 'leave-one-group-out' strategy and return the results.
    Each person is a group.
    """
    # TODO: maybe try k-fold cross-validation strategy?
    
    logo = LeaveOneGroupOut()
    rows = []
    
    for train_index, test_index in logo.split(x, y, groups=groups):
        model = clone(pipeline)    
        model.fit(x.iloc[train_index], y.iloc[train_index])
        
        # predict on test set
        y_test = y.iloc[test_index]
        preds = model.predict(x.iloc[test_index])
        probs = model.predict_proba(x.iloc[test_index])[:, 1]
        
        # calculate metrics
        rows.append({
            "accuracy": accuracy_score(y_test, preds),
            "balanced_accuracy": balanced_accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probs) if y_test.nunique() > 1 else np.nan
        })

    return pd.DataFrame(rows)

def main() -> None:
    # load dataset
    x, y, groups = load_dataset()

    # build pipeline
    pipeline = build_pipeline()
    
    # cross-validate
    cv_results = cross_validate(pipeline, x, y, groups)
    
    # save results
    results_dir = Path("results/p3/")
    results_dir.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(results_dir / "logistic_regression_cv.csv", index=False)
    
    print(f"Cross-validation results saved to: {results_dir / 'logistic_regression_cv.csv'}")
    print(f"Average metrics: \n")
    print(cv_results.mean())

    fitted_model = pipeline.fit(x, y)
    
    # save model
    models_dir = Path("models/p3/")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_model, models_dir / "logistic_regression.pkl")
    
    print(f"Model saved to: {models_dir / 'logistic_regression.pkl'}")


if __name__ == "__main__":
    main()
