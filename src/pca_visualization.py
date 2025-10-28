from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_extractor import DataExtractor

"""
    This script performs PCA on the dataset for a given question, and plots the scatter of the components
    in the results directory. It also saves the resulting dataframe to csv.
    
    args:
        --question: question number [1-5] (by default 4)
        --label: depression or anxiety (by default depression)
        --components: number of components to use for PCA (by default 2)
        --results-dir: directory to save the results (by default pca_visualization/)

    Some examples of execution from root directory:
        python -m src.pca_visualization.py --question 4 --label depression --components 2 := runs PCA on question 4 with depression label and 2 components
        python -m src.pca_visualization.py --question 2 --label anxiety --components 3 := runs PCA on question 2 with anxiety label and 3 components
"""

def load_dataset(question: int, label: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load dataset for the given question
    """
    extractor = DataExtractor()
    df = extractor.extract_csv(
        temporality=False,
        question=question,
        feature="", # get all features
        labels=label
    )

    groups = df["ID"]
    y = df[label_column(label)]
    x = df.drop(columns=["ID", label_column(label)])
    return x, y, groups

def label_column(label: str) -> str:
    """
    Return the column name for the given label
    """
    return {"depression": "S_Depresión", "anxiety": "S_Ansiedad"}[label]

def run_pca(x: pd.DataFrame, n_components: int) -> tuple[pd.DataFrame, PCA]:
    """
    Run PCA on the given dataset
    """
    # pipeline to define the steps of data transformation: input -> scale -> pca
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")), # calculate median for missing values (probably not necessary)
            ("scaler", StandardScaler()), # scale the data such that each column has 0 mean and 1 std
            ("pca", PCA(n_components=n_components, random_state=42)) # apply PCA
        ]
    )
    components = pipeline.fit_transform(X=x) # fit_transform returns an array with the principal components  
    pca = pipeline.named_steps["pca"]
    
    # here also returns pca object to obtain the explained variance
    return pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)]), pca

def plot_scatter(components: pd.DataFrame, y: pd.Series, label: str, output_dir: Path, suffix: str, question_name: str, question_id: str) -> None:
    """
    Plot scatter of the components to the given output dir
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(components["PC1"], components["PC2"], c=y, cmap="coolwarm", alpha=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{question_name} PCA - {label}")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(label)
    fig.tight_layout()
    fig.savefig(output_dir / f"{question_id}_pca_{label}_{suffix}.png", dpi=200)
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments   
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=int, choices=range(1, 6), default=4)
    parser.add_argument("--label", choices=["depression", "anxiety"], default="depression")
    parser.add_argument("--components", type=int, default=2)
    parser.add_argument("--results-dir", default="pca_visualization/")
    return parser.parse_args()

def main() -> None:

    # parse arguments := question, label, components, results-dir
    args = parse_args()

    # load dataset 
    x, y, ids = load_dataset(label=args.label, question=args.question)
     
    extractor = DataExtractor()
    question_name = extractor.validate_question(args.question)
    question_id = question_name.split()[0].lower() # # ids: a1, p1, ..., p4
        
    # run PCA
    components, pca = run_pca(x=x, n_components=args.components)
    
    # make results dir for saving
    base_dir = Path(args.results_dir)
    results_dir = base_dir / question_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # define suffix for filename depending on the number of components requested
    suffix = f"pc{args.components}"

    # concatenate dataframes and save csv
    outfile = results_dir / f"{question_id}_pca_components_{args.label}_{suffix}.csv"
    components_df = pd.concat([components, y.reset_index(drop=True), ids.reset_index(drop=True)], axis=1)
    components_df.to_csv(outfile, index=False)
    
    # visualize scatter
    plot_scatter(components.iloc[:, :2], y, args.label, results_dir, suffix, question_name, question_id)
    
    # get variance explained for each component and then the total variance explained
    explained_ratios = np.array(pca.explained_variance_ratio_)
    total_explained = explained_ratios.sum()

    print(f"Variance explained for each component: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {total_explained}")

if __name__ == "__main__":
    main()



