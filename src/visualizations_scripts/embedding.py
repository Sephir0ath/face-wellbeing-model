from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import umap

from ..data_extractor import DataExtractor

ROOT_DIR = Path(__file__).resolve().parents[2]
FONT_PATH = ROOT_DIR / "fonts" / "DM_Sans.ttf"
fm.fontManager.addfont(FONT_PATH)
FONT_NAME = fm.FontProperties(fname=FONT_PATH).get_name()
sns.set_theme(style="whitegrid", rc={"font.family": FONT_NAME})


def label_column(label: str) -> str:
    return "S_Depresión" if label == "depression" else "S_Ansiedad"


def load_dataset(question: int, label: str) -> tuple[pd.DataFrame, pd.Series]:
    extractor = DataExtractor()
    df, _ = extractor.extract_csv(
        temporality=False,
        question=question,
        feature="",
        labels=label,
    )
    y_col = label_column(label)
    y = df[y_col]
    X = df.drop(columns=["ID", y_col])
    return X, y


def build_pipeline(method: str, components: int) -> Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    match method:
        case "pca":
            steps.append(("embed", PCA(n_components=components, random_state=42)))
        case "tsne":
            steps.append(
                (
                    "embed",
                    TSNE(
                        n_components=components,
                        perplexity=15,
                        learning_rate="auto",
                        init="pca",
                        random_state=42,
                    ),
                )
            )
        case "umap":
            if umap is None:
                raise RuntimeError("umap-learn no está instalado.")
            steps.append(
                (
                    "embed",
                    umap.UMAP(
                        n_components=components,
                        n_neighbors=15,
                        min_dist=0.1,
                        random_state=42,
                    ),
                )
            )
        case _:
            raise ValueError("Método inválido. Usa: pca, tsne o umap.")

    return Pipeline(steps=steps)


def generate_embedding(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipeline.fit_transform(X)


def plot_embedding(
    embedding: np.ndarray,
    y: pd.Series,
    label: str,
    method: str,
    output: Path,
    question_name: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#E9E9F0")
    ax.set_facecolor("#E9E9F0")

    if embedding.shape[1] >= 2:
        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=y,
            palette=["#5D55ED", "#E4572E"],
            ax=ax,
            s=60,
            alpha=0.8,
        )
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
    else:
        sns.stripplot(
            x=embedding[:, 0],
            y=y,
            palette=["#5D55ED", "#E4572E"],
            ax=ax,
            alpha=0.8,
            orient="h",
        )
        ax.set_xlabel("Dim 1")
        ax.set_ylabel(label)

    ax.set_title(f"{question_name} – {method.upper()} ({label})", fontfamily=FONT_NAME)
    ax.legend(title=label, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=int, choices=range(1, 6), default=4)
    parser.add_argument("--label", choices=["depression", "anxiety"], default="depression")
    parser.add_argument("--method", choices=["pca", "tsne", "umap"], default="pca")
    parser.add_argument("--components", type=int, default=2)
    parser.add_argument("--outdir", default="visualizations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X, y = load_dataset(args.question, args.label)
    target_col = label_column(args.label)

    pipeline = build_pipeline(args.method, args.components)
    embedding = generate_embedding(pipeline, X)

    extractor = DataExtractor()
    question_name = extractor.validate_question(args.question)
    question_id = question_name.split()[0].lower()

    output = (
        Path(args.outdir)
        / question_id
        / f"{question_id}_{args.method}_{args.label}_dim{args.components}.png"
    )

    plot_embedding(embedding, y, args.label, args.method, output, question_name)
    print("Embedding guardado en:", output)

    # Guarda los componentes en CSV
    components_df = pd.DataFrame(
        embedding, columns=[f"dim_{i+1}" for i in range(embedding.shape[1])]
    )
    components_df[target_col] = y.reset_index(drop=True)
    csv_path = output.with_suffix(".csv")
    components_df.to_csv(csv_path, index=False)
    print("Componentes guardados en:", csv_path)


if __name__ == "__main__":
    main()
