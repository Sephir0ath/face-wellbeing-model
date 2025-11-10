from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties
import pandas as pd
import seaborn as sns

from ..data_extractor import DataExtractor

ROOT_DIR = Path(__file__).resolve().parents[2]
FONT_PATH = ROOT_DIR / "fonts" / "DM_Sans.ttf"
fm.fontManager.addfont(FONT_PATH)
FONT_NAME = FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["font.family"] = FONT_NAME
sns.set_theme(style="whitegrid", rc={"font.family": FONT_NAME})


def load_counts(question: int | None, label: str) -> pd.DataFrame:
    extractor = DataExtractor()

    if question is None:
        data, _ = extractor.extract_csv(
            temporality=False,
            question=4,   # -> solo para acceder a alguna pregunta 
            feature="",
            labels=label,
        )
    else:
        data, _ = extractor.extract_csv(
            temporality=False,
            question=question,
            feature="",
            labels=label,
        )

    column = "S_Depresión" if label == "depression" else "S_Ansiedad"
    counts = data.groupby(column).size().reset_index(name="count")
    counts[column] = counts[column].map({0: f"No {label}", 1: f"{label.title()}"})
    return counts


def plot_counts(counts: pd.DataFrame, title: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#E9E9F0")
    ax.set_facecolor("#E9E9F0")
    sns.barplot(
        data=counts,
        x=counts.columns[0],
        y="count",
        ax=ax,
        color="#5D55ED",
        edgecolor="black",
    )
    ax.set_title(title, fontfamily=FONT_NAME)
    ax.set_ylabel("Número de personas", fontfamily=FONT_NAME)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, counts["count"].max() * 1.1)
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=int, choices=range(1, 6), help="Si se omite, usa todos los registros")
    parser.add_argument("--label", choices=["depression", "anxiety"], default="depression")
    parser.add_argument("--outdir", default="visualizations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = load_counts(args.question, args.label)

    if args.question is None:
        title = f"Balance de clases ({args.label})"
        outfile = Path(args.outdir) / "overall" / f"class_balance_{args.label}.png"
    else:
        extractor = DataExtractor()
        q_name = extractor.validate_question(args.question)
        title = f"Balance de clases ({args.label}) – {q_name}"
        outfile = Path(args.outdir) / f"{q_name.split()[0].lower()}" / f"class_balance_{args.label}.png"

    plot_counts(counts, title, outfile)
    print("Guardado en:", outfile)


if __name__ == "__main__":
    main()
