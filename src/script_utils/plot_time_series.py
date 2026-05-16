import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties
import pandas as pd
import seaborn as sns
import argparse
from pathlib import Path

from ..data_extractor import DataExtractor

"""
Ejemplo ejecución:

    python -m src.visualizations_scripts.plot_time_series --question 4 --label depression


"""



def plot_time_series(time_series: pd.DataFrame, label: str, output: Path) -> None:
    """Plot a multivariate time series (few variables) for a single subject.

    Expected input:
    - `time_series` contains rows over time for ONE participant (filter by ID before calling)
    - It may contain `frame` and/or `timestamp` columns (as OpenFace typically does)
    - It may also contain non-numeric columns (e.g., ID) which will be ignored
    """

    output.parent.mkdir(parents=True, exist_ok=True)

    # Pick a time axis. Prefer OpenFace-style columns if present.
    if "timestamp" in time_series.columns:
        t_col = "timestamp"
    elif "frame" in time_series.columns:
        t_col = "frame"
    else:
        # Fall back to row index as time.
        time_series = time_series.reset_index(drop=True).copy()
        time_series["_t"] = time_series.index
        t_col = "_t"

    # Choose a small set of numeric variables to plot.
    # Template: override this list with the exact columns you want.
    preferred_cols = [
        # Example OpenFace-ish columns (will be used only if they exist)
        "pose_Rx",
        "pose_Ry",
        "pose_Rz",
        "gaze_angle_x",
        "gaze_angle_y",
        "AU12_r",
        "AU04_r",
    ]
    numeric_cols = [c for c in preferred_cols if c in time_series.columns]

    # Fallback: take the first few numeric columns (excluding time and obvious non-features).
    if not numeric_cols:
        exclude = {"ID", t_col}
        numeric_cols = [
            c
            for c in time_series.select_dtypes(include="number").columns
            if c not in exclude
        ][:6]

    if not numeric_cols:
        raise ValueError(
            "No numeric feature columns found to plot. "
            "Pass a DataFrame with OpenFace numeric columns (e.g., AU*_r, pose_*, gaze_*)."
        )

    # Convert to long format for seaborn.
    long_df = time_series[[t_col, *numeric_cols]].melt(
        id_vars=[t_col], var_name="variable", value_name="value"
    )
    long_df = long_df.sort_values(by=t_col, kind="stable")

    # Plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(
        data=long_df,
        x=t_col,
        y="value",
        hue="variable",
        ax=ax,
        linewidth=1.2,
    )

    # ax.set_title(f"Time series ({label})")
    ax.set_xlabel("Time" if t_col == "timestamp" else t_col)
    ax.set_ylabel("Value")

    # Keep x-axis readable for long sequences
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.legend(loc="upper right", ncol=1, frameon=True)

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=int, choices=range(1, 6), default=4)
    parser.add_argument("--label", choices=["depression", "anxiety"], default="depression")
    parser.add_argument(
        "--id",
        dest="participant_id",
        default=None,
        help="Participant folder name under data/ (exact ID). Defaults to the first ID found.",
    )
    parser.add_argument(
        "--outdir",
        default="visualizations",
        help="Output directory for saved plots.",
    )
    return parser.parse_args()





if __name__ == "__main__":
    args = parse_args()

    extractor = DataExtractor()
    df, _ = extractor.extract_csv(
        temporality=True,
        question=args.question,
        feature="",
        labels=args.label,
    )

    # Filtra por ID para obtener la serie temporal de una sola persona
    participant_id = args.participant_id
    if participant_id is None:
        participant_id = df["ID"].iloc[0]

    time_series = df[df["ID"] == participant_id].copy()

    outdir = Path(args.outdir)
    output = outdir / f"ts_q{args.question}_{args.label}_{participant_id}.png"
    plot_time_series(time_series=time_series, label=args.label, output=output)
    print("Time series guardada en:", output)


