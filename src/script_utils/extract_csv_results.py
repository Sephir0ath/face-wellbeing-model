import csv
import argparse

"""
Este script lee csv de resultados y extrae los mejores modelos (con f1-score más alto)

Ejemplo ejecución:
    
    python -m src.script_utils.extract_csv_results --csv_file <ruta_csv>

"""

def to_int(value: str) -> int:
    return int(value.strip())


def to_float(value: str) -> float:
    return float(value.strip())


def load_rows(csv_file: str) -> list[dict]:
    """Load result rows from CSV.

    Expected header columns:
    question,temporality,feature,label,mode,model,f1_binary_mean,f1_binary_std,f1_macro_mean,f1_macro_std
    """
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            # Parse numeric fields (keep others as strings)
            r["question"] = to_int(r["question"])
            r["f1_binary_mean"] = to_float(r["f1_binary_mean"])
            r["f1_binary_std"] = to_float(r["f1_binary_std"])
            r["f1_macro_mean"] = to_float(r["f1_macro_mean"])
            r["f1_macro_std"] = to_float(r["f1_macro_std"])
            rows.append(r)
    return rows


def top_k_by_question_and_label(rows: list[dict], k: int = 3) -> dict[tuple[int, str], list[dict]]:
    """Return top-k configs per (question, label) by f1_binary_mean.

    Tie-breakers: lower std, then higher macro mean.
    """
    groups: dict[tuple[int, str], list[dict]] = {}
    for r in rows:
        key = (r["question"], r["label"])
        groups.setdefault(key, []).append(r)

    out: dict[tuple[int, str], list[dict]] = {}
    for key, items in groups.items():
        items_sorted = sorted(
            items,
            key=lambda x: (
                -x["f1_binary_mean"],
                x["f1_binary_std"],
                -x["f1_macro_mean"],
            ),
        )
        out[key] = items_sorted[:k]
    return out


def write_summary_csv(
    top: dict[tuple[int, str], list[dict]],
    out_file: str,
) -> None:
    fieldnames = [
        "question",
        "label",
        "rank",
        "temporality",
        "feature",
        "mode",
        "model",
        "f1_binary_mean",
        "f1_binary_std",
        "f1_macro_mean",
        "f1_macro_std",
    ]

    rows_out: list[dict] = []
    for (question, label), items in sorted(top.items(), key=lambda x: (x[0][0], x[0][1])):
        for rank, r in enumerate(items, start=1):
            rows_out.append(
                {
                    "question": question,
                    "label": label,
                    "rank": rank,
                    "temporality": r["temporality"],
                    "feature": r["feature"],
                    "mode": r["mode"],
                    "model": r["model"],
                    "f1_binary_mean": r["f1_binary_mean"],
                    "f1_binary_std": r["f1_binary_std"],
                    "f1_macro_mean": r["f1_macro_mean"],
                    "f1_macro_std": r["f1_macro_std"],
                }
            )

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    _ = parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Ruta al archivo CSV de resultados"
    )

    _ = parser.add_argument(
        "--out",
        type=str,
        default="results_resume.csv",
        help="Ruta del CSV resumen de salida",
    )

    args = parser.parse_args()

    rows = load_rows(args.csv_file)
    top = top_k_by_question_and_label(rows, k=3)

    write_summary_csv(top, args.out)
    print("Resumen guardado en:", args.out)
