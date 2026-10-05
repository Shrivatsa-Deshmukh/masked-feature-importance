"""
Combine the seeds of one experiment folder.

Prints, and writes to results/<folder>/summary_*.csv:
  1. importance: mean / sd / n across seeds per (method, feature, estimand)
  2. agreement:  *_vs_* statistics (Spearman, Pearson, MAE), mean / sd across seeds
  3. accuracy:   full-set log-prob / R^2 / fit gaps, masked NPE vs dedicated NPE
  4. R^2 drops:  masked NPE vs retrain ablation, per left-out feature

Usage (from the repository root):
    python aggregate.py results/jansen_rit
    python aggregate.py results/toy results/jansen_rit_marginal
"""

import argparse
from pathlib import Path

import pandas as pd

READ_KWARGS = dict(keep_default_na=False, na_values=[""], float_precision="round_trip")


def load(folder: Path, table: str) -> pd.DataFrame:
    path = folder / f"{table}.csv"
    return pd.read_csv(path, **READ_KWARGS) if path.exists() else pd.DataFrame()


def summarize(folder: Path) -> None:
    seeds = load(folder, "seeds")
    if seeds.empty:
        print(f"{folder}: no results yet.")
        return
    print(f"\n{'#' * 74}\n# {folder}: {len(seeds)} seeds ({', '.join(map(str, seeds['seed']))})\n{'#' * 74}")

    # 1. importance
    imp = load(folder, "importance")
    s1 = imp.groupby(["method", "estimand", "feature"])["value"].agg(["mean", "std", "count"]).reset_index()
    s1.to_csv(folder / "summary_importance.csv", index=False)
    cell = s1.assign(v=s1["mean"].map("{:.3f}".format) + " ± " + s1["std"].fillna(0).map("{:.3f}".format))
    print("\n=== importance (nats), mean ± sd across seeds")
    print(cell.pivot_table(index="feature", columns=["method", "estimand"], values="v", aggfunc="first").to_string())

    # 2. agreement
    agree = [c for c in seeds.columns if "_vs_" in c]
    if agree:
        s2 = seeds[agree].agg(["mean", "std"]).T
        s2.to_csv(folder / "summary_agreement.csv", index_label="statistic")
        print("\n=== agreement (mean / sd across seeds)")
        print(s2.round(3).to_string())

    # 3. accuracy
    acc = [c for c in seeds.columns
           if c.startswith(("masked_", "dedicated_", "exact_"))]
    s3 = seeds[acc].agg(["mean", "std"]).T
    s3.to_csv(folder / "summary_accuracy.csv", index_label="statistic")
    print("\n=== full-set accuracy and training")
    print(s3.round(4).to_string())

    # 4. R^2 drops
    rec = load(folder, "parameter_recovery")
    if not rec.empty:
        rec = rec[rec["left_out_feature"].fillna("") != ""]
        if not rec.empty:
            s4 = (rec.groupby(["left_out_feature", "model"])["r2_drop_mean"]
                  .agg(["mean", "std", "count"]).unstack("model"))
            s4.to_csv(folder / "summary_r2_drop.csv")
            print("\n=== leave-one-out R^2 drop")
            print(s4.round(4).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path, help="Experiment folders, e.g. results/jansen_rit")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 400)
    for folder in ap.parse_args().folders:
        summarize(folder)


if __name__ == "__main__":
    main()
