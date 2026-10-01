"""
merge_results_kmedian.py

Merges all parquet files produced by kmowa.py's parallel OAR jobs (one
file per (n, k, methods) by default -- see submit_array_oar_kmedian_*.sh)
into a single results file.

UNLIKE merge_results.py / merge_results_elicitation.py, this does NOT
simply keep the last row per key: 'exact' and 'min_sum_k_median' can be
run in SEPARATE jobs (so you don't launch both at once -- see
submit_array_oar_kmedian_*.sh's METHODS parameter), each writing its own
file with the OTHER method's columns left as NaN for the same
(w_type, n, k, epsilon, seed, instance) key. A plain "keep last" merge
would silently erase one method's results with the other's NaNs whenever
both files are merged together. Instead, for each key, every column is
combined by taking the last NON-NULL value found across all matching
rows -- so an 'exact'-only row and a 'min_sum_k_median'-only row for the
same key are merged into ONE row with both sets of columns filled in.

epsilon has no effect on 'exact' (kmowa_solver doesn't take it), so an
'exact'-only run still produces one row PER epsilon value swept (same
exact value repeated) to line up with 'min_sum_k_median' rows -- this
only works if the SAME --epsilon-values were used for both runs (true by
construction if EPSILON_VALUES in submit_array_oar_kmedian_*.sh was not
changed between the two job submissions).

Usage:
    python3 merge_results_kmedian.py 'results/kmedian_*.parquet' \
        --output kmedian_merged.parquet
"""

import argparse
import glob

import numpy as np
import pandas as pd

KEY_COLS = ["w_type", "n", "k", "epsilon", "seed", "instance"]


def _combine_last_nonnull(series: pd.Series):
    """Returns the last non-null value in the series, or NaN if all are
    null. Used to combine several partial rows (one method's columns
    filled, the other's NaN) sharing the same key into a single row."""
    s = series.dropna()
    return s.iloc[-1] if len(s) else np.nan


def merge(paths, output_path):
    dfs = []
    for p in paths:
        try:
            dfs.append(pd.read_parquet(p))
        except Exception as e:
            print(f"[ignore] could not read {p}: {e}")

    if not dfs:
        print("No readable file found.")
        return None

    combined = pd.concat(dfs, ignore_index=True)

    missing_key_cols = [c for c in KEY_COLS if c not in combined.columns]
    if missing_key_cols:
        raise ValueError(
            f"Key column(s) missing from the input files: {missing_key_cols}. "
            "Are these really kmowa.py results?"
        )

    n_before_rows = len(combined)
    other_cols = [c for c in combined.columns if c not in KEY_COLS]

    # Combine, per key, every non-key column by its last non-null value
    # across all rows sharing that key (instead of dropping all but the
    # last ROW, which would erase a method's results whenever the other
    # method's row for the same key has NaN there -- see module docstring).
    combined = (
        combined
        .groupby(KEY_COLS, as_index=False, dropna=False)
        .agg({c: _combine_last_nonnull for c in other_cols})
        .sort_values(by=KEY_COLS)
        .reset_index(drop=True)
    )
    n_after_rows = len(combined)
    if n_after_rows < n_before_rows:
        print(f"{n_before_rows - n_after_rows} duplicate key(s) "
              f"({', '.join(KEY_COLS)}) combined into one row each "
              "(columns merged, not just the last row kept).")

    # ratio_min_sum_k_median could only be computed by run_one() when BOTH
    # values were available in the SAME job -- an 'exact'-only row and a
    # 'min_sum_k_median'-only row each had NaN there (missing the other
    # side). Recompute it here now that both columns have been combined
    # into one row, wherever both are present.
    if {"exact_owa_obj", "min_sum_k_median_owa_obj"}.issubset(combined.columns):
        can_compute = (
            combined["exact_owa_obj"].notna()
            & (combined["exact_owa_obj"] > 0)
            & combined["min_sum_k_median_owa_obj"].notna()
        )
        combined.loc[can_compute, "ratio_min_sum_k_median"] = (
            combined.loc[can_compute, "min_sum_k_median_owa_obj"]
            / combined.loc[can_compute, "exact_owa_obj"]
        )

    combined.to_parquet(output_path, index=False)
    print(f"{n_after_rows} rows written to {output_path}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="+",
                         help="paths or glob patterns of the parquet files to merge "
                              "(e.g. 'results/kmedian_*.parquet')")
    parser.add_argument("--output", default="kmedian_merged.parquet")
    args = parser.parse_args()

    paths = []
    for pattern in args.patterns:
        matched = glob.glob(pattern)
        paths.extend(matched if matched else [pattern])

    print(f"{len(paths)} file(s) found:")
    for p in paths:
        print(" ", p)

    merge(paths, args.output)
