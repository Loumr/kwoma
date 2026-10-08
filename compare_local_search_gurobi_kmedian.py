"""
compare_local_search_gurobi_kmedian.py

Compares kmowa.local_search_single_swap (Arya et al.'s single-swap local
search) with the EXACT solution of the PLAIN (min-sum) k-median problem,
computed by Gurobi (kmowa.kmedian_minsum_solver), and SAVES every result
(objective values, running times, open facilities) in a parquet file so
that the ratios local_search / exact can be analysed afterwards.

Notation (same as kmowa_solver): j = cities (clients), i = facilities.
    C[j, i] = cost of serving city j from facility i
    -> rows = cities, columns = facilities (the layout local_search_single_swap expects).

Instances: random_metric_costs(n, seed=seed + instance), i.e. EXACTLY the
same instances as the OWA experiments of kmowa.py, so the exact cache
(results/exact_minsum_cache_n{n}_k{k}.parquet) is shared with them and the
min-sum / OWA results can be compared instance by instance.

Exact results go through kmowa.solve_minsum_exact_cached:
  - each (n, k, seed, instance) is solved by Gurobi at most ONCE (reused
    across epsilon values, re-runs, and parallel jobs -- file lock);
  - the ORIGINAL Gurobi times are stored in the cache, so a result read
    from the cache still reports its real solve time.

One row per (n, k, seed, instance, epsilon, ls_init) is upserted into
--output after every instance (a job killed by the walltime keeps
everything computed so far).

Columns of the output file:
    n, k, seed, instance, epsilon, ls_init
    ls_obj, ls_time, ls_n_iters, ls_S
    exact_obj, exact_bound, exact_gap, exact_optimal, exact_status,
    exact_time (wall: env + model build + solve), exact_solver_time
    (Gurobi Runtime only), exact_from_cache, exact_S
    ratio        = ls_obj / exact_obj    (NaN unless exact is OPTIMAL)
    ratio_ub     = ls_obj / exact_bound  (upper bound on the true ratio,
                   still valid if Gurobi hit its time limit)
    ls_is_optimal, max_ratio (theoretical 5/(1-eps)), bound_ok

Usage:
    python3 compare_local_search_gurobi_kmedian.py --n-values 20 50 100 \
        --k-values 2 5 10 --nb-instances 30 --epsilons 0 0.1 0.01 \
        --output results/ls_vs_gurobi_minsum.parquet

    # split on a cluster: one job fills the exact cache, the other runs
    # the local search (exact values are read from the cache when present)
    ... --methods exact
    ... --methods local_search

    # rebuild ratios + summary once every job is done (re-reads the cache,
    # useful if the local_search job finished before the exact one)
    python3 compare_local_search_gurobi_kmedian.py --report \
        'results/ls_vs_gurobi_minsum*.parquet' --output ls_vs_gurobi_merged.parquet
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmowa import (
    local_search_single_swap,
    solve_minsum_exact_cached,
    default_minsum_cache_path,
    _minsum_cache_lookup,
    _atomic_to_parquet,
)
from generate_instances_k_median import random_metric_costs, check_triangle_inequality


METHODS = ("local_search", "exact")
KEY_COLS = ["n", "k", "seed", "instance", "epsilon", "ls_init"]
EXACT_COLS = ["exact_obj", "exact_bound", "exact_gap", "exact_optimal", "exact_status",
              "exact_time", "exact_solver_time", "exact_from_cache", "exact_S"]


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def theoretical_ratio(eps):
    """Arya et al.: 5 for the naive single swap, 5/(1-eps) with the threshold."""
    return 5.0 if eps <= 0 else 5.0 / (1.0 - eps)


def empty_exact():
    d = {c: np.nan for c in EXACT_COLS}
    d["exact_status"] = None
    d["exact_S"] = None
    d["exact_optimal"] = None
    d["exact_from_cache"] = None
    return d


def exact_from_result(res, from_cache):
    return {
        "exact_obj": res["obj"],
        "exact_bound": res["bound"],
        "exact_gap": res["gap"],
        "exact_optimal": bool(res["optimal"]),
        "exact_status": res["status"],
        "exact_time": res["wall_time"],
        "exact_solver_time": res["solver_time"],
        "exact_from_cache": bool(from_cache),
        "exact_S": res["S"],
    }


def add_ratios(df):
    """(Re)computes the derived columns from the raw ones."""
    df = df.copy()
    ls = df["ls_obj"].astype(float)
    ex = df["exact_obj"].astype(float)
    lb = df["exact_bound"].astype(float)
    opt = df["exact_optimal"].fillna(False).astype(bool)
    df["ratio"] = np.where(opt & (ex > 0), ls / ex, np.nan)
    df["ratio_ub"] = np.where(lb > 0, ls / lb, np.nan)
    df["ls_is_optimal"] = np.where(opt & ls.notna(), ls <= ex * (1 + 1e-9), np.nan)
    df["max_ratio"] = df["epsilon"].astype(float).map(theoretical_ratio)
    # checked on ratio_ub (valid even without proven optimality)
    df["bound_ok"] = np.where(df["ratio_ub"].notna(),
                              df["ratio_ub"] <= df["max_ratio"] + 1e-6, np.nan)
    return df


def upsert(row, path):
    df_new = pd.DataFrame([row])
    if os.path.exists(path):
        df_new = (pd.concat([pd.read_parquet(path), df_new], ignore_index=True)
                  .drop_duplicates(subset=KEY_COLS, keep="last")
                  .sort_values(KEY_COLS)
                  .reset_index(drop=True))
    _atomic_to_parquet(df_new, path)


def fmt(v, spec=".4f"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return format(v, spec)


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------

def run_comparison(n_values, k_values, nb_instances, seed, epsilons, ls_init,
                   methods, time_limit, threads, cache_dir, output_path,
                   instance_start=0, instance_end=None):
    instance_end = nb_instances if instance_end is None else instance_end
    rows = []

    print(f"{'n':>4} {'k':>4} {'inst':>4} {'eps':>6} | {'ls_obj':>12} {'exact_obj':>12} "
          f"{'ratio':>8} {'iters':>5} | {'t_ls(ms)':>10} {'t_exact(s)':>10} | status")
    print("-" * 105)

    for n in n_values:
        for k in k_values:
            if not 1 <= k < n:
                print(f"[skip] n={n} k={k}: need 1 <= k < n")
                continue
            cache_path = default_minsum_cache_path(n, k, cache_dir)

            for instance in range(instance_start, instance_end):
                C = random_metric_costs(n, seed=seed + instance)
                assert check_triangle_inequality(C), (n, k, instance)

                # ---- exact: ONCE per instance, shared by every epsilon ----
                exact = empty_exact()
                if "exact" in methods:
                    res, from_cache = solve_minsum_exact_cached(
                        C, n, k, seed, instance, cache_path,
                        time_limit=time_limit, threads=threads)
                    exact = exact_from_result(res, from_cache)
                else:
                    # local_search-only job: reuse an exact result if another
                    # job already put it in the cache, never call Gurobi
                    cached = _minsum_cache_lookup(n, k, seed, instance, cache_path, time_limit)
                    if cached is not None:
                        exact = exact_from_result(cached, True)

                # ---- local search: once per epsilon ----
                for eps in epsilons:
                    row = {"n": n, "k": k, "seed": seed, "instance": instance,
                           "epsilon": float(eps), "ls_init": ls_init,
                           "ls_obj": np.nan, "ls_time": np.nan,
                           "ls_n_iters": np.nan, "ls_S": None}
                    if "local_search" in methods:
                        t0 = time.perf_counter()
                        S, obj, n_iters = local_search_single_swap(
                            C, k=k, epsilon=eps, init=ls_init, seed=seed + instance)
                        row.update({"ls_obj": float(obj),
                                    "ls_time": time.perf_counter() - t0,
                                    "ls_n_iters": n_iters,
                                    "ls_S": [int(s) for s in S]})
                    row.update(exact)
                    row = add_ratios(pd.DataFrame([row])).iloc[0].to_dict()
                    rows.append(row)
                    upsert(row, output_path)

                    if row["bound_ok"] is None or (isinstance(row["bound_ok"], float)
                                                   and np.isnan(row["bound_ok"])):
                        status = row["exact_status"] or "no exact"
                    else:
                        status = "OK" if row["bound_ok"] else f"FAIL (> {row['max_ratio']:.3f})"
                    if row["exact_from_cache"] is True:
                        status += " [cache]"

                    t_ls = row["ls_time"] * 1000 if not np.isnan(row["ls_time"]) else np.nan
                    print(f"{n:>4} {k:>4} {instance:>4} {eps:>6.3f} | "
                          f"{fmt(row['ls_obj'], '.1f'):>12} {fmt(row['exact_obj'], '.1f'):>12} "
                          f"{fmt(row['ratio']):>8} {fmt(row['ls_n_iters'], '.0f'):>5} | "
                          f"{fmt(t_ls, '.3f'):>10} {fmt(row['exact_time'], '.3f'):>10} | {status}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Report: re-attach exact results from the cache + summary
# ---------------------------------------------------------------------

def attach_exact_from_cache(df, cache_dir, time_limit=None):
    """Fills the exact_* columns of rows where they are missing, from the
    min-sum exact cache (case: local_search job finished before exact)."""
    df = df.copy().reset_index(drop=True)
    missing = df["exact_obj"].isna() & df["exact_status"].isna()
    for idx in df.index[missing]:
        r = df.loc[idx]
        cache_path = default_minsum_cache_path(int(r.n), int(r.k), cache_dir)
        cached = _minsum_cache_lookup(int(r.n), int(r.k), int(r.seed), int(r.instance),
                                      cache_path, time_limit)
        if cached is not None:
            for c, v in exact_from_result(cached, True).items():
                df.at[idx, c] = v
    return add_ratios(df)


def merge_files(paths):
    """Merges files from split jobs: ls_* columns and exact_* columns are
    taken separately then joined (a plain drop_duplicates would overwrite
    one method's columns with the other's NaN)."""
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    ls_cols = ["ls_obj", "ls_time", "ls_n_iters", "ls_S"]
    inst_key = ["n", "k", "seed", "instance"]
    ls_part = (df[df["ls_obj"].notna()][KEY_COLS + ls_cols]
               .drop_duplicates(subset=KEY_COLS, keep="last"))
    ex_part = (df[df["exact_status"].notna()][inst_key + EXACT_COLS]
               .assign(_opt=lambda d: d["exact_optimal"].fillna(False).astype(bool))
               .sort_values("_opt").drop(columns="_opt")
               .drop_duplicates(subset=inst_key, keep="last"))
    keys = df[KEY_COLS].drop_duplicates()
    out = (keys.merge(ls_part, on=KEY_COLS, how="left")
               .merge(ex_part, on=inst_key, how="left"))
    return add_ratios(out).sort_values(KEY_COLS).reset_index(drop=True)


def summarize(df):
    return (df.groupby(["n", "k", "epsilon", "ls_init"])
              .agg(inst=("instance", "count"),
                   n_ls=("ls_obj", lambda s: int(s.notna().sum())),
                   n_opt=("exact_optimal", lambda s: int(s.fillna(False).astype(bool).sum())),
                   t_ls_mean=("ls_time", "mean"),
                   t_ls_max=("ls_time", "max"),
                   t_exact_mean=("exact_time", "mean"),
                   t_exact_max=("exact_time", "max"),
                   t_gurobi_mean=("exact_solver_time", "mean"),
                   ratio_mean=("ratio", "mean"),
                   ratio_max=("ratio", "max"),
                   pct_ls_opt=("ls_is_optimal",
                               lambda s: 100 * s.dropna().astype(float).mean()
                               if s.notna().any() else np.nan),
                   n_bound_fail=("bound_ok",
                                 lambda s: int((s.dropna().astype(float) == 0).sum())))
              .reset_index())


def print_summary(df):
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n" + summarize(df).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    ratio = df["ratio"].dropna()
    n_fail = int((df["bound_ok"].dropna().astype(float) == 0).sum())
    print(f"\n{len(df)} rows, {len(ratio)} with a proven-optimal exact value, "
          f"worst ratio = {fmt(ratio.max() if len(ratio) else np.nan)}, "
          f"{n_fail} violation(s) of the theoretical bound 5/(1-eps).")
    return n_fail


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-values", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--k-values", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--nb-instances", type=int, default=10)
    parser.add_argument("--instance-start", type=int, default=0)
    parser.add_argument("--instance-end", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.0],
                        help="epsilon values for local_search_single_swap (0 = naive).")
    parser.add_argument("--ls-init", default="greedy", choices=["greedy", "random"])
    parser.add_argument("--methods", default="local_search,exact",
                        help="local_search | exact | local_search,exact (default).")
    parser.add_argument("--time-limit", type=float, default=1800.0,
                        help="Gurobi TimeLimit per instance (s).")
    parser.add_argument("--threads", type=int, default=4, help="Gurobi threads.")
    parser.add_argument("--cache-dir", default="results",
                        help="directory of the exact min-sum cache (shared with kmowa.py).")
    parser.add_argument("--output", default="results/ls_vs_gurobi_minsum.parquet")
    parser.add_argument("--report", default=None, metavar="GLOB",
                        help="do not run anything: merge the files matching GLOB, "
                             "re-attach exact values from the cache, write --output, print summary.")
    args = parser.parse_args()

    if args.report is not None:
        out_abs = os.path.abspath(args.output)
        paths = [p for p in sorted(glob.glob(args.report)) if os.path.abspath(p) != out_abs]
        if not paths:
            sys.exit(f"no file matches {args.report!r}")
        print(f"{len(paths)} file(s) merged: " + ", ".join(paths))
        df = attach_exact_from_cache(merge_files(paths), args.cache_dir, args.time_limit)
        _atomic_to_parquet(df, args.output)
        print(f"-> {args.output}")
        print_summary(df)
        sys.exit(0)

    methods = set(args.methods.split(","))
    if not methods or not methods <= set(METHODS):
        parser.error(f"--methods must be a subset of {METHODS}, got {args.methods!r}")

    t0 = time.perf_counter()
    df = run_comparison(
        n_values=args.n_values, k_values=args.k_values,
        nb_instances=args.nb_instances, seed=args.seed,
        epsilons=args.epsilons, ls_init=args.ls_init, methods=methods,
        time_limit=args.time_limit, threads=args.threads,
        cache_dir=args.cache_dir, output_path=args.output,
        instance_start=args.instance_start, instance_end=args.instance_end,
    )
    print(f"\nresults -> {args.output}   (total wall-clock {time.perf_counter() - t0:.1f}s)")
    n_fail = print_summary(df)
    sys.exit(1 if n_fail else 0)
