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

Time budget (same rule as kmowa.TimeBudget):
    each method gets a TOTAL budget of time_limit x nb_instances seconds,
    counted separately for local_search (per (n, k, epsilon)) and exact
    (per (n, k)). Instances are run in order; once the cumulative time of a
    method exceeds its budget, every remaining instance is SKIPPED for that
    method (ls_skipped / exact_skipped = True, status 'SKIPPED_BUDGET').
    For exact, the time counted is the ORIGINAL Gurobi time, even when the
    result comes from the cache: every job working on the same (n, k)
    therefore skips exactly the same instances, whichever job actually
    called Gurobi, and a resubmitted job rebuilds the same budget.
    --exact-time-limit is the Gurobi TimeLimit of ONE instance (default:
    --time-limit).

Usage:
    # one (n, k, epsilon) triplet -- what each OAR sub-job runs
    python3 compare_local_search_gurobi_kmedian.py --n-values 50 --k-values 5 \
        --epsilons 0.1 --nb-instances 30 --time-limit 1800 \
        --output results/ls_vs_gurobi_minsum_n50_k5_eps0.1.parquet

    # several jobs in parallel on OAR: see submit_array_oar_ls_vs_gurobi_big17.sh

    # merge every job, re-attach exact values from the cache, print the summary
    python3 compare_local_search_gurobi_kmedian.py --report \
        'results/ls_vs_gurobi_minsum_*.parquet' --output ls_vs_gurobi_merged.parquet
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
# Time budget
# ---------------------------------------------------------------------

class TimeBudget:
    """
    Cumulative time per key. A key is stopped for good as soon as its
    cumulative time exceeds limit_seconds * nb_instances. The instance
    that crosses the budget is completed; the following ones are skipped.
    Keys used here: ("local_search", n, k, epsilon) and ("exact", n, k).
    """

    def __init__(self, limit_seconds, nb_instances):
        self.total_budget = float(limit_seconds) * nb_instances
        self.total = {}

    def should_run(self, key):
        return self.total.get(key, 0.0) <= self.total_budget

    def record(self, key, elapsed):
        self.total[key] = self.total.get(key, 0.0) + float(elapsed)

    def used(self, key):
        return self.total.get(key, 0.0)


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
    d["exact_skipped"] = False
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
        "exact_skipped": False,
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
                   methods, time_limit, exact_time_limit, threads, cache_dir,
                   output_path, instance_start=0, instance_end=None):
    """
    time_limit       : average budget per instance (s); total budget of a
                       method = time_limit * nb_instances.
    exact_time_limit : Gurobi TimeLimit of a single instance (s).
    """
    instance_end = nb_instances if instance_end is None else instance_end
    budget = TimeBudget(time_limit, nb_instances)
    rows = []

    print(f"budget per method = {time_limit:.0f}s x {nb_instances} = {budget.total_budget:.0f}s, "
          f"Gurobi TimeLimit per instance = {exact_time_limit:.0f}s")
    print(f"{'n':>4} {'k':>4} {'inst':>4} {'eps':>6} | {'ls_obj':>12} {'exact_obj':>12} "
          f"{'ratio':>8} {'iters':>5} | {'t_ls(ms)':>10} {'t_exact(s)':>10} | status")
    print("-" * 105)

    for n in n_values:
        for k in k_values:
            if not 1 <= k < n:
                print(f"[skip] n={n} k={k}: need 1 <= k < n")
                continue
            cache_path = default_minsum_cache_path(n, k, cache_dir)
            exact_key = ("exact", n, k)

            for instance in range(instance_start, instance_end):
                C = random_metric_costs(n, seed=seed + instance)
                assert check_triangle_inequality(C), (n, k, instance)

                # ---- exact: ONCE per instance, shared by every epsilon ----
                exact = empty_exact()
                if "exact" in methods:
                    if budget.should_run(exact_key):
                        res, from_cache = solve_minsum_exact_cached(
                            C, n, k, seed, instance, cache_path,
                            time_limit=exact_time_limit, threads=threads)
                        exact = exact_from_result(res, from_cache)
                        # ORIGINAL solve time, even from cache (see module docstring)
                        budget.record(exact_key, res["wall_time"])
                    else:
                        exact["exact_skipped"] = True
                        exact["exact_status"] = "SKIPPED_BUDGET"
                else:
                    # local_search-only job: reuse an exact result if another
                    # job already put it in the cache, never call Gurobi
                    cached = _minsum_cache_lookup(n, k, seed, instance, cache_path,
                                                  exact_time_limit)
                    if cached is not None:
                        exact = exact_from_result(cached, True)

                # ---- local search: once per epsilon ----
                for eps in epsilons:
                    ls_key = ("local_search", n, k, float(eps))
                    row = {"n": n, "k": k, "seed": seed, "instance": instance,
                           "epsilon": float(eps), "ls_init": ls_init,
                           "ls_obj": np.nan, "ls_time": np.nan,
                           "ls_n_iters": np.nan, "ls_S": None, "ls_skipped": False}
                    if "local_search" in methods:
                        if budget.should_run(ls_key):
                            t0 = time.perf_counter()
                            S, obj, n_iters = local_search_single_swap(
                                C, k=k, epsilon=eps, init=ls_init, seed=seed + instance)
                            elapsed = time.perf_counter() - t0
                            budget.record(ls_key, elapsed)
                            row.update({"ls_obj": float(obj), "ls_time": elapsed,
                                        "ls_n_iters": n_iters,
                                        "ls_S": [int(s) for s in S]})
                        else:
                            row["ls_skipped"] = True
                    row.update(exact)
                    row = add_ratios(pd.DataFrame([row])).iloc[0].to_dict()
                    rows.append(row)
                    upsert(row, output_path)

                    if row["ls_skipped"] or row["exact_skipped"]:
                        status = "SKIPPED (budget): " + ", ".join(
                            m for m, s in (("local_search", row["ls_skipped"]),
                                           ("exact", row["exact_skipped"])) if s)
                    elif row["bound_ok"] is None or (isinstance(row["bound_ok"], float)
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

            # end of (n, k): budget report
            for key, used in budget.total.items():
                if key[1:3] == (n, k):
                    state = "STOPPED" if not budget.should_run(key) else "ok"
                    print(f"[budget] {key}: {used:.1f}s / {budget.total_budget:.1f}s ({state})")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Report: re-attach exact results from the cache + summary
# ---------------------------------------------------------------------

def attach_exact_from_cache(df, cache_dir, time_limit=None):
    """Fills the exact_* columns of rows where they are missing, from the
    min-sum exact cache (case: local_search job finished before exact)."""
    df = df.copy().reset_index(drop=True)
    if "exact_skipped" not in df:
        df["exact_skipped"] = False
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
    for c in ("ls_skipped", "exact_skipped"):
        if c not in df:
            df[c] = False
    df["ls_skipped"] = df["ls_skipped"].fillna(False).astype(bool)
    df["exact_skipped"] = df["exact_skipped"].fillna(False).astype(bool)
    ls_cols = ["ls_obj", "ls_time", "ls_n_iters", "ls_S", "ls_skipped"]
    inst_key = ["n", "k", "seed", "instance"]
    # a computed LS value wins over a 'skipped' row for the same key
    ls_part = (df[df["ls_obj"].notna() | df["ls_skipped"]][KEY_COLS + ls_cols]
               .assign(_p=lambda d: d["ls_obj"].notna().astype(int))
               .sort_values("_p").drop(columns="_p")
               .drop_duplicates(subset=KEY_COLS, keep="last"))
    # priority: skipped < solved but not optimal < optimal
    ex_part = (df[df["exact_status"].notna()][inst_key + EXACT_COLS + ["exact_skipped"]]
               .assign(_p=lambda d: (~d["exact_skipped"]).astype(int)
                       + d["exact_optimal"].fillna(False).astype(bool).astype(int))
               .sort_values("_p").drop(columns="_p")
               .drop_duplicates(subset=inst_key, keep="last"))
    keys = df[KEY_COLS].drop_duplicates()
    out = (keys.merge(ls_part, on=KEY_COLS, how="left")
               .merge(ex_part, on=inst_key, how="left"))
    out["ls_skipped"] = out["ls_skipped"].fillna(False).astype(bool)
    out["exact_skipped"] = out["exact_skipped"].fillna(False).astype(bool)
    return add_ratios(out).sort_values(KEY_COLS).reset_index(drop=True)


def summarize(df):
    return (df.groupby(["n", "k", "epsilon", "ls_init"])
              .agg(inst=("instance", "count"),
                   n_ls=("ls_obj", lambda s: int(s.notna().sum())),
                   n_ls_skip=("ls_skipped", lambda s: int(s.fillna(False).astype(bool).sum())),
                   n_opt=("exact_optimal", lambda s: int(s.fillna(False).astype(bool).sum())),
                   n_ex_skip=("exact_skipped", lambda s: int(s.fillna(False).astype(bool).sum())),
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
                        help="average time budget per instance (s): each method stops "
                             "once its cumulative time exceeds time_limit x nb_instances.")
    parser.add_argument("--exact-time-limit", type=float, default=None,
                        help="Gurobi TimeLimit of ONE instance (s). Default: --time-limit.")
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
        etl = args.exact_time_limit if args.exact_time_limit is not None else args.time_limit
        df = attach_exact_from_cache(merge_files(paths), args.cache_dir, etl)
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
        time_limit=args.time_limit,
        exact_time_limit=(args.exact_time_limit if args.exact_time_limit is not None
                          else args.time_limit),
        threads=args.threads,
        cache_dir=args.cache_dir, output_path=args.output,
        instance_start=args.instance_start, instance_end=args.instance_end,
    )
    print(f"\nresults -> {args.output}   (total wall-clock {time.perf_counter() - t0:.1f}s)")
    n_fail = print_summary(df)
    sys.exit(1 if n_fail else 0)
