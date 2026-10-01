"""
test_local_search_kmedian.py

Validates kmowa.local_search_single_swap (Arya et al.'s single-swap local
search) against an EXACT brute-force solver for the PLAIN (sum-aggregated)
k-median problem -- i.e. with uniform OWA weights, W = (1, 1, ..., 1),
not yet the OWA-weighted "ordered k-median" that kmowa_solver targets.

Why test the plain version first: local_search_single_swap's 5-approximation
guarantee (Arya et al., 2004) is proven for the SUM objective. Nothing
guarantees it transfers to a general OWA aggregation -- the project has
already run into a very similar trap once (the Punnen-Aneja early-stop
technique, correct for SUM(k), does NOT transpose automatically to a
general-weight OWA approximation, see recapitulatif_projet.md). Checking
the algorithm's correctness and ratio here, where ground truth is cheap
to compute exactly (brute force over C(n, k) facility subsets), is a
separate question from whether/how it should be reused as a building
block for ordered (OWA) k-median.

Brute force is only tractable for small instances (n, k kept modest
below), which is fine here: this script is a correctness/sanity check,
not a performance benchmark.

Requires: numpy. Does NOT require gurobipy (kmowa_solver / the exact MILP
is intentionally not used here).

Usage:
    python3 test_local_search_kmedian.py
    python3 test_local_search_kmedian.py --n-values 10 15 20 --k-values 2 3 5 \
        --nb-instances 20 --max-ratio 5.0
"""

import argparse
import itertools
import sys
import time

import numpy as np

sys.path.insert(0, ".")  # so `import kmowa` works when run from its own directory
from kmowa import local_search_single_swap
from generate_instances_k_median import random_metric_costs, check_triangle_inequality


# ---------------------------------------------------------------------
# Exact brute-force solver for the PLAIN (sum) k-median problem
# ---------------------------------------------------------------------

def exact_kmedian_brute_force(dist_matrix: np.ndarray, k: int):
    """
    Exact plain k-median by exhaustive search over all C(n, k) facility
    subsets. For a FIXED facility set, the sum objective is minimized by
    assigning each client to its nearest open facility (always optimal
    for SUM, unlike general OWA -- see kmowa.local_search_single_swap's
    module docstring).

    :return: (best_S, best_cost, elapsed) -- best_S a sorted list of
        facility indices, best_cost the corresponding total (summed)
        cost, elapsed the wall-clock time (s) spent in this function.
    """
    start = time.time()
    n_facilities = dist_matrix.shape[1]
    best_cost = np.inf
    best_S = None
    for S in itertools.combinations(range(n_facilities), k):
        cost = dist_matrix[:, list(S)].min(axis=1).sum()
        if cost < best_cost:
            best_cost = float(cost)
            best_S = S
    elapsed = time.time() - start
    return list(best_S), best_cost, elapsed


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def run_validation(n_values, k_values, nb_instances, seed, max_ratio, epsilons):
    print(f"{'n':>4} {'k':>4} {'inst':>4} {'eps':>6} | {'approx':>12} {'exact':>12} "
          f"{'ratio':>8} {'iters':>6} | {'t_approx(ms)':>12} {'t_exact(ms)':>12} | status")
    print("-" * 110)

    worst_ratio = 0.0
    n_checked = 0
    n_failed = 0

    # Per-call timings (seconds), kept separately since exact (brute force)
    # is only solved once per instance while local_search is solved once
    # PER (instance, epsilon) -- mixing them into a single list would
    # overweight whichever one runs more often.
    approx_times = []
    exact_times = []

    total_start = time.time()
    for n in n_values:
        for k in k_values:
            if k >= n:
                continue
            for instance in range(nb_instances):
                C = random_metric_costs(n, low=1, high=10_000, seed=seed + instance * 1000 + n * 17 + k)
                assert check_triangle_inequality(C), (
                    f"generated instance (n={n}, k={k}, instance={instance}) "
                    "does not satisfy the triangle inequality -- "
                    "local_search's ratio guarantee would not hold on it."
                )

                exact_S, exact_cost, exact_elapsed = exact_kmedian_brute_force(C, k)
                exact_times.append(exact_elapsed)

                for eps in epsilons:
                    approx_start = time.time()
                    approx_S, approx_cost, n_iters = local_search_single_swap(
                        C.copy(), k=k, epsilon=eps, init="greedy", seed=0
                    )
                    approx_elapsed = time.time() - approx_start
                    approx_times.append(approx_elapsed)

                    ratio = approx_cost / exact_cost if exact_cost > 0 else 1.0
                    worst_ratio = max(worst_ratio, ratio)
                    n_checked += 1

                    ok = ratio <= max_ratio + 1e-6
                    if not ok:
                        n_failed += 1
                    status = "OK" if ok else f"FAIL (> {max_ratio})"

                    print(f"{n:>4} {k:>4} {instance:>4} {eps:>6.2f} | "
                          f"{approx_cost:>12.1f} {exact_cost:>12.1f} "
                          f"{ratio:>8.4f} {n_iters:>6d} | "
                          f"{approx_elapsed * 1000:>12.3f} {exact_elapsed * 1000:>12.3f} | {status}")

    total_elapsed = time.time() - total_start

    print("-" * 110)
    print(f"{n_checked} checks run, {n_failed} exceeded the theoretical bound "
          f"(ratio <= {max_ratio}), worst observed ratio = {worst_ratio:.4f}")

    approx_times = np.array(approx_times)
    exact_times = np.array(exact_times)
    print(f"\nTiming -- local_search_single_swap ({len(approx_times)} calls): "
          f"total={approx_times.sum():.3f}s  mean={approx_times.mean() * 1000:.3f}ms  "
          f"max={approx_times.max() * 1000:.3f}ms")
    print(f"Timing -- exact_kmedian_brute_force ({len(exact_times)} calls): "
          f"total={exact_times.sum():.3f}s  mean={exact_times.mean() * 1000:.3f}ms  "
          f"max={exact_times.max() * 1000:.3f}ms")
    print(f"Total wall-clock time for this script's validation loop: {total_elapsed:.3f}s")

    if n_failed:
        print("\nFAILURE: local_search_single_swap violated its approximation "
              "guarantee on at least one instance -- investigate before "
              "trusting it as a building block for ordered k-median.")
        sys.exit(1)
    else:
        print("\nSUCCESS: every run stayed within the theoretical approximation "
              "ratio for plain (sum) k-median.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", type=int, nargs="+", default=[8, 10, 12],
                         help="number of points per instance. Kept small: brute "
                              "force is O(C(n, k)) over facility subsets.")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3],
                         help="numbers of facilities to open.")
    parser.add_argument("--nb-instances", type=int, default=10,
                         help="number of random instances per (n, k).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.0],
                         help="values of epsilon to test for local_search_single_swap "
                              "(0.0 = naive/no time guarantee; also try e.g. 0.5 1.0 "
                              "to exercise the polynomial-time threshold variant).")
    parser.add_argument("--max-ratio", type=float, default=5.0 + 1e-6,
                         help="theoretical approximation ratio to check against "
                              "(5.0 for epsilon=0; use 5.0/(1-eps) for the largest "
                              "epsilon tested if you include epsilon > 0).")
    args = parser.parse_args()

    # If any epsilon > 0 is requested, the guaranteed ratio degrades to
    # 5/(1-epsilon): widen max_ratio automatically unless the user already
    # overrode it, so the check stays meaningful instead of silently failing.
    if args.max_ratio == 5.0 + 1e-6 and any(e > 0 for e in args.epsilons):
        worst_eps = max(args.epsilons)
        args.max_ratio = 5.0 / (1.0 - worst_eps) + 1e-6

    run_validation(
        n_values=args.n_values,
        k_values=args.k_values,
        nb_instances=args.nb_instances,
        seed=args.seed,
        max_ratio=args.max_ratio,
        epsilons=args.epsilons,
    )
