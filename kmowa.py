import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import os
import time
import sys
import fcntl
import traceback



#####################################################################
################ HELPER FUNCTIONS FOR THE ALGORITHMS ################
#####################################################################


def owa_cost(costs, W):
    """
    Computes the OWA-aggregated value of a 1D vector of per-client costs.
    :param costs: 1D array of length n (one cost per client, e.g. the
        distance from each client to its assigned open facility).
    :param W: 1D array of length n, OWA weights.
    :return: float, the weighted sum of the sorted costs.
    """
    return float(np.dot(W, np.sort(np.asarray(costs))[::-1]))



def costs_from_open_facilities(dist_matrix, S):
    """
    Given a (n_clients, n_facilities) distance matrix and a set S of open
    facility indices, returns the 1D vector of per-client costs under
    nearest-open-facility assignment.
    """
    return dist_matrix[:, list(S)].min(axis=1)



#####################################################################
###################### ALGORITHMS IMPLEMENTATION ####################
#####################################################################

# ---------------------------------------------------------------------
# Local Search Algorithm for k-median (Arya et al., 2004) 
# Version efficace (threshold) : swap simple (p=1), ratio 5
# (or 5/(1-epsilon) with the polynomial-time threshold variant).
# ---------------------------------------------------------------------
 
def _min1_min2_argmin(D_S: np.ndarray):
    """
    D_S : (n_clients, k) distances from the clients to the 
    currently open facilities (in the same order as S).
    
    Returns, for each client:
      - min1     : distance to the nearest open facility
      - min2     : distance to the second nearest open facility
      - argmin1  : position (0..k-1, index in S) to the nearest facility
    """
    argmin1 = np.argmin(D_S, axis=1)
    min1 = D_S[np.arange(D_S.shape[0]), argmin1]
    D_masked = D_S.copy()
    D_masked[np.arange(D_S.shape[0]), argmin1] = np.inf
    min2 = D_masked.min(axis=1)
    return min1, min2, argmin1
 
 

def _forward_greedy_init(dist_matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Greedy initialization before the local search: add at each step the facility 
    that decreases the most the total cost. No theoretical guarantee, but is in practice
    a better starting point than a random sampling..
    Complexite : O(k * n_clients * n_facilites), done ONCE.
    """
    n_clients, n_facilities = dist_matrix.shape
    chosen = []
    current_min = np.full(n_clients, np.inf)  # distance to the nearest open facility (np.inf for now)
    remaining = set(range(n_facilities))
 
    for _ in range(k): # choose each facility one by one
        best_f, best_cost = None, np.inf
        for f in remaining:
            new_min = np.minimum(current_min, dist_matrix[:, f])
            c = new_min.sum()
            if c < best_cost:
                best_cost, best_f = c, f
        if best_f is not None:
            chosen.append(best_f)
            remaining.discard(best_f)
            current_min = np.minimum(current_min, dist_matrix[:, best_f])
        else:
            break  # no more facilities to choose from
 
    return np.array(chosen)
 

 
def local_search_single_swap(
    dist_matrix: np.ndarray,
    k: int,
    epsilon: float = 0.0,
    init: str = "greedy",
    seed: int = 0,
    max_iters: int | None = None,
    tol: float = 1e-6
):
    """
    Local search from Arya et al., simple swap (ratio 5, or 5/(1-epsilon) with the threshold).
    Parameters
    ----------
    dist_matrix : (n_clients, n_facilities) ndarray
        Distance/cost matrix -- MUST verify the triangular inequality for the ratio to be guaranteed. 
    k : int
        Number of facilities to open.
    epsilon : float, default 0.0
        0 => no time guarantee (naive algorithm).
        >0 => execution in polynomial time guaranteed, degraded ratio to 5/(1-eps).
    init : {"greedy", "random"}
        Initialization strategy.
    seed : int
        Seed for random initialization (ignored if init="greedy")
        Graine pour l'initialisation aleatoire (ignoree si init="greedy").
    max_iters : int or None
        Optional (useful if epsilon=0).
 
    Returns
    -------
    S : list[int]
        Indices of open facilities (size k).
    cost : float
        Total cost of returned solution.
    n_iters : int
        Number of iterations.
    """
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    n_clients, n_facilities = dist_matrix.shape
    if k >= n_facilities:
        raise ValueError("k must be less than the number of possible facilities")
 
    if init == "greedy":
        S = _forward_greedy_init(dist_matrix, k)
    else:
        rng = np.random.default_rng(seed)
        S = rng.choice(n_facilities, size=k, replace=False)
 
    S_set = set(S.tolist())
    Q = k  # |Q| in Arya et al.'s proof (section 3.2)
 
    def total_cost(S_arr):
        return float(dist_matrix[:, S_arr].min(axis=1).sum())
 
    cost_S = total_cost(S)

    ratio = (1.0 - epsilon / Q) if epsilon > 0 else None
 
    it = 0
    while max_iters is None or it < max_iters:
        it += 1
        D_S = dist_matrix[:, S]  # (n_clients, k)
        min1, min2, argmin1 = _min1_min2_argmin(D_S)
 
        outside = np.array([f for f in range(n_facilities) if f not in S_set])
        if outside.size == 0:
            break
 
        best_cost = cost_S
        best_swap = None  

        # Find the best swap possible (compute best_swap for each facility in S and keep the best among all facilities)
        for pos in range(k):
            # Fallback cost for each client if S[pos] is removed
            fallback = np.where(argmin1 == pos, min2, min1) # costs of second best solution in S if first one is pos, otherwise best cost in S
            D_out = dist_matrix[:, outside]  # (n_clients, n_outside) only keeps distances from the clients to facilities not in S
            candidate_costs = np.minimum(D_out, fallback[:, None]).sum(axis=0) # candidate_costs[j, f] is the min(min distance from client j to a new facility outside of S, distance from client j to best facility in S\{pos})
            # Thus, candidate_costs has shape (n_outside,) and keeps the total cost of solution S - pos + f
            j = int(np.argmin(candidate_costs)) # Choose the facility that minimizes the cost of the swap
            if candidate_costs[j] < best_cost:
                best_cost = float(candidate_costs[j])
                best_swap = (pos, int(outside[j]))
 
        if best_swap is None:
            break  # local optimum reached

        if ratio is None:
            if best_cost >= cost_S:
                break  # no improvement, stopping

        elif best_cost > ratio * cost_S + tol:
            break  # insufficient amelioration, stopping to respect the time guarantee
 
        # Apply the swap
        pos, f_new = best_swap
        s_old = int(S[pos])
        S[pos] = f_new
        S_set.discard(s_old)
        S_set.add(f_new)
 
        cost_S = best_cost
 
    return sorted(S.tolist()), cost_S, it



#####################################################################
################ MIN-SUM K-MEDIAN : EXACT SOLVER (GUROBI) ###########
#####################################################################

def _gurobi_status_name(code):
    """Readable name of a Gurobi status code (e.g. 2 -> 'OPTIMAL')."""
    for name in dir(GRB.Status):
        if not name.startswith("_") and getattr(GRB.Status, name) == code:
            return name
    return str(code)


def kmedian_minsum_solver(dist_matrix, k, time_limit=None, threads=4,
                          mem_limit=None, start_S=None, verbose=False):
    """
    Exact MILP for the classical (min-sum) k-median problem.

    Convention (SAME as local_search_single_swap):
        dist_matrix[i, j] = cost of serving CLIENT i from FACILITY j
    (rectangular matrices are accepted).

        min  sum_{i,j} d_ij x_ij
        s.t. sum_j x_ij = 1      for every client i
             x_ij <= y_j         for every (i, j)   (strong formulation)
             sum_j y_j = k
             y_j in {0,1},  0 <= x_ij <= 1

    x is CONTINUOUS: once y is fixed, assigning each client to its nearest
    open facility is optimal, so x is integral at the optimum anyway --
    n_facilities binaries instead of n_clients * n_facilities.

    :param time_limit: Gurobi TimeLimit (s), None = no limit.
    :param start_S: optional MIP start (list of k facility indices). Do NOT
        use it when measuring exact solve times (it biases them).
    :return: dict, ALWAYS (never None):
        status (str), optimal (bool), obj (float, NaN if no solution),
        bound (best lower bound), gap, S (sorted list of open facilities
        or None), solver_time (Gurobi Runtime), build_time, wall_time.
        'obj' is RECOMPUTED from S with the same formula as the local
        search, so the ratios ls/exact are not polluted by solver tolerances.
    """
    D = np.asarray(dist_matrix, dtype=float)
    n_clients, n_fac = D.shape
    if not 1 <= k <= n_fac:
        raise ValueError(f"k={k} must be in [1, {n_fac}]")

    t0 = time.perf_counter()
    params = {"OutputFlag": 1 if verbose else 0, "Threads": threads}
    if time_limit is not None:
        params["TimeLimit"] = float(time_limit)
    if mem_limit is not None:
        params["MemLimit"] = mem_limit

    # Context managers: model AND environment are disposed even if an
    # exception is raised (no more m.dispose() / gc.collect() by hand).
    with gp.Env(params=params) as env, gp.Model("min-sum k-median", env=env) as m:
        y = m.addMVar(n_fac, vtype=GRB.BINARY, name="y")
        x = m.addMVar((n_clients, n_fac), lb=0.0, ub=1.0, name="x")

        m.addConstr(x.sum(axis=1) == 1, name="assign")
        m.addConstr(x <= y, name="open")          # broadcast over the rows of x
        m.addConstr(y.sum() == k, name="k_facilities")
        m.setObjective((x * D).sum(), GRB.MINIMIZE)

        if start_S is not None:
            y.Start = np.isin(np.arange(n_fac), list(start_S)).astype(float)

        build_time = time.perf_counter() - t0
        m.optimize()

        status = _gurobi_status_name(m.Status)
        optimal = m.Status == GRB.OPTIMAL
        S, obj = None, np.nan
        if m.SolCount > 0:
            S = sorted(np.flatnonzero(y.X > 0.5).tolist())
            obj = float(D[:, S].min(axis=1).sum())
        bound = float(m.ObjBound) if m.SolCount > 0 or optimal else np.nan
        gap = float(m.MIPGap) if m.SolCount > 0 else np.nan
        solver_time = float(m.Runtime)

    return {
        "status": status,
        "optimal": bool(optimal),
        "obj": obj,
        "bound": bound,
        "gap": gap,
        "S": S,
        "solver_time": solver_time,
        "build_time": build_time,
        "wall_time": time.perf_counter() - t0,
    }


def kmedian_minsum_bruteforce(dist_matrix, k):
    """Enumerates every set of k facilities. ONLY for tests (n <= ~15)."""
    from itertools import combinations
    D = np.asarray(dist_matrix, dtype=float)
    best_S, best = None, np.inf
    for S in combinations(range(D.shape[1]), k):
        c = D[:, S].min(axis=1).sum()
        if c < best:
            best, best_S = c, list(S)
    return best_S, float(best)


def self_test_minsum(nb_tests=20, n_max=10, seed=0):
    """
    Sanity check: kmedian_minsum_solver == brute force, and
    local_search_single_swap >= optimum, on small random metric instances.
    Run it once (python3 kmowa.py --self-test) before launching the cluster jobs.
    """
    rng = np.random.default_rng(seed)
    for t in range(nb_tests):
        n = int(rng.integers(4, n_max + 1))
        k = int(rng.integers(1, n))
        P = rng.uniform(0, 100, size=(n, 2))
        D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        _, bf = kmedian_minsum_bruteforce(D, k)
        res = kmedian_minsum_solver(D, k, threads=1)
        _, ls, _ = local_search_single_swap(D, k)
        assert res["optimal"] and abs(res["obj"] - bf) <= 1e-6 * max(1.0, bf), (t, n, k, res, bf)
        assert ls >= bf - 1e-6 * max(1.0, bf), (t, n, k, ls, bf)
    print(f"[self-test] OK: {nb_tests} instances, exact == brute force, LS >= optimum.")


#####################################################################
############## MIN-SUM K-MEDIAN : CACHE OF EXACT RESULTS ############
#####################################################################

def _atomic_to_parquet(df, path):
    """Writes df to path atomically (temp file + os.replace): a job killed
    by the OAR walltime mid-write can no longer leave a corrupted file."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp_path = f"{path}.tmp{os.getpid()}"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


# The min-sum problem does NOT depend on the OWA weights: no w_type in the
# key nor in the file name (the instances are generated independently of
# w_type anyway, see random_metric_costs(n, seed=seed + i)).
MINSUM_CACHE_KEY_COLS = ["n", "k", "seed", "instance"]


def default_minsum_cache_path(n, k, cache_dir="results"):
    return os.path.join(cache_dir, f"exact_minsum_cache_n{n}_k{k}.parquet")


def _minsum_cache_lookup(n, k, seed, instance, cache_path, time_limit=None):
    """
    Returns the cached exact result (dict) or None. A NON-optimal cached
    result (time limit reached) is only reused if it was obtained with a
    time limit >= the current one -- otherwise we retry with the larger
    budget.
    """
    if not os.path.exists(cache_path):
        return None
    try:
        df = pd.read_parquet(cache_path)
    except Exception:
        return None
    hit = df[(df["n"] == n) & (df["k"] == k) & (df["seed"] == seed) & (df["instance"] == instance)]
    if hit.empty:
        return None
    r = hit.iloc[-1].to_dict()
    if not r["optimal"]:
        cached_tl = r.get("time_limit")
        cached_tl = np.inf if _is_missing(cached_tl) else cached_tl
        wanted_tl = np.inf if time_limit is None else time_limit
        if cached_tl < wanted_tl:
            return None
    if r.get("S") is not None and not _is_missing(r.get("S")):
        r["S"] = [int(s) for s in r["S"]]
    return r


def _minsum_cache_store(n, k, seed, instance, res, time_limit, cache_path):
    row = {
        "n": n, "k": k, "seed": seed, "instance": instance,
        "obj": res["obj"], "bound": res["bound"], "gap": res["gap"],
        "optimal": res["optimal"], "status": res["status"],
        "S": res["S"],
        "solver_time": res["solver_time"], "build_time": res["build_time"],
        "wall_time": res["wall_time"],
        "time_limit": np.nan if time_limit is None else float(time_limit),
    }
    df_new = pd.DataFrame([row])
    df_old = None
    if os.path.exists(cache_path):
        try:
            df_old = pd.read_parquet(cache_path)
        except Exception as e:
            backup = f"{cache_path}.corrupted.{int(time.time())}"
            print(f"[WARNING] unreadable min-sum cache ({e}) -- moved to {backup}.")
            try:
                os.replace(cache_path, backup)
            except OSError:
                pass
    if df_old is not None:
        df_new = (pd.concat([df_old, df_new], ignore_index=True)
                  .drop_duplicates(subset=MINSUM_CACHE_KEY_COLS, keep="last")
                  .sort_values(MINSUM_CACHE_KEY_COLS)
                  .reset_index(drop=True))
    _atomic_to_parquet(df_new, cache_path)


def solve_minsum_exact_cached(cost_matrix, n, k, seed, instance, cache_path,
                              time_limit=None, threads=4):
    """
    Returns (res, from_cache). res has the keys of kmedian_minsum_solver.
    The solve times stored in the cache are the ORIGINAL ones: reading the
    cache gives back the real Gurobi time, not 0 (otherwise every average
    time would be biased downwards as soon as a result comes from cache).
    Same exclusive flock as solve_exact_cached (no double Gurobi run when
    two jobs ask for the same key at the same time).
    """
    d = os.path.dirname(cache_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(cache_path + ".lock", "a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            cached = _minsum_cache_lookup(n, k, seed, instance, cache_path, time_limit)
            if cached is not None:
                return cached, True
            res = kmedian_minsum_solver(cost_matrix, k, time_limit=time_limit, threads=threads)
            _minsum_cache_store(n, k, seed, instance, res, time_limit, cache_path)
            return res, False
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)




def kmowa_solver(cost_matrix, k, W):
    """
    Exact MILP formulation of the OWA-weighted k-median problem, following
    the dual-based OWA linearization of Chassein & Goerigk (2015).
    :param cost_matrix: (n, n) distance matrix (clients = potential facility sites).
    :param k: number of facilities to open.
    :param W: OWA weight vector of length n.
    :return: dict {"obj value", "x", "y", "z"}, or None if the solver
        did not reach an optimal solution (time/memory limit).
    """
    costs = cost_matrix.copy()
    n = np.shape(costs)[0]

    m = gp.Model("OWA k-median")

    m.setParam("Method", 1)     
    m.setParam("Presolve", 0)
    m.setParam("Crossover", 0)
    m.setParam("NumericFocus", 3)
    m.setParam("OutputFlag", 0)
    m.setParam("Threads", 4)
    m.setParam("MemLimit", 12000)
 
    # Add variables
    alpha = m.addMVar(shape=n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="alpha")
    beta = m.addMVar(shape=n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="beta")
    x = m.addMVar(shape=(n, n), lb=0.0, vtype=GRB.BINARY, name="x")
    y = m.addMVar(shape=n, lb=0.0, vtype=GRB.BINARY, name="y")
    z = m.addMVar(n, lb=0.0, name="z")
    
    # Add constraints
    # Constraints
    for j in range(n):
        m.addConstr(z[j] == gp.quicksum(costs[j, i] * x[j, i] for i in range(n)), name=f"linearization_z_{j}")

    for i in range(n):
        for j in range(n):
            m.addConstr(alpha[i] + beta[j] - W[j] * z[i] >= 0)

    for j in range(n):
        m.addConstr(gp.quicksum(x[j, i] for i in range(n)) == 1, name=f"cover_{j}")

    for i in range(n):
        for j in range(n):
            m.addConstr(x[j, i] <= y[i], name=f"open_city_{j}_only_if_{i}")

    m.addConstr(gp.quicksum(y[i] for i in range(n)) <= k, name="k_facilities")

    # Objective function
    m.setObjective(gp.quicksum(alpha[i] + beta[i] for i in range(n)), GRB.MINIMIZE)

    m.update()
    m.optimize()

    if m.Status in [GRB.TIME_LIMIT, GRB.INTERRUPTED]:
        print("Time or memory limit reached.")
        return None

    if m.Status != GRB.OPTIMAL:
        print("not optimal")
        return None

    obj = m.ObjVal
    x_sol = x.X.reshape((n, n)).astype(int)
    y_sol = y.X.astype(int)
    z_sol = z.X.astype(np.float64)
    
    m.dispose()
    del m
    import gc; gc.collect()

    return {"obj value" : obj,
            "x" : x_sol,    
            "y" : y_sol,
            "z" : z_sol
            }






#####################################################################
######################### TESTING AND EXPORT ########################
#####################################################################
 
# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
 
def _solution_to_vector(x: np.ndarray) -> np.ndarray:
    """
    Convert a solution to a 1D vector. Accepts either a 1D vector
    (returned as-is) or an n x n assignment matrix (argmax along rows).
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x.astype(int)
    if x.ndim == 2:
        return np.argmax(x, axis=1).astype(int)
    raise ValueError(f"Unexpected solution dimension: {x.ndim}")
 
 
def _is_missing(value) -> bool:
    """
    Scalar-and-array-aware missing value check. pd.isna() raises on
    numpy arrays, so they are handled separately.
    """
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
 
 
# -------------------------------------------------------------------
# Parquet upsert (called after every single instance)
# -------------------------------------------------------------------
 
# Default upsert key for this file's results: (n, k, epsilon, seed, instance) 
# -- two different k values on the same n are unrelated problems and must never be confused.
KEY_COLS = ["n", "k", "epsilon", "seed", "instance"]
 
 
def _upsert_parquet(row: dict, parquet_path: str, key_cols=None) -> None:
    """
    Append a single result row to the method-specific parquet file. If a
    row with the same key (key_cols, default KEY_COLS = n, k, seed,
    instance) already exists, it is replaced by the new one (keep last).
    """
    key_cols = KEY_COLS if key_cols is None else key_cols
    df_new = pd.DataFrame([row])
 
    if os.path.exists(parquet_path):
        df_old = pd.read_parquet(parquet_path)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined = (
            df_combined
            .drop_duplicates(subset=key_cols, keep="last")
            .sort_values(by=key_cols)
            .reset_index(drop=True)
        )
    else:
        df_combined = df_new
 
    df_combined.to_parquet(parquet_path, index=False)
 
 
# -------------------------------------------------------------------
# Cache for "exact" results (Gurobi)
# -------------------------------------------------------------------
 
# k is now part of the cache key (see KEY_COLS note above: two
# different k are two different problems for the same instance).
EXACT_CACHE_KEY_COLS = ["w_type", "n", "k", "seed", "instance"]
 
 
def default_exact_cache_path(w_type, n, k, cache_dir="results"):
    """
    Default path of the exact cache, ONE FILE PER (w_type, n, k) -- k is
    included in the filename as well as in the cache key, since two
    different k values on the same n are unrelated problems and must
    never be confused.
    """
    return os.path.join(cache_dir, f"exact_cache_{w_type}_n{n}_k{k}.parquet")
 
 
def _exact_cache_lookup(w_type, n, k, seed, instance, cache_path):
    """Returns {"exact obj value": ..., "x": ...} if an exact result
    already exists in the cache for this key, else None."""
    if not os.path.exists(cache_path):
        return None
    try:
        df = pd.read_parquet(cache_path)
    except Exception:
        return None
    if df.empty:
        return None
    mask = (
        (df["w_type"] == w_type)
        & (df["n"] == n)
        & (df["k"] == k)
        & (df["seed"] == seed)
        & (df["instance"] == instance)
    )
    hit = df.loc[mask]
    if hit.empty:
        return None
    last = hit.iloc[-1]
    return {"exact obj value": last["exact_obj"], "x": last["exact_sol"]}
 
 
def _exact_cache_store(w_type, n, k, seed, instance, obj_value, sol_vector, cache_path):
    row = {
        "w_type": w_type,
        "n": n,
        "k": k,
        "seed": seed,
        "instance": instance,
        "exact_obj": obj_value,
        "exact_sol": np.asarray(sol_vector),
    }
    df_new = pd.DataFrame([row])
 
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
 
    df_old = None
    if os.path.exists(cache_path):
        try:
            df_old = pd.read_parquet(cache_path)
        except Exception as e:
            # Corrupted cache -- typically a previous job killed by the
            # OAR walltime mid-write, BEFORE the atomic write below. Do
            # not crash: move the unreadable file aside (for later
            # inspection) and start over with just the new row. Old
            # entries from the corrupted file are lost, but will simply
            # be recomputed next time since _exact_cache_lookup will no
            # longer find them.
            backup_path = f"{cache_path}.corrupted.{int(time.time())}"
            print(f"[WARNING] unreadable exact cache ({e}) -- "
                  f"moved to {backup_path} and reinitialised.")
            try:
                os.replace(cache_path, backup_path)
            except OSError:
                pass
            df_old = None
 
    if df_old is not None:
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined = (
            df_combined
            .drop_duplicates(subset=EXACT_CACHE_KEY_COLS, keep="last")
            .sort_values(by=EXACT_CACHE_KEY_COLS)
            .reset_index(drop=True)
        )
    else:
        df_combined = df_new
 
    # Atomic write: write to a temp file, then rename over cache_path
    # (os.replace is atomic on the same POSIX filesystem). If THIS
    # process is killed during to_parquet, only the .tmp file is
    # incomplete -- cache_path itself stays intact.
    tmp_path = f"{cache_path}.tmp{os.getpid()}"
    df_combined.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, cache_path)
 
 
def has_cached_exact(w_type, n, k, seed, instance, cache_path):
    """True if an exact result already exists in the cache for this key
    (w_type, n, k, seed, instance) -- lets the caller reuse the cache
    even when the 'exact' method's time budget is already exhausted
    (reusing it costs nothing)."""
    return _exact_cache_lookup(w_type, n, k, seed, instance, cache_path) is not None
 
 
def _exact_cache_lock_path(cache_path):
    return cache_path + ".lock"
 
 
def solve_exact_cached(cost_matrix, W, w_type, n, k, seed, instance, cache_path):
    """
    Returns (obj_value, sol_vector, elapsed, from_cache).
 
    If the exact result for (w_type, n, k, seed, instance) is already in
    the cache, it is reused as-is (elapsed=0.0, from_cache=True) -- Gurobi
    is NOT re-run. Otherwise kmowa_solver is called normally, the result
    is stored in the cache, then returned.
 
    If the solver fails (res is None, e.g. Gurobi hit a time/memory
    limit), nothing is cached and (None, None, elapsed, False) is
    returned.
 
    LOCKING: the whole "check cache -> compute if missing -> store" block
    is protected by an EXCLUSIVE file lock (fcntl.flock, blocking) on
    cache_path + '.lock'. Without it, two processes requesting the exact
    result for the SAME key AT THE SAME TIME could both find the cache
    empty and both launch Gurobi -- no file corruption
    (drop_duplicates(keep='last') absorbs that), but the computation
    would effectively be done twice, exactly what this is meant to
    avoid. With the lock, the second process waits for the first one to
    finish writing to the cache, then reads its result back instead of
    re-solving the MILP.
 
    CAUTION: fcntl.flock assumes working file locking on the filesystem
    shared between cluster nodes (NFS with lockd active, or a local
    filesystem). If your OAR jobs run on different nodes with NFS
    locking disabled/unreliable, the lock may not prevent a cross-node
    race (it remains effective within a single node).
    """
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
 
    lock_path = _exact_cache_lock_path(cache_path)
 
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)  # blocking: waits if another process holds the lock
        try:
            cached = _exact_cache_lookup(w_type, n, k, seed, instance, cache_path)
            if cached is not None:
                return (
                    cached["exact obj value"],
                    _solution_to_vector(cached["x"]),
                    0.0,
                    True,
                )
 
            start = time.time()
            res = kmowa_solver(cost_matrix, k=k, W=W)
            elapsed = time.time() - start
 
            if res is None:
                return None, None, elapsed, False
 
            obj_value = res["obj value"]
            sol_vector = _solution_to_vector(res["x"])
            _exact_cache_store(w_type, n, k, seed, instance, obj_value, sol_vector, cache_path)
            return obj_value, sol_vector, elapsed, False
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
 
 
# -------------------------------------------------------------------
# Time budget (same permanent-stop design as elicitation.py's TimeBudget)
# -------------------------------------------------------------------
 
class TimeBudget:
    """Tracks CUMULATIVE time per (method, n, k) and stops that method
    for good once the cumulative time exceeds nb_instances * limit_seconds."""
 
    def __init__(self, limit_seconds=1800.0, nb_instances=30):
        self.limit = limit_seconds
        self.nb_instances = nb_instances
        self.total_time = {}
        self.count = {}
        self.stopped = set()
 
    @property
    def total_budget(self):
        return self.limit * self.nb_instances
 
    def should_run(self, method, n, k):
        return (method, n, k) not in self.stopped
 
    def is_stopped(self, method, n, k):
        return (method, n, k) in self.stopped
 
    def record(self, method, n, k, elapsed):
        key = (method, n, k)
        self.total_time[key] = self.total_time.get(key, 0.0) + elapsed
        self.count[key] = self.count.get(key, 0) + 1
        if self.total_time[key] > self.total_budget:
            self.stopped.add(key)
 
    def current_total(self, method, n, k):
        return self.total_time.get((method, n, k), 0.0)
 
 
# -------------------------------------------------------------------
# Main comparison function: local_search_single_swap vs exact (kmowa_solver)
# -------------------------------------------------------------------
 
def run_one(cost_matrix, W, n, k, instance, seed, budget: TimeBudget, w_type,
            methods=None, exact_cache_path=None, ls_epsilon=0.0, ls_init="greedy"):
    """
    Runs the requested methods on a single instance and returns one
    result row. methods : subset of {'min_sum_k_median', 'exact'}. None = both.
 
    Note: min_sum_k_median (the thresholding heuristic built on top of
    local_search_single_swap, mirroring k_fowama.min_lorenz for the
    assignment problem) targets the OWA objective directly, but is NOT
    proven exact for general OWA weights -- like min_lorenz, it can land
    a few percent above the true optimum on some instances (verified: up
    to ~8% on small random instances). Comparing it against kmowa_solver
    (exact) is exactly how to measure that gap for your own instances.
    """
    if methods is None:
        methods = {"min_sum_k_median", "exact"}
 
    row = {"n": n, "k": k, "epsilon": ls_epsilon, "instance": instance, "seed": seed}
 
    # --- min_sum_k_median (OWA k-median heuristic via thresholding) ---
    if "min_sum_k_median" not in methods:
        row["min_sum_k_median_time"] = np.nan
        row["min_sum_k_median_owa_obj"] = np.nan
        row["min_sum_k_median_nb_sols"] = np.nan
    elif budget.should_run("min_sum_k_median", n, k):
        start = time.time()
        S, nb_sols = min_sum_k_median(cost_matrix, k, W, epsilon=ls_epsilon, init=ls_init, seed=seed)
        elapsed = time.time() - start
        budget.record("min_sum_k_median", n, k, elapsed)
        costs_S = costs_from_open_facilities(cost_matrix, S)
        row["min_sum_k_median_time"] = elapsed
        row["min_sum_k_median_owa_obj"] = owa_cost(costs_S, W)
        row["min_sum_k_median_nb_sols"] = nb_sols
    else:
        row["min_sum_k_median_time"] = np.nan
        row["min_sum_k_median_owa_obj"] = np.nan
        row["min_sum_k_median_nb_sols"] = np.nan
 
    # --- exact (Gurobi, OWA-optimal, with shared cache) ---
    if "exact" not in methods:
        row["exact_time"] = np.nan
        row["exact_owa_obj"] = np.nan
    else:
        cache_path = exact_cache_path or default_exact_cache_path(w_type, n, k)
        if budget.should_run("exact", n, k) or has_cached_exact(w_type, n, k, seed, instance, cache_path):
            obj_value, sol_vector, elapsed, from_cache = solve_exact_cached(
                cost_matrix, W, w_type, n, k, seed, instance, cache_path
            )
            if not from_cache:
                budget.record("exact", n, k, elapsed)
            row["exact_time"] = 0.0 if from_cache else elapsed
            row["exact_owa_obj"] = obj_value if obj_value is not None else np.nan
        else:
            row["exact_time"] = np.nan
            row["exact_owa_obj"] = np.nan
 
    # --- ratio (OWA objective, min_sum_k_median vs exact) ---
    if not np.isnan(row["exact_owa_obj"]) and row["exact_owa_obj"] > 0 and not np.isnan(row["min_sum_k_median_owa_obj"]):
        row["ratio_min_sum_k_median"] = row["min_sum_k_median_owa_obj"] / row["exact_owa_obj"]
    else:
        row["ratio_min_sum_k_median"] = np.nan
 
    return row


 
 
def run_comparison(
    n_values,
    w_type,
    k_values,
    nb_instances=30,
    seed=0,
    time_limit=1800.0,
    output_path=None,
    instance_start=0,
    instance_end=None,
    methods=None,
    exact_cache_dir="results",
    epsilon_values=(0.0,),
    ls_init="greedy",
):
    """
    Compares min_sum_k_median and kmowa_solver (exact) over a grid of
    metric k-median instances swept across n_values x k_values x
    epsilon_values, one row per (n, k, epsilon, instance), upserted into
    a parquet file after each instance (same pattern as
    compare_min_lorenz.run_comparison).
 
    epsilon_values only affects min_sum_k_median (it has no meaning for
    the exact MILP). epsilon is looped as the INNERMOST dimension so
    that, for a given (n, k, seed, instance), 'exact' is computed (or
    looked up from cache) ONCE and reused across every epsilon value --
    no extra Gurobi calls just from sweeping epsilon.
 
    Requires generate_instances_k_median.py (random_metric_costs,
    get_weights) to generate instances with the triangle inequality and
    the SAME OWA weight families as the assignment problem (k_fowama.py)
    -- imported lazily below to avoid a hard dependency for callers who
    only need the algorithms themselves.
    """
    from generate_instances_k_median import random_metric_costs, get_weights
 
    instance_end = nb_instances if instance_end is None else instance_end
    budget = TimeBudget(limit_seconds=time_limit, nb_instances=nb_instances)
 
    if output_path is None:
        output_path = f"results/kmedian_{w_type}.parquet"
 
    rows = []
    for n in n_values:
        cost_matrices = [random_metric_costs(n, seed=seed + i) for i in range(nb_instances)]
        for k in k_values:
            exact_cache_path = default_exact_cache_path(w_type, n, k, exact_cache_dir)
            for i in range(instance_start, instance_end):
                W = get_weights(w_type, n, i)
                cost_matrix = cost_matrices[i]
 
                skip_info = []
                for method in ["min_sum_k_median", "exact"]:
                    if budget.is_stopped(method, n, k):
                        skip_info.append(
                            f"{method} (cumul={budget.current_total(method, n, k):.0f}s "
                            f"> budget={budget.total_budget:.0f}s)"
                        )
                skip_str = f" [STOPPED: {', '.join(skip_info)}]" if skip_info else ""
 
                for eps in epsilon_values:
                    print(f"[running] n={n} k={k} epsilon={eps} instance={i} w_type={w_type}{skip_str}")
 
                    row = run_one(
                        cost_matrix, W, n, k, i, seed, budget, w_type,
                        methods=methods, exact_cache_path=exact_cache_path,
                        ls_epsilon=eps, ls_init=ls_init,
                    )
                    row["w_type"] = w_type
                    rows.append(row)
 
                    _upsert_parquet(row, output_path)
 
                    def fmt(v):
                        return "  --  " if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.4f}"
 
                    print(f"  -> min_sum_k_median: {fmt(row['min_sum_k_median_time'])} "
                          f"(OWA obj {fmt(row['min_sum_k_median_owa_obj'])})  "
                          f"exact: {fmt(row['exact_time'])} (OWA obj {fmt(row['exact_owa_obj'])})  "
                          f"ratio: {fmt(row['ratio_min_sum_k_median'])}")
 
    return pd.DataFrame(rows)
 

 
 
def print_comparison(df: pd.DataFrame):
    pd.set_option("display.width", 160)
 
    agg = df.groupby(["w_type", "n", "k", "epsilon"]).agg(
        instances=("instance", "count"),
        min_sum_k_median_time=("min_sum_k_median_time", "mean"),
        exact_time=("exact_time", "mean"),
        ratio_mean=("ratio_min_sum_k_median", "mean"),
        ratio_max=("ratio_min_sum_k_median", "max"),
        n_exact=("exact_owa_obj", lambda s: s.notna().sum()),
        n_ls=("min_sum_k_median_owa_obj", lambda s: s.notna().sum()),
    ).reset_index()
 
    for w_type in agg["w_type"].unique():
        sub = agg[agg["w_type"] == w_type].sort_values(["k", "n", "epsilon"])
        print(f"\n{'=' * 110}\n{w_type}\n{'=' * 110}")
        print(f"{'n':>5} {'k':>5} {'eps':>6} {'inst':>5} | {'n_ls':>5} {'n_exact':>7} | "
              f"{'t_ls':>9} {'t_exact':>9} | {'ratio':>10} {'max':>7}")
        for _, r in sub.iterrows():
            def fmt(v):
                return "  --  " if np.isnan(v) else f"{v:.4f}"
            print(f"{int(r.n):>5} {int(r.k):>5} {r.epsilon:>6.2f} {int(r.instances):>5} | "
                  f"{int(r.n_ls):>5} {int(r.n_exact):>7} | "
                  f"{fmt(r.min_sum_k_median_time):>9} {fmt(r.exact_time):>9} | "
                  f"{fmt(r.ratio_mean):>10} {fmt(r.ratio_max):>7}")
 
    print(f"\n{'=' * 110}\n{len(df)} runs total")
 
 

if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--k-values", type=int, nargs="+", required=True,
                         help="numbers of facilities to open (swept, like --n-values).")
    parser.add_argument("--w-type", default="s_gini_1.5_weights")
    parser.add_argument("--nb-instances", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=1800.0,
                         help="average time budget (s) per (method, n, k) -- default 1800s. "
                              "Shared across all epsilon values tested for a given (method, n, k).")
    parser.add_argument("--methods", default=None,
                         help="comma-separated subset of min_sum_k_median,exact. Default: both.")
    parser.add_argument("--instance-start", type=int, default=0)
    parser.add_argument("--instance-end", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--exact-cache-dir", default="results")
    parser.add_argument("--epsilon-values", type=float, nargs="+", default=[0.0],
                         help="min_sum_k_median's epsilon values to sweep (0 = naive, "
                              ">0 = time-guaranteed). Has no effect on 'exact'; 'exact' is "
                              "computed/cached once per (n, k, seed, instance) and reused "
                              "across every epsilon value.")
    parser.add_argument("--ls-init", default="greedy", choices=["greedy", "random"])
    args = parser.parse_args()
 
    output_path = args.output
    if output_path is None:
        if args.instance_start != 0 or args.instance_end is not None:
            end = args.instance_end if args.instance_end is not None else args.nb_instances
            output_path = f"results/kmedian_{args.w_type}_inst{args.instance_start}-{end}.parquet"
        else:
            output_path = f"results/kmedian_{args.w_type}.parquet"
 
    methods = set(args.methods.split(",")) if args.methods else None
 
    df = run_comparison(
        n_values=args.n_values,
        w_type=args.w_type,
        k_values=args.k_values,
        nb_instances=args.nb_instances,
        seed=args.seed,
        time_limit=args.time_limit,
        output_path=output_path,
        instance_start=args.instance_start,
        instance_end=args.instance_end,
        methods=methods,
        exact_cache_dir=args.exact_cache_dir,
        epsilon_values=args.epsilon_values,
        ls_init=args.ls_init,
    )
    print_comparison(df)
