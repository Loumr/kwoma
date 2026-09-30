import numpy as np
import gurobipy as gp
from scipy.optimize import linear_sum_assignment
from gurobipy import GRB
import numpy.typing as npt
import pandas as pd
import os
import time
import sys
import fcntl
import multiprocessing
import traceback
import random
import glob
import re
import lap as lp
from pathlib import Path
from openpyxl import Workbook
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


#####################################################################
################ HELPER FUNCTIONS FOR THE ALGORITHMS ################
#####################################################################


# ------------------------------------------------------------------
# Generation d'instances
# ------------------------------------------------------------------

def compute_obj_matrix(x_sol, cost_matrix, W):
    """
    Compute objective .
    """
    row_costs = (cost_matrix * x_sol).sum(axis=1)

    return np.dot(W, np.sort(row_costs)[::-1])




#####################################################################
###################### ALGORITHMS IMPLEMENTATION ####################
#####################################################################


def min_lorenz(cost_matrix, W):
    """
    Compute all assignments minimizing the min sum version of the problem
    with modified cost matrices and returns the best one 
    """

    costs = cost_matrix.copy()
    n = np.shape(costs)[0]
    _, assign = linear_sum_assignment(costs)

    f_assign = compute_obj_matrix(np.eye(n, dtype=int)[assign], costs, W)

    unique_solutions = set()
    unique_solutions.add(tuple(assign))

    for t in np.unique(costs)[1::]:
        # Compute the temporary cost matrix where costs smaller than t are replaced by t
        temp_costs_t = np.maximum(costs, t)

        # Compute assignment minimizing the sum of the costs per agent with costs temp_costs_t
        _, assign_t = linear_sum_assignment(temp_costs_t)

        f_assign_t = compute_obj_matrix(np.eye(n, dtype=int)[assign_t], costs, W)

        # Keep unique cost vectors
        unique_solutions.add(tuple(assign_t))
    
        if f_assign_t < f_assign:
            assign = assign_t
            f_assign = f_assign_t

    return assign, len(unique_solutions)






def kmowa_solver(cost_matrix, k, W):
    "MILP formulation of the OWA k-median problem with Chassein and Goerigk (2015) formulation"
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
    m.setParam("NumericFocus", 3)

 
    # Add variables
    alpha = m.addMVar(shape=n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="alpha")
    beta = m.addMVar(shape=n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="beta")
    x = m.addMVar(shape=(n, n), lb=0.0, vtype=GRB.BINARY, name="x")
    y = m.addMVar(shape=n, lb=0.0, vtype=GRB.BINARY, name="y")
    z = m.addMVar(n, lb=0.0, name="z")
    
    # Add constraints
    # Constraints
    for j in range(n):
        m.addConstr(z[j] == gp.quicksum(costs[i, j] * x[i, j] for i in range(n)), name=f"linearization_z_{j}")

    for i in range(n):
        for j in range(n):
            m.addConstr(alpha[i] + beta[j] - W[j] * z[i] >= 0)

    for j in range(n):
        m.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1, name=f"cover_{j}")

    for i in range(n):
        for j in range(n):
            m.addConstr(x[i, j] <= y[i], name=f"open_{i}_{j}")

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



# ---------------------------------------------------------------------
# Local Search Algorithm for k-median
# Version efficace (threshold) : swap simple (p=1), ratio 5
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
 
    for _ in range(k): # Choose each facility one by one
        best_f, best_cost = None, np.inf
        for f in remaining:
            new_min = np.minimum(current_min, dist_matrix[:, f])
            c = new_min.sum()
            if c < best_cost:
                best_cost, best_f = c, f
        chosen.append(best_f)
        remaining.discard(best_f)
        current_min = np.minimum(current_min, dist_matrix[:, best_f])
 
    return np.array(chosen)
 

 
def local_search_single_swap(
    dist_matrix: np.ndarray,
    k: int,
    epsilon: float = 0.0,
    init: str = "greedy",
    seed: int = 0,
    max_iters: int | None = None,
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
        Indices of open facilities (sike k).
    cost : float
        Total cost of returned solution.
    n_iters : int
        Number of iterations.
    """
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
    threshold = (1.0 - epsilon / Q) if epsilon > 0 else None
 
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
 
        if epsilon > 0 and best_cost > threshold * cost_S:
            break  # insufficient amelioration, stopping to respect time guarantee
        
        # Update variables after the swap
        pos, f_new = best_swap
        s_old = int(S[pos])
        S[pos] = f_new
        S_set.discard(s_old)
        S_set.add(f_new)
 
        cost_S = best_cost
 
    return sorted(S.tolist()), cost_S, it



#####################################################################
######################### TESTING AND EXPORT ########################
#####################################################################


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def wrapper(q, func, args):
    try:
        res = func(*args)
        q.put(("OK", res))
    except Exception:
        q.put(("ERR", traceback.format_exc()))




def run_with_timeout(func, args=(), timeout=10):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=wrapper, args=(q, func, args))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return "TIMEOUT"

    if not q.empty():
        status, result = q.get()
        if status == "ERR":
            print(result)
            raise RuntimeError("Error in child process")
        return result

    return None




def _solution_to_vector(x: np.ndarray) -> np.ndarray:
    """
    Convert an assignment solution to a 1D vector of size n where
    tab[i] is the index of the object assigned to agent i.
    Accepts either a 1D vector (returned as-is) or an n x n
    permutation matrix (argmax along rows).
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x.astype(int)
    if x.ndim == 2:
        return np.argmax(x, axis=1).astype(int)
    raise ValueError(f"Unexpected solution dimension: {x.ndim}")




def _is_missing(value) -> bool:
    """
    Scalar-and-array-aware missing value check.
    pd.isna() raises on numpy arrays, so we handle them separately.
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
def _upsert_parquet(row: dict, parquet_path: str, key_cols=None) -> None:
    """
    Append a single result row to the method-specific parquet file.
    If a row with the same key (key_cols, default KEY_COLS = n, seed, instance)
    already exists, it is replaced by the new one (keep last).
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
# Cache "exact" results (Gurobi)
# -------------------------------------------------------------------

EXACT_CACHE_KEY_COLS = ["w_type", "n", "seed", "instance"]


def default_exact_cache_path(w_type, n, cache_dir="results"):
    """
    Chemin par defaut du cache exact, UN FICHIER PAR (w_type, n) --
    aligne sur le decoupage des jobs OAR (un job par n dans
    submit_array_oar.sh), pour qu'un job donne ne lise/n'ecrive jamais
    dans le meme fichier qu'un job traitant un autre n. Le seul risque
    de concurrence restant est un job compare_min_lorenz.py et un job
    elicitation.py tournant EN MEME TEMPS sur le MEME n : dans ce cas
    (comme pour _upsert_parquet), une course est possible mais reste
    benigne -- au pire les deux recalculent la meme instance une fois
    chacun (temps perdu), jamais de corruption de fichier, car
    drop_duplicates(keep="last") retient juste l'une des deux lignes
    identiques."""
    return os.path.join(cache_dir, f"exact_cache_{w_type}_n{n}.parquet")


def _exact_cache_lookup(w_type, n, seed, instance, cache_path):
    """Retourne {"exact obj value": ..., "x": ...} si un resultat exact
    existe deja dans le cache pour cette cle, sinon None."""
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
        & (df["seed"] == seed)
        & (df["instance"] == instance)
    )
    hit = df.loc[mask]
    if hit.empty:
        return None
    last = hit.iloc[-1]
    return {"exact obj value": last["exact_obj"], "x": last["exact_sol"]}


def _exact_cache_store(w_type, n, seed, instance, obj_value, sol_vector, cache_path):
    row = {
        "w_type": w_type,
        "n": n,
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
            # Cache corrompu -- typiquement un job precedent tue par le
            # walltime OAR en pleine ecriture, AVANT l'ecriture atomique
            # ci-dessous. On ne plante pas : on met le fichier illisible
            # de cote (pour investigation eventuelle) et on repart d'un
            # cache ne contenant que la nouvelle ligne. Les anciennes
            # entrees de ce fichier corrompu sont perdues, mais elles
            # seront simplement recalculees au prochain passage puisque
            # _exact_cache_lookup ne les trouvera plus.
            backup_path = f"{cache_path}.corrupted.{int(time.time())}"
            print(f"[WARNING] cache exact illisible ({e}) -- "
                  f"sauvegarde dans {backup_path} et reinitialisation.")
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

    # Ecriture ATOMIQUE : on ecrit dans un fichier temporaire puis on le
    # renomme par-dessus cache_path (os.replace est atomique sur un
    # meme systeme de fichiers POSIX). Si CE processus est tue pendant
    # le to_parquet, seul le fichier .tmp est incomplet -- cache_path
    # lui-meme reste intact (l'ancienne version valide, ou absent si
    # c'est la toute premiere ecriture). Ca elimine a la source le
    # risque de cache corrompu par une coupure en cours d'ecriture.
    tmp_path = f"{cache_path}.tmp{os.getpid()}"
    df_combined.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, cache_path)


def has_cached_exact(w_type, n, seed, instance, cache_path):
    """True si un resultat exact est deja present dans le cache pour
    cette cle (w_type, n, seed, instance) -- permet de decider de
    reutiliser le cache meme quand le budget de temps de la methode
    'exact' est deja depasse (la reutilisation ne coute rien)."""
    return _exact_cache_lookup(w_type, n, seed, instance, cache_path) is not None


def _exact_cache_lock_path(cache_path):
    return cache_path + ".lock"


def solve_exact_cached(cost_matrix, W, w_type, n, k, seed, instance, cache_path):
    """
    Retourne (obj_value, sol_vector, elapsed, from_cache).

    Si le resultat exact pour (w_type, n, seed, instance) est deja dans
    le cache, il est reutilise tel quel (elapsed=0.0, from_cache=True) --
    Gurobi n'est PAS relance. Sinon, affectation_1_1_solver est appele
    normalement, le resultat est stocke dans le cache puis renvoye.

    En cas d'echec du solveur (res is None, ex. limite de temps/memoire
    Gurobi atteinte), rien n'est mis en cache et (None, None, elapsed,
    False) est renvoye.

    VERROUILLAGE : tout le bloc "verifier le cache -> calculer si absent
    -> stocker" est protege par un verrou de fichier EXCLUSIF
    (fcntl.flock, bloquant) sur cache_path + '.lock'. Sans ca, deux
    processus qui demandent l'exact pour la MEME cle EN MEME TEMPS (ex.
    deux valeurs de compute_X_method lancees en parallele avec le meme
    w_type et le meme n) peuvent tous les deux trouver le cache vide et
    lancer Gurobi chacun de leur cote -- pas de corruption de fichier
    (drop_duplicates(keep='last') absorbe ca), mais le calcul est
    effectivement fait deux fois, ce qui est exactement ce qu'on veut
    eviter. Avec le verrou, le second processus attend que le premier
    ait fini d'ecrire dans le cache, puis relit son resultat au lieu de
    relancer le MILP.

    ATTENTION : fcntl.flock suppose un verrouillage de fichiers
    fonctionnel sur le systeme de fichiers partage entre les noeuds du
    cluster (NFS avec lockd actif, ou systeme de fichiers local). Si vos
    jobs OAR tournent sur des noeuds differents avec un NFS ou le
    verrouillage est desactive/peu fiable, le verrou peut ne pas
    empecher la course inter-noeuds (il reste efficace intra-noeud, et
    dans le cas usuel ou un seul noeud heberge tous les sous-jobs d'un
    meme n via 'core=4', cf. submit_array_oar.sh, ca couvre le cas
    decrit ici).
    """
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    lock_path = _exact_cache_lock_path(cache_path)

    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)  # bloquant : attend si un autre processus tient le verrou
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
            _exact_cache_store(w_type, n, seed, instance, obj_value, sol_vector, cache_path)
            return obj_value, sol_vector, elapsed, False
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)




    # -------------------------------------------------------------------
# Main test function
# -------------------------------------------------------------------


def test_approx(
    n: int,
    k: int,
    nb_instances: int,
    method: str,
    save: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Run nb_instances assignment tests on random n x n cost matrices
    and save results to a method-specific parquet file after each instance.

    :param n: number of agents / objects
    :param nb_instances: number of random instances to test
    :param method: "exact"  -> computes exact solution only.
                   "lorenz" -> computes approx solution (kfw.approx_from_lorenz)
                               as well as the exact solution needed to compute
                               the ratio at export time.
    :param save: if True, upsert each instance result into the parquet file
                 right after it is computed
    :param seed: random seed for reproducibility
    :return: DataFrame with one row per instance
    """
    print(f"Weight type: {WEIGHTS_TYPE}; method: {method}; n: {n}")
    if method not in ("lorenz", "exact", "approx_precomputed", "approx_weights"):
        raise ValueError(f"method must be 'lorenz' or 'exact', got: {method}")

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)

    parquet_path = f"{RESULTS_DIR}/results_{method}_n{n}.parquet"

    # Generate random instances with a reproducible seed
    rng = np.random.default_rng(seed)

    # Generate weights W based on the selected WEIGHTS_TYPE
    if WEIGHTS_TYPE == "linear":
        W = np.arange(1, n + 1)[::-1] / (n * (n + 1))

    if WEIGHTS_TYPE.startswith("s_gini"): 
        delta = float(WEIGHTS_TYPE.split("_")[2])
        W = s_gini_weights(n, delta)

    if WEIGHTS_TYPE == "sqrt":
        W = np.ones(n)
        W[0] = np.sqrt(n)


    rows = []
    for i in range(nb_instances):
        print("Instance:", i)

        cost_matrix = rng.integers(1, 10**4, (n, n), dtype=int)

        if method == "exact":
            start = time.time()
            res = kmowa_solver(cost_matrix, k, W)
            elapsed = time.time() - start

            row = {
                "n": n,
                "instance": i,
                "seed": seed,
                "exact sol": _solution_to_vector(res["x"]),
                "exact obj value": res["obj value"],
                "exact sol time": elapsed,
            }

      
        elif method == "approx_precomputed":
            start = time.time()
            pre_sol, nb_sol = min_lorenz(cost_matrix, W)
            sol = np.eye(n, dtype=int)[pre_sol]
            obj_value = compute_obj_matrix(sol, cost_matrix, W)
            res_approx_precomputed = {"x": sol, "obj value": obj_value}
            elapsed = time.time() - start

            row = {
                "n": n,
                "instance": i,
                "seed": seed,
                "approx pre sol": _solution_to_vector(res_approx_precomputed["x"]),
                "approx pre obj value": res_approx_precomputed["obj value"],
                "approx pre time": elapsed,
                "approx pre nb sols": nb_sol,
            }

        rows.append(row)

        if save:
            _upsert_parquet(row, parquet_path)
            print(f"[{method}] n={n}, instance {i+1}/{nb_instances} saved.")

    return pd.DataFrame(rows)

