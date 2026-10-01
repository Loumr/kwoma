import numpy as np



# ----------------------------------------------------------------------
# Couts
# ----------------------------------------------------------------------

def random_metric_costs(n, low=1, high=10000, seed=0):
    """Generates a random cost matrix and make it metric by applying Floyd-Warshall
    (closure by shortest path): the result is exactly the shortest path distance in the
    complete graph with random weights, which always satisfies the triangle inequality."""
    rng = np.random.default_rng(seed)
    C = rng.integers(low, high, size=(n, n)).astype(float)
    iu = np.triu_indices(n, k=1)
    C[(iu[1], iu[0])] = C[iu]  # makes the cost matrix symmetric (distance from a to b is equal to distance from b to a)
    np.fill_diagonal(C, 0)
    for k in range(n):
        C = np.minimum(C, C[:, [k]] + C[[k], :])
    return C.astype(int)


def generate_cost_matrices(n, nb_instances, seed=0):
    cost_matrices = []
    for i in range(nb_instances):
        inst = random_metric_costs(n, seed=seed+i)
        cost_matrices.append(inst)
    return cost_matrices


def check_triangle_inequality(C, tol=1e-9):
    """Check if a cost matrix satisfies the triangle inequality."""
    n = C.shape[0]
    for k in range(n):
        if np.any(C > C[:, [k]] + C[[k], :] + tol):
            return False
    return True


# ----------------------------------------------------------------------
# OWA weights
# ----------------------------------------------------------------------
def s_gini_weights(n, delta):
    "Returns the S-Gini weights of size n with parameter delta."
    k = np.arange(n, -1, -1)
    f = (k / n) ** delta
    return f[:-1] - f[1:]


def get_weights(w_type, n, instance):
    """w_type must be in one of the following forms: 
    'linear_weights', 's_gini_1.5_weights', 'random_weights'
    n(int): size of the instance (number of possible facilities/clients)
    instance(int): index of the instance
    """
    base = w_type.replace("_weights", "")

    if base == "linear":
        return np.arange(1, n + 1)[::-1] / (n * (n + 1))

    if base.startswith("s_gini"):
        delta = float(base.split("_")[2])
        return s_gini_weights(n, delta)

    if base == "random":
        BASE_SEED = 12345
        rng_i = np.random.default_rng(BASE_SEED + instance)
        rand = rng_i.uniform(0, 1, n)
        return np.sort(rand)[::-1]

    raise ValueError(f"w_type inconnu : {w_type}")


# ----------------------------------------------------------------------------------------------------------------
# Complete instance: costs satisfying the triangle inequality + OWA weights + k (number of facilities to be open)
# ----------------------------------------------------------------------------------------------------------------
def generate_instances(n, nb_instances, w_type, k, cost_seed=0):
    """
    Generates nb_instances instances for (n, k, w_type)
    Returns a dict of lists:
        {"instance": i, "cost_matrix": ..., "W": ..., "k": ...}
    """
    cost_matrices = generate_cost_matrices(n, nb_instances, seed=cost_seed)
    instances = []
    for i in range(nb_instances):
        W = get_weights(w_type, n, i)
        instances.append({
            "instance": i,
            "cost_matrix": cost_matrices[i],
            "W": W,
            "k": k,
        })
    return instances
