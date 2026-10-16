"""Writes one line "N K EPS" per (n, k, epsilon) triplet (Unix line endings).
    python3 make_params_ls_vs_gurobi.py > params_ls_vs_gurobi.txt
"""
N_VALUES = [100, 150, 200, 250, 300]
K_VALUES = [10, 15, 20, 25]
#EPSILONS = ["0", "0.01", "0.1"]
EPSILONS = ["0"]
# ordered by (n, k) so that the jobs sharing an exact cache start together
for n in N_VALUES:
    for k in K_VALUES:
        if k < n:
            for eps in EPSILONS:
                print(f"{n} {k} {eps} exact")

