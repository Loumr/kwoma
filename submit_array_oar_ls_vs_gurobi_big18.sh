#!/bin/bash
#OAR -l {host='big18'}/core=4,walltime=48:00:00
#OAR --notify mail:lou.moulin-roussel@lip6.fr
#OAR -O logs/job_%jobid%.out
#OAR -E logs/job_%jobid%.err

# ---------------------------------------------------------------------
# Job array OAR: local_search_single_swap vs Gurobi (min-sum k-median).
# ONE sub-job per (n, k, epsilon) triplet = one line of the params file:
#     N K EPS [METHODS]
#     50 5 0
#     50 5 0.1
#     50 5 0.01
# METHODS (optional): local_search | exact | local_search,exact (default).
#
# Jobs on the same (n, k) share the exact cache: each instance is solved
# by Gurobi ONCE (file lock); the other jobs wait for it and read the
# result. The exact budget counts the ORIGINAL Gurobi times, so every job
# of a given (n, k) skips the same instances.
#
# Budget: each method stops once its cumulative time exceeds
# TIME_LIMIT x NB_INSTANCES (local_search per (n, k, eps), exact per (n, k)).
# Worst case per job ~ 2 x TIME_LIMIT x NB_INSTANCES + EXACT_TIME_LIMIT
# (= 30.5 h with the values below) -> keep walltime above that.
#
#   python3 make_params_ls_vs_gurobi.py > params_ls_vs_gurobi.txt   (or write it by hand)
#   chmod +x submit_array_oar_ls_vs_gurobi_big17.sh
#   mkdir -p results logs
#   oarsub -S ./submit_array_oar_ls_vs_gurobi_big17.sh --array-param-file params_ls_vs_gurobi.txt
#
# Once every sub-job is done:
#   python3 ~/projet_kmowa/kmowa/compare_local_search_gurobi_kmedian.py \
#       --report 'results/ls_vs_gurobi_minsum_*.parquet' --output ls_vs_gurobi_merged.parquet
# ---------------------------------------------------------------------

set -euo pipefail

# strip \r (params file edited under Windows)
N=${1:-};   N=${N//$'\r'/}
K=${2:-};   K=${K//$'\r'/}
EPS=${3:-}; EPS=${EPS//$'\r'/}
METHODS=${4:-"local_search,exact"}; METHODS=${METHODS//$'\r'/}

LS_INIT="greedy"
TIME_LIMIT=1800          # average budget per instance -> total = TIME_LIMIT x NB_INSTANCES per method
EXACT_TIME_LIMIT=1800    # Gurobi TimeLimit of ONE instance
NB_INSTANCES=30
RESULTS_DIR="results"

if [ -z "$N" ] || [ -z "$K" ] || [ -z "$EPS" ]; then
    echo "Usage: $0 n k eps [methods]" >&2; exit 1
fi
IFS=',' read -ra METHOD_LIST <<< "$METHODS"
for m in "${METHOD_LIST[@]}"; do
    case "$m" in
        local_search|exact) ;;
        *) echo "Invalid method: '$m' (local_search | exact)" >&2; exit 1 ;;
    esac
done
METHODS_LABEL=$(echo "$METHODS" | tr ',' '_')

source ~/venvs/gurobi-135/bin/activate
export GUROBI_HOME=$HOME/gurobi/gurobi1300/linux64
export PATH=$GUROBI_HOME/bin:${PATH:-}
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:${LD_LIBRARY_PATH:-}
export GRB_LICENSE_FILE=$HOME/gurobi_licenses/big18.lic
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p "$RESULTS_DIR" logs
# one file per (n, k, eps, init, methods): parallel jobs never write the same file
OUTPUT="${RESULTS_DIR}/ls_vs_gurobi_minsum_n${N}_k${K}_eps${EPS}_${LS_INIT}_${METHODS_LABEL}.parquet"
echo "[submit] n=${N} k=${K} eps=${EPS} methods=${METHODS} -> ${OUTPUT}"

python3 ~/projet_kmowa/kmowa/compare_local_search_gurobi_kmedian.py \
    --n-values "$N" \
    --k-values "$K" \
    --epsilons "$EPS" \
    --nb-instances "$NB_INSTANCES" \
    --ls-init "$LS_INIT" \
    --methods "$METHODS" \
    --time-limit "$TIME_LIMIT" \
    --exact-time-limit "$EXACT_TIME_LIMIT" \
    --threads 4 \
    --cache-dir "$RESULTS_DIR" \
    --output "$OUTPUT"
