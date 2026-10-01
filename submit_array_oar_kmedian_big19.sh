#!/bin/bash
#OAR -l {host='big19'}/core=4,walltime=48:00:00
#OAR --notify mail:lou.moulin-roussel@lip6.fr
#OAR -O logs/job_%jobid%.out
#OAR -E logs/job_%jobid%.err

# ---------------------------------------------------------------------
# Job array OAR pour kmowa.py -- compare min_sum_k_median (heuristique
# OWA k-median par seuillage) et kmowa_solver (exact, MILP Gurobi) sur
# un ensemble d'instances metriques.
#
# UN JOB PAR (n, k, methods). epsilon_values et w_type sont FIXES
# ci-dessous (meme principe que W_TYPE dans submit_array_oar.sh / la
# version elicitation) : pour tester une autre famille de poids ou
# d'autres valeurs d'epsilon, editez ces deux variables et relancez une
# nouvelle soumission. n, k et methods, eux, viennent de
# params_kmedian.txt (un job par ligne).
#
# Format de params_kmedian.txt : "N K [METHODS]"
#   METHODS optionnel, defaut "min_sum_k_median,exact" (les deux dans le
#   meme job, comportement d'origine). Pour ne JAMAIS lancer exact et
#   min_sum_k_median en meme temps, mettez une valeur UNIQUE par ligne et
#   dupliquez la ligne (n, k) une fois par methode :
#     20 4 exact
#     20 4 min_sum_k_median
#     30 6 exact
#     30 6 min_sum_k_median
#   Chaque job ecrit dans un fichier dedie a sa combinaison (n, k,
#   methods) -- aucun conflit entre un job 'exact' et un job
#   'min_sum_k_median' portant sur le meme (n, k), meme s'ils tournent en
#   parallele. merge_results_kmedian.py recombine ensuite les colonnes de
#   chaque fichier (voir plus bas) : NE PAS fusionner "a la main" avec un
#   simple drop_duplicates, ca effacerait les colonnes d'une methode avec
#   les NaN de l'autre.
#
# PAS de '/nodes=1' dans la ligne #OAR ci-dessus : sur beaucoup de
# configs OAR, ce niveau reserve le noeud ENTIER de facon exclusive
# meme si seuls quelques coeurs sont demandes, empechant plusieurs
# sous-jobs de tourner en parallele sur le meme hote.
#
# Soumission (un sous-job par ligne de params_kmedian.txt) :
#   chmod +x submit_array_oar_kmedian_big19.sh
#   mkdir -p results logs
#   oarsub -S ./submit_array_oar_kmedian_big19.sh --array-param-file params_kmedian.txt
#
# Suivi :
#   oarstat -u $USER
#   cat logs/job_<jobid>.out
#   cat logs/job_<jobid>.err
#   oardel <jobid>
#
# Fusion une fois tous les sous-jobs termines :
#   python3 merge_results_kmedian.py 'results/kmedian_*.parquet' \
#       --output kmedian_merged.parquet
# ---------------------------------------------------------------------

N=$1
K=$2
METHODS=${3:-"min_sum_k_median,exact"}
W_TYPE="s_gini_1.5_weights"
EPSILON_VALUES="0.0 0.3 0.6"   # liste separee par des espaces, passee telle quelle a --epsilon-values

if [ -z "$N" ] || [ -z "$K" ]; then
    echo "Usage : $0 n k [methods]" >&2
    echo "  methods : min_sum_k_median | exact | min_sum_k_median,exact (defaut)" >&2
    exit 1
fi

# Valide chaque methode demandee (separees par des virgules) contre la
# liste connue -- un nom invalide ici planterait silencieusement bien
# plus tard, cote Python, apres activation du venv/Gurobi.
IFS=',' read -ra METHOD_LIST <<< "$METHODS"
for m in "${METHOD_LIST[@]}"; do
    case "$m" in
        min_sum_k_median|exact) ;;
        *) echo "Methode invalide : '$m' (min_sum_k_median | exact)" >&2; exit 1 ;;
    esac
done
METHODS_LABEL=$(echo "$METHODS" | tr ',' '_')

# Activate your Python virtual environment
source ~/venvs/gurobi-135/bin/activate

# Set gurobi variables (verifier que ce .lic correspond bien a l'hote
# cible ci-dessus dans la ligne #OAR -- big19)
export GUROBI_HOME=$HOME/gurobi/gurobi1300/linux64
export PATH=$GUROBI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
export GRB_LICENSE_FILE=$HOME/gurobi_licenses/big19.lic

mkdir -p results logs

# Fichier de sortie DEDIE par (n, k, w_type, methods) -- aucun conflit
# entre sous-jobs paralleles, y compris entre un job 'exact' et un job
# 'min_sum_k_median' portant sur le MEME (n, k) : ils n'ecrivent jamais
# dans le meme fichier, donc jamais l'un par-dessus l'autre. Les valeurs
# d'epsilon ci-dessus partagent le MEME resultat 'exact' pour une
# instance donnee (cache partage dans kmowa.solve_exact_cached), donc
# balayer epsilon ne multiplie pas les appels a Gurobi, meme si 'exact'
# tourne dans son propre job.
OUTPUT="../results/kmedian_${W_TYPE}_n${N}_k${K}_${METHODS_LABEL}.parquet"

echo "[submit] n=${N} k=${K} methods=${METHODS} w_type=${W_TYPE} epsilon=${EPSILON_VALUES} -> ${OUTPUT}"

python3 ~/projet_kmowa/kmowa/kmowa.py \
    --n-values $N \
    --k-values $K \
    --w-type $W_TYPE \
    --nb-instances 30 \
    --time-limit 1800 \
    --epsilon-values $EPSILON_VALUES \
    --methods "$METHODS" \
    --exact-cache-dir results \
    --output "$OUTPUT"
