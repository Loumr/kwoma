"""
merge_results_kmedian.py

Fusionne les fichiers parquet produits par les jobs paralleles de
kmowa.py (un fichier par (w_type, n, k, methods), cf.
submit_array_oar_kmedian_big17.sh) en un seul fichier de resultats.

Pourquoi pas un simple concat + drop_duplicates : un job 'exact' et un
job 'min_sum_k_median' sur le meme (n, k) ecrivent chacun des lignes
completes, ou les colonnes de l'AUTRE methode sont a NaN. Dedupliquer
les lignes ecraserait les resultats d'une methode avec les NaN de
l'autre. On reconstruit donc les deux blocs de colonnes separement :

  - heuristique (min_sum_k_median_*) : cle (w_type, n, k, epsilon, seed,
    instance) -- le resultat depend d'epsilon ;
  - exact (exact_*) : cle (w_type, n, k, seed, instance) -- epsilon n'a
    aucun sens pour la methode exacte et est ignore.

puis on rattache le resultat exact a chaque ligne heuristique (jointure
externe : une instance resolue par une seule des deux methodes est
conservee, avec des NaN pour l'autre), et on recalcule le ratio.

Temps exact : la version actuelle de kmowa.py stocke le temps du calcul
Gurobi dans le cache et le reecrit a chaque lecture du cache. Les
fichiers produits par les versions precedentes contiennent en revanche
exact_time = 0.0 pour les instances lues depuis le cache (une fois par
epsilon apres le premier, ou apres une resoumission). Pour chaque
instance on garde donc le temps le plus grand observe (le vrai calcul).
Si aucun vrai temps n'est disponible (seulement 0.0 ou NaN), le temps
est inconnu : il est mis a NaN et signale.

Usage :
    python3 merge_results_kmedian.py 'results/kmedian_*.parquet' \\
        --output kmedian_merged.parquet
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

HEUR_KEY = ["w_type", "n", "k", "epsilon", "seed", "instance"]
EXACT_KEY = ["w_type", "n", "k", "seed", "instance"]

HEUR_COLS = ["min_sum_k_median_time", "min_sum_k_median_owa_obj",
             "min_sum_k_median_nb_sols", "min_sum_k_median_S"]
EXACT_COLS = ["exact_time", "exact_owa_obj", "exact_S"]


def _read_all(paths):
    dfs = []
    # ordre chronologique : en cas de doublon, le fichier le plus recent gagne
    for order, p in enumerate(sorted(paths, key=os.path.getmtime)):
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"[ignore] impossible de lire {p} : {e}")
            continue
        missing = [c for c in EXACT_KEY + ["epsilon"] if c not in df.columns]
        if missing:
            print(f"[ignore] {p} : colonnes manquantes {missing}")
            continue
        for c in HEUR_COLS + EXACT_COLS:
            if c not in df.columns:
                df[c] = np.nan
        df["_order"] = order
        dfs.append(df)
    return dfs


def _heuristic_block(combined):
    """Une ligne par (w_type, n, k, epsilon, seed, instance) ou l'heuristique a tourne."""
    h = combined[combined["min_sum_k_median_time"].notna()
                 | combined["min_sum_k_median_owa_obj"].notna()]
    h = h[HEUR_KEY + HEUR_COLS + ["_order"]].copy()
    # priorite : resultat present, puis fichier le plus recent
    h["_has_obj"] = h["min_sum_k_median_owa_obj"].notna()
    h = (h.sort_values(["_has_obj", "_order"])
          .drop_duplicates(subset=HEUR_KEY, keep="last"))
    return h.drop(columns=["_has_obj", "_order"])


def _exact_block(combined):
    """Une ligne par (w_type, n, k, seed, instance) ou l'exact a tourne (epsilon ignore)."""
    e = combined[combined["exact_time"].notna() | combined["exact_owa_obj"].notna()]
    e = e[EXACT_KEY + EXACT_COLS + ["_order"]].copy()

    # temps reel = le plus grand observe pour l'instance (les autres valent 0.0, cache)
    real_time = e.groupby(EXACT_KEY, dropna=False)["exact_time"].max()

    # priorite : resultat present, puis fichier le plus recent
    e["_has_obj"] = e["exact_owa_obj"].notna()
    e = (e.sort_values(["_has_obj", "_order"])
          .drop_duplicates(subset=EXACT_KEY, keep="last")
          .drop(columns=["_has_obj", "_order", "exact_time"]))
    e = e.merge(real_time.rename("exact_time").reset_index(), on=EXACT_KEY, how="left")

    only_cache = ((e["exact_time"] == 0.0) | e["exact_time"].isna()) & e["exact_owa_obj"].notna()
    if only_cache.any():
        print(f"[attention] {only_cache.sum()} instance(s) exacte(s) lue(s) uniquement "
              f"depuis le cache : temps inconnu, exact_time mis a NaN.")
        e.loc[only_cache, "exact_time"] = np.nan
    return e


def merge(paths, output_path):
    dfs = _read_all(paths)
    if not dfs:
        print("Aucun fichier lisible trouve.")
        return None
    combined = pd.concat(dfs, ignore_index=True)

    heur = _heuristic_block(combined)
    exact = _exact_block(combined)

    merged = heur.merge(exact, on=EXACT_KEY, how="outer")

    ok = (merged["exact_owa_obj"].notna() & (merged["exact_owa_obj"] > 0)
          & merged["min_sum_k_median_owa_obj"].notna())
    merged["ratio_min_sum_k_median"] = np.where(
        ok, merged["min_sum_k_median_owa_obj"] / merged["exact_owa_obj"], np.nan)

    cols = HEUR_KEY + HEUR_COLS + EXACT_COLS + ["ratio_min_sum_k_median"]
    merged = (merged[cols]
              .sort_values(by=HEUR_KEY, na_position="first")
              .reset_index(drop=True))

    merged.to_parquet(output_path, index=False)
    print(f"{len(merged)} lignes ecrites dans {output_path}")
    _print_coverage(heur, exact)
    return merged


def _print_coverage(heur, exact):
    """Nombre d'instances resolues par methode, par (w_type, n, k)."""
    g = ["w_type", "n", "k"]
    nb_exact = (exact[exact["exact_owa_obj"].notna()]
                .groupby(g).size().rename("exact"))
    nb_heur = (heur[heur["min_sum_k_median_owa_obj"].notna()]
               .drop_duplicates(subset=EXACT_KEY)
               .groupby(g).size().rename("min_sum_k_median"))
    cov = pd.concat([nb_exact, nb_heur], axis=1).fillna(0).astype(int)
    print("\nInstances resolues par (w_type, n, k) :")
    print(cov.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="+",
                        help="chemins ou motifs glob des fichiers parquet a fusionner "
                             "(ex. 'results/kmedian_*.parquet')")
    parser.add_argument("--output", default="kmedian_merged.parquet")
    args = parser.parse_args()

    paths = []
    for pattern in args.patterns:
        matched = glob.glob(pattern)
        paths.extend(matched if matched else [pattern])
    # ne jamais relire le fichier de sortie lui-meme (s'il correspond au motif)
    out = os.path.abspath(args.output)
    paths = sorted({p for p in paths if os.path.abspath(p) != out})

    print(f"{len(paths)} fichier(s) trouve(s) :")
    for p in paths:
        print(" ", p)

    merge(paths, args.output)
