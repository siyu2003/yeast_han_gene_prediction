#!/usr/bin/env python3
"""
V7 ─ Pathway-aware validation for yeast metabolite → alopecia candidate ranking
================================================================================

Purpose
-------
This script does NOT create another complicated final ranking.
Instead, it validates whether the existing GNN/evidence-calibrated metabolite
ranking contains pathway-level biological signal.

The key correction from V6 is conceptual:
  - Pathway-overlap clustering is treated only as a visualization / redundancy
    reduction step.
  - The primary validation is pathway enrichment of top-ranked metabolites
    compared against random and degree-matched null models.
  - A secondary, less-circular validation clusters metabolites in ranking-feature
    space first, and only then asks which pathways are enriched in each cluster.

Inputs
------
1) pathway_file: BioCyc-like TSV with columns:
   Pathways, Compounds of pathway, Reactions of pathway, SMILES
2) ranking_file: CSV generated from the GNN candidate ranking. Expected columns
   include metabolite_name, final_rank/final_score, metabolite_gnn_score,
   path scores, max_tanimoto, target_relevance_score. Drug score columns are
   optional and are used only for overlay/annotation by default.

Outputs
-------
- pathway_enrichment_actual.csv
- pathway_enrichment_with_null_empirical_p.csv
- null_model_metrics.csv
- enriched_pathway_network_nodes.csv / edges.csv
- metabolite_feature_clusters.csv
- metabolite_cluster_pathway_enrichment.csv
- multiple PNG figures
- pathway_validation_report_v7.html

Recommended command
-------------------
python inductive_yeast_han_rwpe_v7.py \
  --pathway_file All-pathways-of-S-cerevisiae-S288c_cleaned_compounds.txt \
  --ranking_file metabolite_alopecia_final_ranking_evidence_calibrated_with_50drug_score.csv \
  --top_n 100 \
  --n_perm 1000 \
  --out_dir v7_pathway_validation_results
"""

from __future__ import annotations

import argparse
import html
import math
import os
import random
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import hypergeom
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    import networkx as nx
except Exception as exc:  # pragma: no cover
    raise ImportError("networkx is required for pathway network visualization") from exc


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="V7 pathway enrichment + null-model validation for GNN metabolite ranking"
    )
    p.add_argument("--pathway_file", default="All-pathways-of-S-cerevisiae-S288c_cleaned_compounds.txt")
    p.add_argument("--ranking_file", default="metabolite_alopecia_final_ranking_evidence_calibrated_with_50drug_score.csv")
    p.add_argument("--out_dir", default="v7_pathway_validation_results")

    # Foreground selection.
    p.add_argument("--top_n", type=int, default=100,
                   help="Number of high-confidence metabolites used as foreground")
    p.add_argument("--foreground_mode", choices=["final_rank", "final_score", "validation_score"],
                   default="final_rank",
                   help="How to choose foreground metabolites. final_rank is most faithful to the original ranking.")

    # Cleaning.
    p.add_argument("--pathway_hub_quantile", type=float, default=0.95,
                   help="Remove metabolites above this pathway-membership quantile as hub/currency-like nodes")
    p.add_argument("--min_hub_pathway_count", type=int, default=8,
                   help="Minimum pathway count before a metabolite can be considered a hub")
    p.add_argument("--keep_drug_evidence_candidates", action="store_true",
                   help="Do not remove a candidate with drug_ranking_score>0 even if it is a pathway hub. Default is stricter.")

    # Enrichment.
    p.add_argument("--min_pathway_size", type=int, default=3)
    p.add_argument("--max_pathway_size", type=int, default=80,
                   help="Exclude very broad pathways from primary enrichment table")
    p.add_argument("--min_hits", type=int, default=2)
    p.add_argument("--fdr_alpha", type=float, default=0.10)
    p.add_argument("--p_alpha", type=float, default=0.05)

    # Null model.
    p.add_argument("--n_perm", type=int, default=1000)
    p.add_argument("--degree_bins", type=int, default=8,
                   help="Number of bins for pathway-degree-matched random sampling")
    p.add_argument("--seed", type=int, default=42)

    # Network visualization.
    p.add_argument("--network_top_k", type=int, default=35,
                   help="Use the top K enriched pathways for redundancy-reduction network")
    p.add_argument("--min_overlap", type=float, default=0.12)
    p.add_argument("--min_shared_hits", type=int, default=1)

    # Metabolite feature-space clustering.
    p.add_argument("--metabolite_cluster_k", type=int, default=7)
    p.add_argument("--feature_cluster_top_n", type=int, default=180,
                   help="Top candidates used for metabolite feature-space clustering")

    # Optional drug evidence weighting. Off by default to avoid forcing expected drug candidates.
    p.add_argument("--include_drug_score_in_validation", action="store_true",
                   help="Include drug_ranking_score_0_14 with a small weight in validation_candidate_score")

    return p.parse_args()


# =============================================================================
# Basic utilities
# =============================================================================

COMPOUND_SYNONYMS = {
    "3-hydroxy-3-methylglutaryl-coa": "(s)-3-hydroxy-3-methylglutaryl-coa",
    "mevalonic acid": "(r)-mevalonate",
    "mevalonate 5-phosphate": "(r)-5-phosphomevalonate",
    "mevalonate 5-diphosphate": "(r)-mevalonate diphosphate",
    "isopentenyl pyrophosphate": "isopentenyl diphosphate",
    "dimethylallyl pyrophosphate": "dimethylallyl diphosphate",
    "geranyl pyrophosphate": "geranyl diphosphate",
    "farnesyl pyrophosphate": "(2e,6e)-farnesyl diphosphate",
    "geranylgeranyl pyrophosphate": "geranylgeranyl diphosphate",
    "2,3-oxidosqualene": "(3s)-2,3-epoxy-2,3-dihydrosqualene",
    "farnesol": "(2e,6e)-farnesol",
}

EXPLICIT_CURRENCY = {
    "h+", "h2o", "water", "co2", "carbon dioxide", "o2", "dioxygen",
    "phosphate", "diphosphate", "pyrophosphate", "orthophosphate",
    "atp", "adp", "amp", "gtp", "gdp", "gmp", "ctp", "cdp", "cmp", "utp", "udp", "ump",
    "datp", "dadp", "dgtp", "dgdp", "dctp", "dcdp", "dttp", "dtdp",
    "nad+", "nadh", "nadp+", "nadph", "fad", "fadh2", "fmn",
    "coenzyme a", "coa", "acetyl-coa", "an acyl-coa",
    "l-glutamate", "l-glutamine", "glycine", "l-aspartate", "pyruvate", "2-oxoglutarate",
    "ammonia", "formate", "a reduced thioredoxin", "an oxidized thioredoxin",
    "a reduced glutaredoxin", "a [glutaredoxin]-glutathione disulfide",
    "s-adenosyl-l-methionine", "s-adenosyl-l-homocysteine",
    "a tetrahydrofolate", "an n10-formyltetrahydrofolate", "a 5-methyltetrahydrofolate",
}

ARTIFACT_PATTERNS = [
    r"\b(2-methylbutanol|3-methylbutanol|methionol|isobutanol|ethanol|methanol|propanol|butanol)\b",
    r"\b(acetaldehyde|phenylacetaldehyde|formaldehyde)\b",
    r"\b(nitric oxide|hydrogen peroxide|superoxide)\b",
    r"\b(an alcohol|a phosphate monoester|a carboxylate|a proteinogenic amino acid|an l-amino acid)\b",
    r"\[r\]|\[r1\]|\[r2\]|\[a |carrier protein|a protein|an enzyme|phosphoglucomutase",
    r"glucopyranose|galactopyranose|mannopyranose|fructofuranose|trehalose|sucrose",
]

BROAD_PATHWAY_PATTERNS = [
    r"superpathway of purine", r"purine nucleotides", r"pyrimidine", r"nucleotide",
    r"glycolysis", r"gluconeogenesis", r"tca cycle", r"tricarboxylic",
    r"amino acid biosynthesis", r"charging", r"trna", r"central carbon",
]

MODULE_KEYWORDS = [
    ("Sterol / isoprenoid", ["ergosterol", "sterol", "zymosterol", "lanosterol", "squalene", "mevalonate", "farnes", "geranyl", "isoprenoid"]),
    ("Sphingolipid / ceramide", ["sphingo", "ceramide", "phytosphingosine", "sphinganine"]),
    ("Phospholipid / fatty acid", ["phosphatid", "choline", "glycerolipid", "oleate", "palmit", "fatty acid", "monoacylglycerol", "triacylglycerol", "phospholipid"]),
    ("Polyamine", ["spermidine", "spermine", "putrescine", "methylthioadenosine", "polyamine"]),
    ("Vitamin / cofactor", ["biotin", "riboflavin", "pyridox", "thiamin", "folate", "lipoate", "nad", "pantothen", "cofactor"]),
    ("Inositol phosphate", ["inositol", "phosphoinosit", "phytate"]),
    ("Aromatic amino acid / chorismate", ["chorismate", "shikimate", "phenylalanine", "tyrosine", "tryptophan", "aromatic"]),
    ("Nucleotide / salvage", ["adenosine", "adenine", "guan", "purine", "pyrimidine", "nucleotide", "nucleoside", "salvage"]),
    ("Amino acid / sulfur", ["cysteine", "methionine", "sulfur", "sulfate", "glutathione"]),
    ("Carbohydrate / glycosylation", ["glucose", "glucosamine", "udp", "chitin", "glycosyl", "mannose", "trehalose"]),
]


def norm_name(x) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return COMPOUND_SYNONYMS.get(s, s)


def safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(np.full(len(df), default), index=df.index, dtype=float)


def robust_scale(s: pd.Series, qlo: float = 0.05, qhi: float = 0.95) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(x) == 0:
        return x
    lo, hi = x.quantile(qlo), x.quantile(qhi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = x.min(), x.max()
    if hi <= lo:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return ((x.clip(lo, hi) - lo) / (hi - lo)).clip(0.0, 1.0)


def soft_tanimoto(x: pd.Series, midpoint: float = 0.35, temperature: float = 0.08) -> pd.Series:
    z = pd.to_numeric(x, errors="coerce").fillna(0.0)
    return 1.0 / (1.0 + np.exp(-(z - midpoint) / max(temperature, 1e-6)))


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[order[i]] = prev
    return np.clip(q, 0.0, 1.0)


def is_name_artifact(name: str) -> bool:
    n = norm_name(name)
    return any(re.search(p, n) for p in ARTIFACT_PATTERNS)


def is_broad_pathway(pathway: str) -> bool:
    p = str(pathway).lower()
    return any(re.search(pattern, p) for pattern in BROAD_PATHWAY_PATTERNS)


def module_label(pathway: str) -> str:
    p = str(pathway).lower()
    for label, keys in MODULE_KEYWORDS:
        if any(k in p for k in keys):
            return label
    return "Other metabolism"


def metabolite_module_label(name: str, met_to_paths: Dict[str, Set[str]], enriched_pathways: Set[str] | None = None) -> str:
    key = norm_name(name)
    paths = met_to_paths.get(key, set())
    if enriched_pathways:
        paths = paths & enriched_pathways or paths
    labels = [module_label(p) for p in paths]
    if not labels:
        return "No pathway label"
    return Counter(labels).most_common(1)[0][0]


# =============================================================================
# Loading pathway and ranking data
# =============================================================================

def load_pathway_membership(pathway_file: str) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], pd.DataFrame]:
    df = pd.read_csv(pathway_file, sep="\t", dtype=str).fillna("")
    if "Pathways" not in df.columns or "Compounds of pathway" not in df.columns:
        raise ValueError("Pathway file must have columns 'Pathways' and 'Compounds of pathway'")

    path_to_mets: Dict[str, Set[str]] = defaultdict(set)
    met_to_paths: Dict[str, Set[str]] = defaultdict(set)

    rows = []
    for _, row in df.iterrows():
        path = str(row["Pathways"]).strip()
        if not path:
            continue
        compounds = [c.strip() for c in str(row["Compounds of pathway"]).split(" // ") if c.strip()]
        for c in compounds:
            cn = norm_name(c)
            if not cn:
                continue
            path_to_mets[path].add(cn)
            met_to_paths[cn].add(path)
        rows.append({"pathway": path, "n_compounds_raw": len(set(map(norm_name, compounds)))})

    path_stats = pd.DataFrame(rows).drop_duplicates("pathway")
    return dict(path_to_mets), dict(met_to_paths), path_stats


def load_and_clean_ranking(
    ranking_file: str,
    met_to_paths: Dict[str, Set[str]],
    pathway_hub_quantile: float,
    min_hub_pathway_count: int,
    keep_drug_evidence_candidates: bool,
    include_drug_score_in_validation: bool,
) -> Tuple[pd.DataFrame, Set[str]]:
    df = pd.read_csv(ranking_file)
    if "metabolite_name" not in df.columns:
        raise ValueError("ranking_file must contain metabolite_name column")
    df = df.copy()
    df["metabolite_norm"] = df["metabolite_name"].map(norm_name)

    # One row per metabolite. Keep best final_rank or highest final_score.
    if "final_rank" in df.columns:
        df = df.sort_values("final_rank", na_position="last").drop_duplicates("metabolite_norm", keep="first")
    else:
        df = df.sort_values("final_score", ascending=False).drop_duplicates("metabolite_norm", keep="first")

    df["pathway_degree"] = df["metabolite_norm"].map(lambda m: len(met_to_paths.get(m, set()))).fillna(0).astype(int)
    degrees = df.loc[df["pathway_degree"] > 0, "pathway_degree"]
    if len(degrees):
        hub_threshold = max(min_hub_pathway_count, int(math.ceil(degrees.quantile(pathway_hub_quantile))))
    else:
        hub_threshold = min_hub_pathway_count
    hub_set = set(df.loc[df["pathway_degree"] >= hub_threshold, "metabolite_norm"])

    df["is_explicit_currency"] = df["metabolite_norm"].isin(EXPLICIT_CURRENCY)
    df["is_pathway_hub"] = df["metabolite_norm"].isin(hub_set)
    df["is_name_artifact"] = df["metabolite_name"].map(is_name_artifact)
    df["has_pathway"] = df["metabolite_norm"].map(lambda m: m in met_to_paths)

    if keep_drug_evidence_candidates and "drug_ranking_score_0_14" in df.columns:
        drug_supported = safe_col(df, "drug_ranking_score_0_14") > 0
    else:
        drug_supported = pd.Series(False, index=df.index)

    df["remove_reason"] = ""
    df.loc[df["is_explicit_currency"], "remove_reason"] = "explicit_currency"
    df.loc[df["is_pathway_hub"] & ~drug_supported, "remove_reason"] = "pathway_hub"
    df.loc[df["is_name_artifact"], "remove_reason"] = "name_artifact"
    df.loc[~df["has_pathway"], "remove_reason"] = "no_pathway_membership"
    df["is_removed_from_validation"] = df["remove_reason"] != ""

    # Transparent validation score used only for foreground selection / plots.
    final_score_scaled = robust_scale(safe_col(df, "final_score"))
    gnn_scaled = robust_scale(safe_col(df, "metabolite_gnn_score"))
    weighted_path_scaled = robust_scale(np.log1p(safe_col(df, "sum_top3_weighted_path_score")))
    target_rel = safe_col(df, "target_relevance_score").clip(0, 1)
    tani_soft = soft_tanimoto(safe_col(df, "max_tanimoto"))

    if "final_rank" in df.columns:
        rank = safe_col(df, "final_rank", default=np.nan)
        max_rank = np.nanmax(rank) if np.isfinite(rank).any() else len(df)
        rank_position = (1.0 - (rank.fillna(max_rank) - 1) / max(max_rank - 1, 1)).clip(0, 1)
    elif "rank" in df.columns:
        rank = safe_col(df, "rank", default=np.nan)
        max_rank = np.nanmax(rank) if np.isfinite(rank).any() else len(df)
        rank_position = (1.0 - (rank.fillna(max_rank) - 1) / max(max_rank - 1, 1)).clip(0, 1)
    else:
        rank_position = final_score_scaled

    score = (
        0.30 * final_score_scaled
        + 0.25 * rank_position
        + 0.20 * gnn_scaled
        + 0.15 * weighted_path_scaled
        + 0.07 * target_rel
        + 0.03 * tani_soft
    )
    if include_drug_score_in_validation and "drug_ranking_score_0_14" in df.columns:
        drug_scaled = (safe_col(df, "drug_ranking_score_0_14") / 14.0).clip(0, 1)
        score = 0.92 * score + 0.08 * drug_scaled

    penalty = pd.Series(0.0, index=df.index)
    if "artifact_penalty" in df.columns:
        penalty += 0.30 * safe_col(df, "artifact_penalty").clip(0, 1)
    penalty += 0.70 * df["is_pathway_hub"].astype(float)
    penalty += 0.85 * df["is_explicit_currency"].astype(float)
    penalty += 0.90 * df["is_name_artifact"].astype(float)
    penalty += 0.30 * (~df["has_pathway"]).astype(float)
    df["validation_candidate_score_raw"] = score.clip(0, 1)
    df["validation_penalty"] = penalty.clip(0, 1)
    df["validation_candidate_score"] = (score * (1 - penalty.clip(0, 1))).clip(0, 1)
    return df.reset_index(drop=True), hub_set


def select_foreground(clean_df: pd.DataFrame, top_n: int, mode: str) -> pd.DataFrame:
    usable = clean_df[~clean_df["is_removed_from_validation"]].copy()
    if mode == "final_rank" and "final_rank" in usable.columns:
        usable = usable.sort_values(["final_rank", "validation_candidate_score"], ascending=[True, False], na_position="last")
    elif mode == "final_score" and "final_score" in usable.columns:
        usable = usable.sort_values(["final_score", "validation_candidate_score"], ascending=[False, False])
    else:
        usable = usable.sort_values("validation_candidate_score", ascending=False)
    return usable.head(min(top_n, len(usable))).copy().reset_index(drop=True)


# =============================================================================
# Enrichment and null models
# =============================================================================

def prepare_pathway_sets(
    path_to_mets: Dict[str, Set[str]],
    background: Set[str],
    min_pathway_size: int,
    max_pathway_size: int,
) -> Dict[str, Set[str]]:
    prepared = {}
    for p, mets in path_to_mets.items():
        s = set(mets) & background
        if min_pathway_size <= len(s) <= max_pathway_size:
            prepared[p] = s
    return prepared


def compute_pathway_enrichment(
    pathway_sets: Dict[str, Set[str]],
    foreground: Set[str],
    background: Set[str],
    selected_scores: Dict[str, float] | None = None,
) -> pd.DataFrame:
    N = len(background)
    n = len(foreground)
    rows = []
    for p, mets in pathway_sets.items():
        K = len(mets)
        hits = sorted(mets & foreground)
        k = len(hits)
        expected = n * K / max(N, 1)
        pval = float(hypergeom.sf(k - 1, N, K, n)) if k > 0 else 1.0
        oe = k / expected if expected > 0 else 0.0
        if selected_scores:
            mean_score = float(np.mean([selected_scores.get(m, 0.0) for m in hits])) if hits else 0.0
            max_score = float(np.max([selected_scores.get(m, 0.0) for m in hits])) if hits else 0.0
        else:
            mean_score = 0.0
            max_score = 0.0
        rows.append({
            "pathway": p,
            "module_label": module_label(p),
            "pathway_size_in_background": K,
            "selected_hits": k,
            "expected_hits": expected,
            "observed_expected_ratio": oe,
            "hypergeom_p": pval,
            "neglog10_p": -math.log10(max(pval, 1e-300)),
            "hit_fraction_in_pathway": k / K if K else 0.0,
            "mean_selected_candidate_score": mean_score,
            "max_selected_candidate_score": max_score,
            "broad_pathway_flag": is_broad_pathway(p),
            "hit_metabolites": " | ".join(hits),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["fdr_bh"] = bh_fdr(out["hypergeom_p"].to_numpy())
        # A transparent ranking score for pathway table only.
        out["pathway_validation_score"] = (
            0.40 * robust_scale(out["neglog10_p"])
            + 0.25 * robust_scale(np.log1p(out["observed_expected_ratio"]))
            + 0.20 * robust_scale(out["mean_selected_candidate_score"])
            + 0.15 * robust_scale(out["hit_fraction_in_pathway"])
        )
        out.loc[out["broad_pathway_flag"], "pathway_validation_score"] *= 0.75
        out = out.sort_values(["pathway_validation_score", "neglog10_p", "observed_expected_ratio"], ascending=False)
    return out.reset_index(drop=True)


def null_metrics_from_enrichment(enrich: pd.DataFrame, min_hits: int, p_alpha: float, fdr_alpha: float | None = None) -> dict:
    if len(enrich) == 0:
        return {
            "n_pathways_p_lt_alpha": 0,
            "n_pathways_fdr_lt_alpha": 0,
            "max_neglog10_p": 0.0,
            "sum_top10_neglog10_p": 0.0,
            "mean_top10_neglog10_p": 0.0,
        }
    valid = enrich[enrich["selected_hits"] >= min_hits].copy()
    n_sig_p = int(((valid["hypergeom_p"] <= p_alpha)).sum())
    n_sig_fdr = int(((valid["fdr_bh"] <= fdr_alpha)).sum()) if fdr_alpha is not None and "fdr_bh" in valid else 0
    top = valid["neglog10_p"].sort_values(ascending=False).head(10)
    return {
        "n_pathways_p_lt_alpha": n_sig_p,
        "n_pathways_fdr_lt_alpha": n_sig_fdr,
        "max_neglog10_p": float(top.max()) if len(top) else 0.0,
        "sum_top10_neglog10_p": float(top.sum()) if len(top) else 0.0,
        "mean_top10_neglog10_p": float(top.mean()) if len(top) else 0.0,
    }


def assign_degree_bins(background: List[str], met_to_paths: Dict[str, Set[str]], n_bins: int) -> Dict[str, int]:
    deg = pd.Series({m: len(met_to_paths.get(m, set())) for m in background})
    # qcut can fail with many ties; rank first to make stable bins.
    ranks = deg.rank(method="first")
    try:
        bins = pd.qcut(ranks, q=min(n_bins, len(ranks)), labels=False, duplicates="drop")
    except Exception:
        bins = pd.Series(np.zeros(len(deg), dtype=int), index=deg.index)
    return bins.astype(int).to_dict()


def sample_degree_matched(
    background: List[str],
    foreground: Set[str],
    degree_bins: Dict[str, int],
    rng: np.random.Generator,
) -> Set[str]:
    bg_by_bin = defaultdict(list)
    fg_by_bin = Counter()
    for m in background:
        bg_by_bin[degree_bins.get(m, 0)].append(m)
    for m in foreground:
        fg_by_bin[degree_bins.get(m, 0)] += 1

    sampled = []
    for b, count in fg_by_bin.items():
        pool = bg_by_bin[b]
        if len(pool) >= count:
            sampled.extend(rng.choice(pool, size=count, replace=False).tolist())
        else:
            sampled.extend(rng.choice(pool, size=count, replace=True).tolist())
    # De-duplicate and refill if needed.
    sampled_set = set(sampled)
    if len(sampled_set) < len(foreground):
        remaining = list(set(background) - sampled_set)
        need = len(foreground) - len(sampled_set)
        if remaining:
            sampled_set.update(rng.choice(remaining, size=min(need, len(remaining)), replace=False).tolist())
    return set(list(sampled_set)[:len(foreground)])


def run_null_models(
    pathway_sets: Dict[str, Set[str]],
    foreground: Set[str],
    background: Set[str],
    met_to_paths: Dict[str, Set[str]],
    n_perm: int,
    degree_bins_n: int,
    min_hits: int,
    p_alpha: float,
    fdr_alpha: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fast random and degree-matched null models.

    Earlier versions rebuilt a full enrichment DataFrame in every permutation.
    This vectorized version precomputes a pathway x metabolite membership matrix
    and evaluates all pathway p-values by array operations.
    """
    rng = np.random.default_rng(seed)
    bg_list = sorted(background)
    fg_size = len(foreground)
    met_index = {m: i for i, m in enumerate(bg_list)}
    path_names = list(pathway_sets.keys())
    P, N = len(path_names), len(bg_list)

    # Dense matrix is fine here (hundreds of metabolites/pathways).
    M = np.zeros((P, N), dtype=np.int8)
    K = np.zeros(P, dtype=int)
    for pi, p in enumerate(path_names):
        idxs = [met_index[m] for m in pathway_sets[p] if m in met_index]
        if idxs:
            M[pi, idxs] = 1
            K[pi] = len(idxs)

    degree_bins = assign_degree_bins(bg_list, met_to_paths, degree_bins_n)
    bg_by_bin_idx = defaultdict(list)
    fg_by_bin = Counter()
    for m, idx in met_index.items():
        bg_by_bin_idx[degree_bins.get(m, 0)].append(idx)
    for m in foreground:
        fg_by_bin[degree_bins.get(m, 0)] += 1

    def sample_degree_matched_idx() -> np.ndarray:
        sampled = []
        for b, count in fg_by_bin.items():
            pool = np.array(bg_by_bin_idx[b], dtype=int)
            if len(pool) >= count:
                sampled.extend(rng.choice(pool, size=count, replace=False).tolist())
            else:
                sampled.extend(rng.choice(pool, size=count, replace=True).tolist())
        sampled = list(dict.fromkeys(sampled))
        if len(sampled) < fg_size:
            remaining = np.array(list(set(range(N)) - set(sampled)), dtype=int)
            need = fg_size - len(sampled)
            if len(remaining):
                sampled.extend(rng.choice(remaining, size=min(need, len(remaining)), replace=False).tolist())
        return np.array(sampled[:fg_size], dtype=int)

    def evaluate_indices(indices: np.ndarray):
        k = M[:, indices].sum(axis=1).astype(int)
        pvals = np.ones(P, dtype=float)
        mask = k > 0
        pvals[mask] = hypergeom.sf(k[mask] - 1, N, K[mask], fg_size)
        qvals = bh_fdr(pvals)
        neglog = -np.log10(np.maximum(pvals, 1e-300))
        valid = k >= min_hits
        top = np.sort(neglog[valid])[::-1][:10]
        metrics = {
            "n_pathways_p_lt_alpha": int(np.sum(valid & (pvals <= p_alpha))),
            "n_pathways_fdr_lt_alpha": int(np.sum(valid & (qvals <= fdr_alpha))),
            "max_neglog10_p": float(top.max()) if len(top) else 0.0,
            "sum_top10_neglog10_p": float(top.sum()) if len(top) else 0.0,
            "mean_top10_neglog10_p": float(top.mean()) if len(top) else 0.0,
        }
        return metrics, neglog

    metric_rows = []
    random_mat = np.zeros((n_perm, P), dtype=np.float32)
    degree_mat = np.zeros((n_perm, P), dtype=np.float32)

    for i in range(n_perm):
        rand_idx = rng.choice(N, size=fg_size, replace=False)
        deg_idx = sample_degree_matched_idx()

        rm, rneg = evaluate_indices(rand_idx)
        dm, dneg = evaluate_indices(deg_idx)
        rm.update({"perm": i, "null_model": "random"})
        dm.update({"perm": i, "null_model": "degree_matched"})
        metric_rows.extend([rm, dm])
        random_mat[i, :] = rneg
        degree_mat[i, :] = dneg

    null_metrics = pd.DataFrame(metric_rows)
    random_dist = pd.DataFrame(random_mat, columns=path_names)
    degree_dist = pd.DataFrame(degree_mat, columns=path_names)
    return null_metrics, random_dist, degree_dist


def add_empirical_p(actual: pd.DataFrame, random_dist: pd.DataFrame, degree_dist: pd.DataFrame) -> pd.DataFrame:
    out = actual.copy()
    rand_emp = []
    deg_emp = []
    for _, row in out.iterrows():
        p = row["pathway"]
        val = float(row["neglog10_p"])
        if p in random_dist:
            arr = random_dist[p].dropna().to_numpy()
            rand_emp.append(float((np.sum(arr >= val) + 1) / (len(arr) + 1)))
        else:
            rand_emp.append(np.nan)
        if p in degree_dist:
            arr = degree_dist[p].dropna().to_numpy()
            deg_emp.append(float((np.sum(arr >= val) + 1) / (len(arr) + 1)))
        else:
            deg_emp.append(np.nan)
    out["empirical_p_random_neglogp"] = rand_emp
    out["empirical_p_degree_matched_neglogp"] = deg_emp
    return out


# =============================================================================
# Redundancy reduction / pathway network visualization
# =============================================================================

def build_enriched_pathway_network(enriched: pd.DataFrame, foreground: Set[str], top_k: int, min_overlap: float, min_shared_hits: int):
    if len(enriched) == 0:
        return nx.Graph()
    use = enriched.head(top_k).copy()
    G = nx.Graph()
    hit_sets = {}
    for _, row in use.iterrows():
        hits = set(str(row["hit_metabolites"]).split(" | ")) if row.get("hit_metabolites") else set()
        hits = {h for h in hits if h and h != "nan"}
        hit_sets[row["pathway"]] = hits
        G.add_node(
            row["pathway"],
            score=float(row["pathway_validation_score"]),
            module=row["module_label"],
            hits=int(row["selected_hits"]),
            fdr=float(row["fdr_bh"]),
            oe=float(row["observed_expected_ratio"]),
            neglogp=float(row["neglog10_p"]),
        )

    paths = list(hit_sets)
    for i, a in enumerate(paths):
        for b in paths[i + 1:]:
            A, B = hit_sets[a], hit_sets[b]
            shared = A & B
            if not shared:
                continue
            union = A | B
            jacc = len(shared) / len(union) if union else 0.0
            overlap_coef = len(shared) / max(1, min(len(A), len(B)))
            weight = 0.65 * jacc + 0.35 * overlap_coef
            if weight >= min_overlap and len(shared) >= min_shared_hits:
                G.add_edge(a, b, weight=weight, shared_hits=len(shared), shared_metabolites=" | ".join(sorted(shared)))
    if G.number_of_nodes() and G.number_of_edges():
        communities = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
        for cid, comm in enumerate(communities):
            for node in comm:
                G.nodes[node]["cluster_id"] = cid
    else:
        for idx, node in enumerate(G.nodes):
            G.nodes[node]["cluster_id"] = idx
    return G


def graph_to_tables(G: nx.Graph) -> Tuple[pd.DataFrame, pd.DataFrame]:
    node_rows = []
    for n, attrs in G.nodes(data=True):
        node_rows.append({"pathway": n, **attrs})
    edge_rows = []
    for a, b, attrs in G.edges(data=True):
        edge_rows.append({"source": a, "target": b, **attrs})
    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


# =============================================================================
# Metabolite feature-space clustering
# =============================================================================

def metabolite_feature_clustering(
    clean_df: pd.DataFrame,
    foreground_n: int,
    k: int,
    met_to_paths: Dict[str, Set[str]],
    background: Set[str],
    pathway_sets: Dict[str, Set[str]],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    usable = clean_df[~clean_df["is_removed_from_validation"]].copy()
    usable = usable.sort_values("validation_candidate_score", ascending=False).head(min(foreground_n, len(usable))).copy()

    feature_cols = [
        "final_score", "metabolite_gnn_score", "best_path_score", "best_weighted_path_score",
        "sum_top3_weighted_path_score", "max_tanimoto", "target_relevance_score",
        "validation_candidate_score",
    ]
    Xcols = []
    for col in feature_cols:
        if col in usable.columns:
            usable[col] = safe_col(usable, col)
            Xcols.append(col)
    if not Xcols:
        raise ValueError("No numeric ranking features found for metabolite clustering")

    X = usable[Xcols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(Xs)

    n_clusters = max(2, min(k, len(usable) // 8 if len(usable) >= 16 else 2))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    labels = km.fit_predict(Xs)
    sil = float(silhouette_score(Xs, labels)) if len(set(labels)) > 1 and len(usable) > len(set(labels)) else np.nan

    usable["feature_cluster_id"] = labels
    usable["pca1"] = coords[:, 0]
    usable["pca2"] = coords[:, 1]
    usable["dominant_pathway_module"] = usable["metabolite_norm"].map(lambda m: metabolite_module_label(m, met_to_paths))

    # Enrichment of pathways within each metabolite feature cluster.
    rows = []
    for cid in sorted(set(labels)):
        fg = set(usable.loc[usable["feature_cluster_id"] == cid, "metabolite_norm"])
        if len(fg) < 3:
            continue
        enr = compute_pathway_enrichment(pathway_sets, fg, background)
        enr = enr[enr["selected_hits"] >= 2].head(10).copy()
        enr.insert(0, "feature_cluster_id", cid)
        rows.append(enr)
    cluster_enrich = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return usable, cluster_enrich, sil


# =============================================================================
# Plotting
# =============================================================================

def save_barplot_enrichment(enriched: pd.DataFrame, out_path: Path, top_k: int = 20):
    top = enriched.head(top_k).copy()
    if len(top) == 0:
        return
    labels = [textwrap.shorten(p, width=55, placeholder="…") for p in top["pathway"]]
    y = np.arange(len(top))[::-1]
    fig, ax = plt.subplots(figsize=(11, max(5, len(top) * 0.35)))
    ax.barh(y, top["neglog10_p"].values[::-1])
    ax.set_yticks(y)
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel("-log10(hypergeometric p)")
    ax.set_title("Actual top-ranked metabolites: pathway enrichment")
    for i, (_, row) in enumerate(top.iloc[::-1].iterrows()):
        ax.text(row["neglog10_p"] + 0.03, i, f"hits={int(row['selected_hits'])}, OE={row['observed_expected_ratio']:.2f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_null_comparison(null_metrics: pd.DataFrame, actual_metrics: dict, out_path: Path):
    if len(null_metrics) == 0:
        return
    metric = "sum_top10_neglog10_p"
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in ["random", "degree_matched"]:
        vals = null_metrics.loc[null_metrics["null_model"] == model, metric].astype(float).values
        if len(vals):
            ax.hist(vals, bins=30, alpha=0.45, label=model)
    ax.axvline(actual_metrics.get(metric, 0.0), linestyle="--", linewidth=2, label="actual")
    ax.set_xlabel("Sum of top 10 pathway -log10(p)")
    ax.set_ylabel("Permutation count")
    ax.set_title("Null-model comparison: actual enrichment vs random baselines")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_pathway_network_plot(G: nx.Graph, out_path: Path):
    if G.number_of_nodes() == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 9))
    seed = 42
    pos = nx.spring_layout(G, seed=seed, weight="weight", k=0.9)
    clusters = [G.nodes[n].get("cluster_id", 0) for n in G.nodes]
    scores = np.array([G.nodes[n].get("score", 0.1) for n in G.nodes], dtype=float)
    sizes = 250 + 1800 * (scores / max(scores.max(), 1e-9))
    widths = [0.5 + 3.0 * G.edges[e].get("weight", 0.1) for e in G.edges]
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.35, ax=ax)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=clusters, cmap="tab20", alpha=0.9, ax=ax)
    # Label only high-score or high-hit nodes.
    label_nodes = {}
    for n in G.nodes:
        if G.nodes[n].get("score", 0) >= np.quantile(scores, 0.65) or G.nodes[n].get("hits", 0) >= 5:
            label_nodes[n] = textwrap.shorten(n, width=28, placeholder="…")
    nx.draw_networkx_labels(G, pos, labels=label_nodes, font_size=8, ax=ax)
    ax.set_title("Visualization only: enriched pathway redundancy network")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metabolite_feature_scatter(met_clusters: pd.DataFrame, out_path: Path):
    if len(met_clusters) == 0:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    labels = met_clusters["feature_cluster_id"].values
    sizes = 30 + 220 * robust_scale(met_clusters["validation_candidate_score"]).values
    sc = ax.scatter(met_clusters["pca1"], met_clusters["pca2"], c=labels, s=sizes, cmap="tab20", alpha=0.82, edgecolors="k", linewidths=0.25)
    # Label top few by validation/drug score.
    label_df = met_clusters.sort_values("validation_candidate_score", ascending=False).head(15)
    if "drug_ranking_score_0_14" in met_clusters.columns:
        label_df = pd.concat([label_df, met_clusters.sort_values("drug_ranking_score_0_14", ascending=False).head(8)]).drop_duplicates("metabolite_norm")
    for _, row in label_df.iterrows():
        ax.text(row["pca1"], row["pca2"], str(row["metabolite_name"])[:24], fontsize=7)
    ax.set_xlabel("PCA1 of ranking feature space")
    ax.set_ylabel("PCA2 of ranking feature space")
    ax.set_title("Less-circular validation: metabolite clustering first, pathway labels later")
    fig.colorbar(sc, ax=ax, label="feature-space cluster")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_drug_overlay(clean_df: pd.DataFrame, out_path: Path):
    if "drug_ranking_score_0_14" not in clean_df.columns:
        return
    df = clean_df[~clean_df["is_removed_from_validation"]].copy()
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    x = safe_col(df, "validation_candidate_score")
    y = safe_col(df, "drug_ranking_score_0_14")
    ax.scatter(x, y, alpha=0.65, s=45)
    top = df.sort_values("drug_ranking_score_0_14", ascending=False).head(12)
    for _, row in top.iterrows():
        ax.text(row["validation_candidate_score"], row["drug_ranking_score_0_14"], str(row["metabolite_name"])[:24], fontsize=7)
    ax.set_xlabel("GNN-derived validation candidate score")
    ax.set_ylabel("External drug-evidence score (annotation only)")
    ax.set_title("Drug-evidence overlay: not used by default for enrichment")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Reporting
# =============================================================================

def make_cluster_summary(G: nx.Graph) -> pd.DataFrame:
    if G.number_of_nodes() == 0:
        return pd.DataFrame()
    rows = []
    clusters = defaultdict(list)
    for n, attrs in G.nodes(data=True):
        clusters[attrs.get("cluster_id", -1)].append((n, attrs))
    for cid, items in sorted(clusters.items()):
        modules = Counter(attrs.get("module", "Other") for _, attrs in items)
        top_module, top_count = modules.most_common(1)[0]
        top_paths = sorted(items, key=lambda x: x[1].get("score", 0), reverse=True)[:5]
        rows.append({
            "cluster_id": cid,
            "n_pathways": len(items),
            "dominant_module": top_module,
            "module_purity": top_count / len(items) if items else 0.0,
            "mean_pathway_score": float(np.mean([attrs.get("score", 0) for _, attrs in items])),
            "top_pathways": " | ".join([p for p, _ in top_paths]),
        })
    return pd.DataFrame(rows).sort_values(["mean_pathway_score", "n_pathways"], ascending=False)


def write_html_report(
    out_dir: Path,
    args: argparse.Namespace,
    actual_enrich: pd.DataFrame,
    actual_metrics: dict,
    null_metrics: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    met_cluster_silhouette: float,
):
    def table_html(df: pd.DataFrame, n=15):
        if df is None or len(df) == 0:
            return "<p>No rows.</p>"
        return df.head(n).to_html(index=False, escape=True, classes="table")

    # Empirical p for global metric.
    global_rows = []
    for metric in ["n_pathways_p_lt_alpha", "max_neglog10_p", "sum_top10_neglog10_p"]:
        row = {"metric": metric, "actual": actual_metrics.get(metric, np.nan)}
        for model in ["random", "degree_matched"]:
            vals = null_metrics.loc[null_metrics["null_model"] == model, metric].astype(float).values
            if len(vals):
                row[f"{model}_mean"] = float(np.mean(vals))
                row[f"{model}_95pct"] = float(np.quantile(vals, 0.95))
                row[f"empirical_p_vs_{model}"] = float((np.sum(vals >= row["actual"]) + 1) / (len(vals) + 1))
        global_rows.append(row)
    global_df = pd.DataFrame(global_rows)
    global_df.to_csv(out_dir / "global_null_comparison_summary.csv", index=False)

    interp = f"""
    <h2>Interpretation guide</h2>
    <p><b>Primary validation:</b> whether top-ranked metabolites are enriched in specific pathways more than expected by chance.</p>
    <p><b>Null models:</b> random top-N and pathway-degree-matched random top-N. Degree matching controls for metabolites that naturally appear in many pathways.</p>
    <p><b>Pathway network:</b> visualization only. Pathways are connected after enrichment if they share high-scoring metabolites. It summarizes redundancy; it is not the main proof.</p>
    <p><b>Metabolite feature clustering:</b> a less-circular validation. Metabolites are clustered by ranking features first; pathway enrichment is added afterward as annotation.</p>
    """

    html_text = f"""
    <!DOCTYPE html>
    <html><head><meta charset="utf-8"><title>V7 Pathway Validation Report</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 28px; line-height: 1.45; }}
      .table {{ border-collapse: collapse; font-size: 12px; }}
      .table th, .table td {{ border: 1px solid #ccc; padding: 4px 6px; }}
      img {{ max-width: 100%; border: 1px solid #ddd; margin: 12px 0; }}
      code {{ background: #f4f4f4; padding: 2px 4px; }}
    </style></head><body>
    <h1>V7 Pathway-aware Validation</h1>
    <p>This report validates the existing GNN/evidence-calibrated metabolite ranking at the pathway level.</p>
    <ul>
      <li>top_n: {args.top_n}</li>
      <li>foreground_mode: {html.escape(args.foreground_mode)}</li>
      <li>n_perm: {args.n_perm}</li>
      <li>drug score included in validation: {args.include_drug_score_in_validation}</li>
      <li>metabolite feature-space silhouette: {met_cluster_silhouette:.3f}</li>
    </ul>
    {interp}
    <h2>Global null comparison</h2>
    {table_html(global_df, 10)}
    <img src="null_model_comparison.png">
    <h2>Top actual pathway enrichments</h2>
    {table_html(actual_enrich[["pathway", "module_label", "selected_hits", "expected_hits", "observed_expected_ratio", "hypergeom_p", "fdr_bh", "empirical_p_degree_matched_neglogp", "hit_metabolites"]], 20)}
    <img src="pathway_enrichment_barplot.png">
    <h2>Enriched pathway redundancy network</h2>
    <p>This is a summary visualization of enriched pathways, not the primary validation.</p>
    {table_html(cluster_summary, 20)}
    <img src="enriched_pathway_overlap_network.png">
    <h2>Metabolite feature-space clustering</h2>
    <p>Metabolites are clustered using ranking/GNN/path-score features first; pathway labels are added later.</p>
    <img src="metabolite_feature_space_clusters.png">
    <h2>Drug evidence overlay</h2>
    <p>Drug evidence is annotation by default, not a driver of enrichment.</p>
    <img src="drug_evidence_overlay.png">
    </body></html>
    """
    (out_dir / "pathway_validation_report_v7.html").write_text(html_text, encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/8] Loading pathway membership ...")
    path_to_mets, met_to_paths, path_stats = load_pathway_membership(args.pathway_file)
    path_stats.to_csv(out_dir / "pathway_raw_stats.csv", index=False)

    print("[2/8] Loading and cleaning ranking ...")
    clean_df, hub_set = load_and_clean_ranking(
        args.ranking_file,
        met_to_paths,
        args.pathway_hub_quantile,
        args.min_hub_pathway_count,
        args.keep_drug_evidence_candidates,
        args.include_drug_score_in_validation,
    )
    clean_df.to_csv(out_dir / "candidate_scores_and_filters_v7.csv", index=False)
    pd.DataFrame({"hub_metabolite": sorted(hub_set)}).to_csv(out_dir / "removed_pathway_hub_metabolites_v7.csv", index=False)

    foreground_df = select_foreground(clean_df, args.top_n, args.foreground_mode)
    foreground_df.to_csv(out_dir / "selected_foreground_metabolites_v7.csv", index=False)

    background = set(clean_df.loc[~clean_df["is_removed_from_validation"], "metabolite_norm"])
    foreground = set(foreground_df["metabolite_norm"])
    selected_scores = dict(zip(clean_df["metabolite_norm"], clean_df["validation_candidate_score"]))

    print(f"  background metabolites: {len(background)}")
    print(f"  foreground metabolites: {len(foreground)}")

    print("[3/8] Preparing pathway sets ...")
    pathway_sets = prepare_pathway_sets(path_to_mets, background, args.min_pathway_size, args.max_pathway_size)
    print(f"  pathways tested: {len(pathway_sets)}")

    print("[4/8] Computing actual pathway enrichment ...")
    actual = compute_pathway_enrichment(pathway_sets, foreground, background, selected_scores=selected_scores)
    actual_metrics = null_metrics_from_enrichment(actual, args.min_hits, args.p_alpha, args.fdr_alpha)

    print("[5/8] Running null models ...")
    null_metrics, random_dist, degree_dist = run_null_models(
        pathway_sets, foreground, background, met_to_paths,
        args.n_perm, args.degree_bins, args.min_hits, args.p_alpha, args.fdr_alpha, args.seed,
    )
    null_metrics.to_csv(out_dir / "null_model_metrics.csv", index=False)
    # Wide null distributions can be large; save compressed-ish csv for reproducibility.
    random_dist.to_csv(out_dir / "null_random_neglogp_by_pathway.csv", index=False)
    degree_dist.to_csv(out_dir / "null_degree_matched_neglogp_by_pathway.csv", index=False)

    actual = add_empirical_p(actual, random_dist, degree_dist)
    actual.to_csv(out_dir / "pathway_enrichment_actual_with_null_v7.csv", index=False)

    # Significant/enriched set for downstream visualization.
    enriched = actual[(actual["selected_hits"] >= args.min_hits) & ((actual["fdr_bh"] <= args.fdr_alpha) | (actual["hypergeom_p"] <= args.p_alpha))].copy()
    if len(enriched) == 0:
        enriched = actual[actual["selected_hits"] >= args.min_hits].head(args.network_top_k).copy()
    enriched.to_csv(out_dir / "enriched_pathways_for_visualization_v7.csv", index=False)

    print("[6/8] Building enriched pathway overlap network for visualization ...")
    G = build_enriched_pathway_network(enriched, foreground, args.network_top_k, args.min_overlap, args.min_shared_hits)
    nodes, edges = graph_to_tables(G)
    nodes.to_csv(out_dir / "enriched_pathway_network_nodes_v7.csv", index=False)
    edges.to_csv(out_dir / "enriched_pathway_network_edges_v7.csv", index=False)
    cluster_summary = make_cluster_summary(G)
    cluster_summary.to_csv(out_dir / "enriched_pathway_cluster_summary_v7.csv", index=False)

    print("[7/8] Running metabolite feature-space clustering ...")
    met_clusters, met_cluster_enrich, met_cluster_sil = metabolite_feature_clustering(
        clean_df, args.feature_cluster_top_n, args.metabolite_cluster_k, met_to_paths,
        background, pathway_sets, args.seed,
    )
    met_clusters.to_csv(out_dir / "metabolite_feature_clusters_v7.csv", index=False)
    met_cluster_enrich.to_csv(out_dir / "metabolite_cluster_pathway_enrichment_v7.csv", index=False)

    print("[8/8] Saving plots and HTML report ...")
    save_barplot_enrichment(actual[(actual["selected_hits"] >= args.min_hits)].copy(), out_dir / "pathway_enrichment_barplot.png")
    save_null_comparison(null_metrics, actual_metrics, out_dir / "null_model_comparison.png")
    save_pathway_network_plot(G, out_dir / "enriched_pathway_overlap_network.png")
    save_metabolite_feature_scatter(met_clusters, out_dir / "metabolite_feature_space_clusters.png")
    save_drug_overlay(clean_df, out_dir / "drug_evidence_overlay.png")

    write_html_report(out_dir, args, actual, actual_metrics, null_metrics, cluster_summary, met_cluster_sil)

    # Console summary.
    global_summary = pd.read_csv(out_dir / "global_null_comparison_summary.csv")
    print("\n=== V7 validation summary ===")
    print(f"Output directory: {out_dir}")
    print("Top enriched pathways:")
    cols = ["pathway", "module_label", "selected_hits", "expected_hits", "observed_expected_ratio", "hypergeom_p", "fdr_bh", "empirical_p_degree_matched_neglogp"]
    print(actual[cols].head(10).to_string(index=False))
    print("\nGlobal null comparison:")
    print(global_summary.to_string(index=False))
    print("\nMain report:", out_dir / "pathway_validation_report_v7.html")


if __name__ == "__main__":
    main()
