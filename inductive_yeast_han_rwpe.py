#!/usr/bin/env python3
"""
inductive_yeast_han_rwpe.py
────────────────────────────────────────────────────────────────────────────
Cross-Species Inductive HAN + Random Walk Positional Encoding (RWPE)
for Alopecia Candidate Metabolite Discovery

Novelty over inductive_Yeast_HAN_new_MMR.ipynb
  • RWPE computed from S. cerevisiae reaction graph
    (All-pathways-of-S-cerevisiae-S288c_cleaned_compounds.txt)
  • Metabolite node feature = concat(Morgan FP [2048], RWPE [K=20])
    → dim 2068 total
  • Inductive HAN trained for link prediction (metabolite→drug, drug→target)
  • Currency-metabolite awareness: RWPE naturally differentiates hub
    nodes (H2O, ATP, …) from terminal sterol metabolites

Usage (same directory as the pathway .txt file):
  python inductive_yeast_han_rwpe.py
  python inductive_yeast_han_rwpe.py --walk-length 20 --epochs 50

Requires: rdkit, torch-geometric, scipy, scikit-learn, tqdm, requests, matplotlib
────────────────────────────────────────────────────────────────────────────
"""

import os, re, sys, time, random, warnings
import argparse
import requests
import numpy as np
import pandas as pd
import scipy.sparse as sp
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
from collections import defaultdict
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    roc_curve, precision_recall_curve,
)
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import HANConv
from torch_geometric.loader import LinkNeighborLoader
import matplotlib
matplotlib.use("Agg")          # headless-safe; change to "TkAgg" if you want popups
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 0.  CLI + Global Config
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pathway-file", default=os.path.join(
        SCRIPT_DIR,
        "All-pathways-of-S-cerevisiae-S288c_cleaned_compounds.txt"),
        help="Path to S. cerevisiae pathway TSV")
    p.add_argument("--walk-length", type=int, default=20,
        help="RWPE walk length K (default 20)")
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--hidden",      type=int, default=64)
    p.add_argument("--heads",       type=int, default=4)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--top-k",       type=int,   default=5,   help="Recall@K")
    p.add_argument("--no-plot",     action="store_true",
        help="Skip matplotlib plots (useful for headless runs)")
    return p.parse_args()

args      = parse_args()
SEED      = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WALK_LEN     = args.walk_length
FP_DIM       = 1024
MORGAN_RADIUS = 2                        # Morgan fingerprint radius
MET_DIM      = FP_DIM + WALK_LEN        # metabolite feature dimension
print(f"DEVICE: {DEVICE}   RWPE walk_length={WALK_LEN}   metabolite_dim={MET_DIM}")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Dataset
#     • gene → metabolite  : curated S. cerevisiae sterol-pathway (toy)
#     • drug → target       : real ChEMBL data  (real_drug_target_edges.csv)
#     • target → disease    : real OpenTargets data (real_target_disease_edges.csv)
# ══════════════════════════════════════════════════════════════════════════════

# --- Gene → Metabolite (curated toy, sterol/mevalonate pathway) ---
gene_metabolite_edges = pd.DataFrame([
    # Mevalonate pathway
    ["ERG10", "acetoacetyl-CoA"],
    ["ERG13", "3-hydroxy-3-methylglutaryl-CoA"],
    ["HMG1",  "mevalonic acid"],
    ["HMG2",  "mevalonic acid"],
    ["ERG12", "mevalonate 5-phosphate"],
    ["ERG8",  "mevalonate 5-diphosphate"],
    ["ERG19", "isopentenyl pyrophosphate"],
    # Isoprenoid
    ["IDI1",  "dimethylallyl pyrophosphate"],
    ["IDI1",  "isopentenyl pyrophosphate"],
    ["ERG20", "geranyl pyrophosphate"],
    ["ERG20", "farnesyl pyrophosphate"],
    ["BTS1",  "geranylgeranyl pyrophosphate"],
    # Sterol biosynthesis
    ["ERG9",  "squalene"],
    ["ERG1",  "2,3-oxidosqualene"],
    ["ERG7",  "lanosterol"],
    ["ERG11", "lanosterol"],
    ["ERG24", "zymosterol"],
    ["ERG6",  "fecosterol"],
    ["ERG2",  "episterol"],
    ["ERG3",  "ergosterol"],
    ["ERG5",  "ergosterol"],
    ["ERG4",  "ergosterol"],
    # Cross-species structural analogues
    ["ERG11", "cholesterol"],
    ["ERG24", "desmosterol"],
    ["ERG6",  "campesterol"],
    ["ERG6",  "stigmasterol"],
    ["ERG5",  "beta-sitosterol"],
    # Small isoprenoid alcohols
    ["ERG20", "geraniol"],
    ["ERG20", "farnesol"],
    ["BTS1",  "geranylgeraniol"],
], columns=["gene_id", "metabolite_name"])

# --- Drug → Target  (real ChEMBL data) ---
_dt_path = os.path.join(SCRIPT_DIR, "real_drug_target_edges.csv")
_dt_raw  = pd.read_csv(_dt_path)
drug_target_edges = (_dt_raw[["drug_id","target_symbol","confidence"]]
                     .rename(columns={"drug_id":"drug_name","target_symbol":"target_id"})
                     .copy())
print(f"Loaded drug-target edges: {len(drug_target_edges)} rows  "
      f"({drug_target_edges['drug_name'].nunique()} drugs, "
      f"{drug_target_edges['target_id'].nunique()} targets)")

# --- Target → Disease  (real OpenTargets data) ---
_td_path = os.path.join(SCRIPT_DIR, "real_target_disease_edges.csv")
_td_raw  = pd.read_csv(_td_path)
target_disease_edges = (_td_raw[["target_symbol","disease_id","disease_name","association_score"]]
                        .rename(columns={"target_symbol":"target_id"})
                        .copy())
# Normalise disease_id to "ALOPECIA" for compatibility with downstream ranking code
target_disease_edges["disease_id"]   = "ALOPECIA"
target_disease_edges["disease_name"] = "Alopecia"
print(f"Loaded target-disease edges: {len(target_disease_edges)} rows  "
      f"({target_disease_edges['target_id'].nunique()} targets)")

# --- Node tables ---
genes_df       = pd.DataFrame({"gene_id":    sorted(gene_metabolite_edges["gene_id"].unique())})
metabolites_df = pd.DataFrame({"metabolite_name": sorted(gene_metabolite_edges["metabolite_name"].unique())})
drugs_df       = pd.DataFrame({"drug_name":  sorted(drug_target_edges["drug_name"].unique())})
targets_df     = pd.DataFrame({"target_id":  sorted(
    set(drug_target_edges["target_id"]) | set(target_disease_edges["target_id"]))})
diseases_df    = pd.DataFrame({"disease_id": ["ALOPECIA"], "disease_name": ["Alopecia"]})

print(f"Graph nodes  genes={len(genes_df)}  metabolites={len(metabolites_df)}  "
      f"drugs={len(drugs_df)}  targets={len(targets_df)}  diseases={len(diseases_df)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Compound name synonym mapping  (toy-dataset names → pathway-file names)
#     Used for both SMILES lookup AND RWPE lookup.
# ══════════════════════════════════════════════════════════════════════════════
COMPOUND_SYNONYMS = {
    # pyrophosphate ↔ diphosphate
    "isopentenyl pyrophosphate":     "isopentenyl diphosphate",
    "dimethylallyl pyrophosphate":   "dimethylallyl diphosphate",
    "geranyl pyrophosphate":         "geranyl diphosphate",
    "farnesyl pyrophosphate":        "(2e,6e)-farnesyl diphosphate",
    "geranylgeranyl pyrophosphate":  "geranylgeranyl diphosphate",
    # common name ↔ IUPAC/stereochemical name
    "3-hydroxy-3-methylglutaryl-coa":  "(s)-3-hydroxy-3-methylglutaryl-coa",
    "mevalonic acid":                  "(r)-mevalonate",
    "mevalonate 5-phosphate":          "(r)-5-phosphomevalonate",
    "mevalonate 5-diphosphate":        "(r)-mevalonate diphosphate",
    "2,3-oxidosqualene":               "(3s)-2,3-epoxy-2,3-dihydrosqualene",
    "farnesol":                        "(2e,6e)-farnesol",
}

# ══════════════════════════════════════════════════════════════════════════════
# 3.  SMILES
#     • Metabolites  → read directly from pathway file (column 4, no network)
#     • Drugs        → PubChem / ChEMBL fetch
# ══════════════════════════════════════════════════════════════════════════════
def load_pathway_compound_smiles(pathway_file: str) -> dict:
    """
    Parse pathway TSV and return  {compound_name_lower: SMILES}.
    Column layout (tab-separated):
        Pathway | Compounds (// sep) | Reactions (// sep) | SMILES (// sep)
    The i-th SMILES corresponds to the i-th compound name.
    """
    comp_smiles: dict = {}
    with open(pathway_file, encoding="utf-8") as fh:
        next(fh)   # skip header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            names_col  = [c.strip() for c in parts[1].split(" // ")]
            smiles_col = [s.strip() for s in parts[3].split(" // ")]
            for name, smi in zip(names_col, smiles_col):
                key = name.lower()
                if key and smi and key not in comp_smiles:
                    comp_smiles[key] = smi
    print(f"Pathway SMILES dict: {len(comp_smiles)} unique compounds")
    return comp_smiles

def lookup_smiles_from_pathway(name: str, pathway_smiles: dict) -> str | None:
    key = name.lower()
    resolved = COMPOUND_SYNONYMS.get(key, key)
    return pathway_smiles.get(resolved) or pathway_smiles.get(key)

# --- Metabolites: pathway file (offline, no network) ---
print("\n--- Loading metabolite SMILES from pathway file ---")
pathway_compound_smiles = load_pathway_compound_smiles(args.pathway_file)

metabolites_df["smiles"] = metabolites_df["metabolite_name"].apply(
    lambda n: lookup_smiles_from_pathway(n, pathway_compound_smiles))
met_found = metabolites_df["smiles"].notna().sum()
print(f"Metabolite SMILES matched: {met_found}/{len(metabolites_df)}")

# --- Drugs: PubChem / ChEMBL (network required) ---
def fetch_single_smiles(name: str, retries: int = 3):
    """Fetch SMILES from PubChem (or ChEMBL) with retry + exponential back-off."""
    if name.upper().startswith("CHEMBL"):
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{name.upper()}.json"
        for attempt in range(retries):
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    s = r.json().get("molecule_structures") or {}
                    sm = s.get("canonical_smiles") or s.get("standard_smiles")
                    if sm: return name, sm
                elif r.status_code == 429:
                    time.sleep(2 ** attempt); continue
            except Exception:
                time.sleep(2 ** attempt)
        return name, None

    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           f"{quote(name)}/property/CanonicalSMILES,IsomericSMILES/JSON")
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                props = r.json().get("PropertyTable", {}).get("Properties", [])
                if props:
                    sm = props[0].get("CanonicalSMILES") or props[0].get("IsomericSMILES")
                    if sm: return name, sm
                break
            elif r.status_code == 429:
                time.sleep(2 ** attempt + 1); continue
            elif r.status_code == 404:
                break
        except Exception:
            time.sleep(2 ** attempt)
    return name, None

def fetch_smiles_concurrently(names, max_workers=5):
    results = []
    def _fetch(n):
        res = fetch_single_smiles(n)
        time.sleep(0.3)
        return res
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch, n): n for n in set(names)}
        for f in tqdm(as_completed(futs), total=len(futs), desc="Fetching SMILES"):
            n, sm = f.result()
            results.append({"Name": n, "smiles": sm})
    df = pd.DataFrame(results)
    print(f"  → {df['smiles'].notna().sum()}/{len(df)} SMILES retrieved")
    return df

print(f"--- Fetching drug SMILES from ChEMBL ({len(drugs_df)} drugs) ---")
drug_smiles_df = fetch_smiles_concurrently(drugs_df["drug_name"].tolist())
tmp = drug_smiles_df.copy()
tmp["key"] = tmp["Name"].str.lower()
drugs_df["key"] = drugs_df["drug_name"].str.lower()
drugs_df = drugs_df.merge(tmp[["key","smiles"]], on="key", how="left").drop(columns=["key"])

# Drop rows with no SMILES
metabolites_df = metabolites_df.dropna(subset=["smiles"]).reset_index(drop=True)
drugs_df       = drugs_df.dropna(subset=["smiles"]).reset_index(drop=True)
gene_metabolite_edges = gene_metabolite_edges[
    gene_metabolite_edges["metabolite_name"].isin(metabolites_df["metabolite_name"])
].reset_index(drop=True)
drug_target_edges = drug_target_edges[
    drug_target_edges["drug_name"].isin(drugs_df["drug_name"])
].reset_index(drop=True)

print(f"\nAfter SMILES filter  metabolites={len(metabolites_df)}  drugs={len(drugs_df)}")
if len(metabolites_df) == 0:
    print("[ERROR] No metabolite SMILES found in pathway file. Check file path.")
    sys.exit(1)
if len(drugs_df) == 0:
    print("[ERROR] No drug SMILES retrieved from PubChem. Check network access.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Morgan Fingerprints
# ══════════════════════════════════════════════════════════════════════════════
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=FP_DIM)

def smiles_to_fp(smi):
    if pd.isna(smi): return None
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None: return None
    return morgan_gen.GetFingerprint(mol)

def fp_to_tensor(fp):
    arr = np.zeros(FP_DIM, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return torch.tensor(arr, dtype=torch.float32)

metabolites_df["fp"] = metabolites_df["smiles"].apply(smiles_to_fp)
drugs_df["fp"]       = drugs_df["smiles"].apply(smiles_to_fp)
metabolites_df = metabolites_df[metabolites_df["fp"].notna()].reset_index(drop=True)
drugs_df       = drugs_df[drugs_df["fp"].notna()].reset_index(drop=True)
print(f"Valid fingerprints  metabolites={len(metabolites_df)}  drugs={len(drugs_df)}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  RWPE – Random Walk Positional Encoding from Pathway Reaction Graph
# ══════════════════════════════════════════════════════════════════════════════
def _parse_side(side_str: str) -> list:
    """Split one half of a reaction string on '+', strip stoichiometry."""
    out = []
    for tok in side_str.split("+"):
        tok = tok.strip()
        tok = re.sub(r"^\d+\s+", "", tok)   # remove "2 " prefix
        if tok:
            out.append(tok.lower())
    return out

def _split_reaction(rxn: str):
    """Return (left_compounds, right_compounds) for a reaction string."""
    rxn = rxn.strip()
    # reversible  <-->  or  <->
    m = re.search(r"<-+>", rxn)
    if m:
        left_str  = rxn[:m.start()]
        right_str = rxn[m.end():]
        return _parse_side(left_str), _parse_side(right_str)
    # reverse  <-
    if "<-" in rxn:
        parts = rxn.split("<-", 1)
        return _parse_side(parts[1]), _parse_side(parts[0])
    # forward  ->
    if "->" in rxn:
        parts = rxn.split("->", 1)
        return _parse_side(parts[0]), _parse_side(parts[1])
    return [], []

def build_pathway_graph(pathway_file: str):
    """
    Parse S. cerevisiae pathway TSV and build a metabolite–metabolite
    reaction graph (undirected).

    Returns
    -------
    comp2idx : dict  name(lower) → node_id
    A        : scipy.sparse.csr_matrix  (N×N) binary adjacency
    """
    comp2idx = {}   # lowercase compound name → int
    edges_src, edges_dst = [], []

    with open(pathway_file, encoding="utf-8") as fh:
        next(fh)   # skip header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            reactions_field = parts[2]
            for rxn in reactions_field.split(" // "):
                left, right = _split_reaction(rxn)
                if not left or not right:
                    continue
                # Connect every substrate ↔ every product
                for s in left:
                    for p in right:
                        if s == p:
                            continue
                        for c in (s, p):
                            if c not in comp2idx:
                                comp2idx[c] = len(comp2idx)
                        si, pi = comp2idx[s], comp2idx[p]
                        edges_src += [si, pi]
                        edges_dst += [pi, si]

    N = len(comp2idx)
    if N == 0:
        return comp2idx, sp.csr_matrix((0, 0))

    data = np.ones(len(edges_src), dtype=np.float32)
    A = sp.csr_matrix((data, (edges_src, edges_dst)), shape=(N, N))
    A = (A > 0).astype(np.float32)   # binarise
    print(f"Pathway graph  nodes={N}  edges={A.nnz // 2}")
    return comp2idx, A

def compute_rwpe_matrix(A: sp.csr_matrix, K: int) -> np.ndarray:
    """
    Compute Random Walk PE matrix.

    RWPE[i, k] = (T^{k+1})[i,i],  k = 0 … K-1
    where  T = D^{-1} A  (row-stochastic transition matrix)

    Interpretation: the probability of returning to node i after k+1 steps.
    High values → well-connected / hub node (currency metabolites).
    Low values  → terminal / specific metabolite (sterol intermediates).

    Returns numpy array shape (N, K).
    """
    N = A.shape[0]
    degree = np.array(A.sum(axis=1), dtype=np.float64).flatten()
    degree[degree == 0] = 1.0
    D_inv = sp.diags(1.0 / degree)
    T = (D_inv @ A).astype(np.float32)

    rwpe = np.zeros((N, K), dtype=np.float32)
    T_k  = T.copy()
    for k in range(K):
        rwpe[:, k] = np.array(T_k.diagonal()).flatten()
        if k < K - 1:
            T_k = T_k @ T   # sparse-sparse product; fill-in grows but stays sparse
    return rwpe

def get_metabolite_rwpe(
    metabolite_names: list,
    comp2idx: dict,
    rwpe_full: np.ndarray,
    K: int,
) -> np.ndarray:
    """
    Look up RWPE vectors for a list of metabolite names.
    Returns shape (N_metabolites, K).

    Uses module-level COMPOUND_SYNONYMS to bridge toy-dataset naming
    (common names / pyrophosphate) to BioCyc pathway-file names.
    Unmatched metabolites get a zero vector.
    """
    out = np.zeros((len(metabolite_names), K), dtype=np.float32)
    found = 0
    for i, name in enumerate(metabolite_names):
        key = name.lower()
        resolved = COMPOUND_SYNONYMS.get(key, key)   # apply synonym table
        if resolved in comp2idx:
            out[i] = rwpe_full[comp2idx[resolved]]
            found += 1
    print(f"RWPE lookup: {found}/{len(metabolite_names)} metabolites matched "
          f"(unmatched → zero vector)")
    return out

# Build graph and compute RWPE
print(f"\n--- Building S. cerevisiae pathway reaction graph ---")
comp2idx, A_pathway = build_pathway_graph(args.pathway_file)

print("--- Computing RWPE (this may take ~10–30 s for K=20) ---")
if A_pathway.shape[0] > 0:
    rwpe_full = compute_rwpe_matrix(A_pathway, WALK_LEN)
else:
    print("[WARNING] Pathway graph is empty – RWPE set to zeros.")
    rwpe_full = np.zeros((0, WALK_LEN), dtype=np.float32)

# Fetch RWPE for our metabolites
rwpe_met = get_metabolite_rwpe(
    metabolites_df["metabolite_name"].tolist(),
    comp2idx, rwpe_full, WALK_LEN
)
metabolites_df["rwpe"] = list(rwpe_met)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Tanimoto-based Metabolite → Drug Latent Edges  (Top-K)
# ══════════════════════════════════════════════════════════════════════════════
TOP_K_TANIMOTO = 5
sim_rows = []
drug_names_list = drugs_df["drug_name"].tolist()
drug_fps_list   = drugs_df["fp"].tolist()

for row in tqdm(metabolites_df.itertuples(index=False), total=len(metabolites_df),
                desc="Tanimoto similarity"):
    sims = DataStructs.BulkTanimotoSimilarity(row.fp, drug_fps_list)
    for j in np.argsort(sims)[::-1][:TOP_K_TANIMOTO]:
        sim_rows.append({"metabolite_name": row.metabolite_name,
                         "drug_name": drug_names_list[j],
                         "tanimoto": float(sims[j])})

metabolite_drug_edges = pd.DataFrame(sim_rows)
print(f"Metabolite–Drug Tanimoto edges: {len(metabolite_drug_edges)}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Index Mappings
# ══════════════════════════════════════════════════════════════════════════════
def make_mapping(values):
    vals = sorted(pd.Series(values).dropna().astype(str).unique())
    id2idx = {v: i for i, v in enumerate(vals)}
    idx2id = {i: v for v, i in id2idx.items()}
    return id2idx, idx2id

gene2idx,    idx2gene    = make_mapping(genes_df["gene_id"])
met2idx,     idx2met     = make_mapping(metabolites_df["metabolite_name"])
drug2idx,    idx2drug    = make_mapping(drugs_df["drug_name"])
target2idx,  idx2target  = make_mapping(targets_df["target_id"])
disease2idx, idx2disease = make_mapping(diseases_df["disease_id"])


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Heterogeneous Graph Construction
# ══════════════════════════════════════════════════════════════════════════════
def build_edge_index(df, src_col, dst_col, src_map, dst_map):
    srcs, dsts = [], []
    for _, row in df.iterrows():
        s, d = str(row[src_col]), str(row[dst_col])
        if s in src_map and d in dst_map:
            srcs.append(src_map[s]); dsts.append(dst_map[d])
    if not srcs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([srcs, dsts], dtype=torch.long)

data = HeteroData()
data["gene"].num_nodes     = len(gene2idx)
data["metabolite"].num_nodes = len(met2idx)
data["drug"].num_nodes     = len(drug2idx)
data["target"].num_nodes   = len(target2idx)
data["disease"].num_nodes  = len(disease2idx)

gm  = build_edge_index(gene_metabolite_edges, "gene_id", "metabolite_name", gene2idx,   met2idx)
md  = build_edge_index(metabolite_drug_edges,  "metabolite_name", "drug_name", met2idx, drug2idx)
dt  = build_edge_index(drug_target_edges,       "drug_name", "target_id",  drug2idx,    target2idx)
td_ = build_edge_index(target_disease_edges,    "target_id", "disease_id", target2idx,  disease2idx)

data["gene",       "produces",       "metabolite"].edge_index = gm
data["metabolite", "rev_produces",   "gene"      ].edge_index = gm.flip(0)
data["metabolite", "similar_to",     "drug"      ].edge_index = md
data["drug",       "rev_similar_to", "metabolite"].edge_index = md.flip(0)
data["drug",       "inhibits",       "target"    ].edge_index = dt
data["target",     "rev_inhibits",   "drug"      ].edge_index = dt.flip(0)
data["target",     "associated_with","disease"   ].edge_index = td_
data["disease",    "rev_associated_with","target"].edge_index = td_.flip(0)

print("\nHeteroData:", data)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Node Features  (Morgan FP  +  RWPE for metabolites)
# ══════════════════════════════════════════════════════════════════════════════
# Drug features: Morgan FP only (2048-d) – unchanged for inductive new-drug inference
drug_x = torch.stack([fp_to_tensor(fp) for fp in drugs_df["fp"]])
data["drug"].x = drug_x
print("drug.x     :", data["drug"].x.shape)

# Metabolite features: concat(Morgan FP, RWPE)  →  (N_met, 2048+K)
met_fps  = torch.stack([fp_to_tensor(fp) for fp in metabolites_df["fp"]])
met_rwpe = torch.tensor(rwpe_met, dtype=torch.float32)
metabolite_x = torch.cat([met_fps, met_rwpe], dim=-1)
data["metabolite"].x = metabolite_x
print("metabolite.x:", data["metabolite"].x.shape,
      f"  (Morgan {FP_DIM} + RWPE {WALK_LEN})")

# Gene / target / disease: constant 1-d feature
data["gene"].x    = torch.ones((data["gene"].num_nodes,    1))
data["target"].x  = torch.ones((data["target"].num_nodes,  1))
data["disease"].x = torch.ones((data["disease"].num_nodes, 1))


# ══════════════════════════════════════════════════════════════════════════════
# 9.  Train / Val / Test Split  +  DataLoaders
# ══════════════════════════════════════════════════════════════════════════════
TASK_EDGE_TYPES = [
    ("metabolite", "similar_to", "drug"),
    ("drug",       "inhibits",   "target"),
]
REV_EDGE_TYPES = [
    ("drug",   "rev_similar_to", "metabolite"),
    ("target", "rev_inhibits",   "drug"),
]

transform = RandomLinkSplit(
    num_val=0.2, num_test=0.2,
    is_undirected=False,
    add_negative_train_samples=True,
    neg_sampling_ratio=1.0,
    edge_types=TASK_EDGE_TYPES,
    rev_edge_types=REV_EDGE_TYPES,
)
train_data, val_data, test_data = transform(data)

def make_loaders(split_data, edge_types, shuffle=False):
    loaders = []
    for et in edge_types:
        if et in split_data.edge_types and hasattr(split_data[et], "edge_label_index"):
            loaders.append(LinkNeighborLoader(
                split_data,
                num_neighbors=[10, 5],
                edge_label_index=(et, split_data[et].edge_label_index),
                edge_label=split_data[et].edge_label,
                batch_size=128,
                shuffle=shuffle,
                neg_sampling_ratio=1.0 if shuffle else 0.0,
            ))
    return loaders

train_loader = make_loaders(train_data, TASK_EDGE_TYPES, shuffle=True)
val_loader   = make_loaders(val_data,   TASK_EDGE_TYPES)
test_loader  = make_loaders(test_data,  TASK_EDGE_TYPES)


# ══════════════════════════════════════════════════════════════════════════════
# 10.  Inductive HAN Model
# ══════════════════════════════════════════════════════════════════════════════
class InductiveHANLinkPredictor(nn.Module):
    """
    Two-layer Heterogeneous Attention Network with residual connections.
    Each node type gets its own linear input projection.
    """
    def __init__(self, metadata, in_channels_dict,
                 hidden_channels=64, out_channels=64, heads=4, dropout=0.1):
        super().__init__()
        self.node_proj = nn.ModuleDict({
            nt: nn.Linear(ic, hidden_channels)
            for nt, ic in in_channels_dict.items()
        })
        self.conv1 = HANConv(hidden_channels, hidden_channels,
                             metadata=metadata, heads=heads, dropout=dropout)
        self.conv2 = HANConv(hidden_channels, out_channels,
                             metadata=metadata, heads=heads, dropout=dropout)

    def forward(self, data):
        x = {nt: self.node_proj[nt](feat)
             for nt, feat in data.x_dict.items()}

        # Layer 1 + residual
        o1 = self.conv1(x, data.edge_index_dict)
        x1 = {k: F.elu(o1.get(k, x[k]) + x[k]) for k in x}

        # Layer 2 + residual (no activation on final)
        o2 = self.conv2(x1, data.edge_index_dict)
        x2 = {k: o2.get(k, x1[k]) + x1[k] for k in x1}
        return x2

    def decode(self, z_dict, edge_label_index, src_type, dst_type):
        src = z_dict[src_type][edge_label_index[0]]
        dst = z_dict[dst_type][edge_label_index[1]]
        return (src * dst).sum(dim=-1)

in_channels_dict = {nt: data[nt].x.size(-1) for nt in data.node_types}
print("\nin_channels_dict:", in_channels_dict)

model = InductiveHANLinkPredictor(
    metadata=data.metadata(),
    in_channels_dict=in_channels_dict,
    hidden_channels=args.hidden,
    out_channels=args.hidden,
    heads=args.heads,
    dropout=0.1,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
print(model)


# ══════════════════════════════════════════════════════════════════════════════
# 11.  Training + Evaluation Utilities
# ══════════════════════════════════════════════════════════════════════════════
def compute_loss(model, batch, edge_types):
    z = model(batch)
    losses = []
    for et in edge_types:
        if et not in batch.edge_types or not hasattr(batch[et], "edge_label_index"):
            continue
        src, _, dst = et
        logits = model.decode(z, batch[et].edge_label_index, src, dst)
        losses.append(F.binary_cross_entropy_with_logits(
            logits, batch[et].edge_label.float()))
    if not losses:
        return torch.tensor(0.0, device=next(model.parameters()).device,
                            requires_grad=True)
    return sum(losses) / len(losses)

@torch.no_grad()
def evaluate(model, loaders, edge_types, device):
    model.eval()
    y_true_d, y_prob_d = defaultdict(list), defaultdict(list)
    for loader, et in zip(loaders, edge_types):
        for batch in loader:
            batch = batch.to(device)
            if et not in batch.edge_types or not hasattr(batch[et], "edge_label_index"):
                continue
            src, _, dst = et
            z = model(batch)
            logits = model.decode(z, batch[et].edge_label_index, src, dst)
            y_prob_d[et].extend(torch.sigmoid(logits).cpu().numpy())
            y_true_d[et].extend(batch[et].edge_label.float().cpu().numpy())
    res = {}
    for et in edge_types:
        yt = np.array(y_true_d.get(et, []))
        yp = np.array(y_prob_d.get(et, []))
        if len(yt) == 0 or len(np.unique(yt)) < 2:
            continue
        res[et] = {
            "AUC": roc_auc_score(yt, yp),
            "AP":  average_precision_score(yt, yp),
            "F1":  f1_score(yt, (yp > 0.5).astype(int), zero_division=0),
        }
    return res

@torch.no_grad()
def recall_at_k(model, split_data, edge_type, k=3, device=DEVICE):
    model.eval()
    split_data = split_data.to(device)
    if edge_type not in split_data.edge_types:
        return np.nan
    if not hasattr(split_data[edge_type], "edge_label_index"):
        return np.nan
    src_t, _, dst_t = edge_type
    z = model(split_data)
    eli   = split_data[edge_type].edge_label_index
    elabel = split_data[edge_type].edge_label
    pos_idx = eli[:, elabel > 0.5]
    if pos_idx.size(1) == 0:
        return np.nan
    dst_all = z[dst_t]
    hits = 0
    for i in range(pos_idx.size(1)):
        src_vec = z[src_t][pos_idx[0, i]]
        scores  = torch.matmul(dst_all, src_vec)
        topk    = torch.topk(scores, k=min(k, dst_all.size(0))).indices
        if (topk == pos_idx[1, i]).any():
            hits += 1
    return hits / pos_idx.size(1)


# ══════════════════════════════════════════════════════════════════════════════
# 12.  Training Loop
# ══════════════════════════════════════════════════════════════════════════════
valid_train_ets = [et for et in TASK_EDGE_TYPES
                   if et in train_data.edge_types
                   and hasattr(train_data[et], "edge_label_index")]
valid_val_ets   = [et for et in TASK_EDGE_TYPES
                   if et in val_data.edge_types
                   and hasattr(val_data[et], "edge_label_index")]

history = {"train_loss": [], "val_ap": [], "epochs_logged": []}
best_val_ap, best_state = -1.0, None

print(f"\n=== Training ({args.epochs} epochs) ===")
for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss, n_batches = 0.0, 0
    iterators = [(et, iter(ldr)) for et, ldr in zip(valid_train_ets, train_loader)]
    active = iterators[:]
    while active:
        et, it = active.pop(0)
        try:
            batch = next(it).to(DEVICE)
            active.append((et, it))
        except StopIteration:
            continue
        optimizer.zero_grad()
        loss = compute_loss(model, batch, [et])
        if loss.item() > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    history["train_loss"].append(avg_loss)

    if epoch % 5 == 0:
        val_res = evaluate(model, val_loader, valid_val_ets, DEVICE)
        mean_ap = float(np.nanmean([v["AP"] for v in val_res.values()])) if val_res else 0.0
        history["val_ap"].append(mean_ap); history["epochs_logged"].append(epoch)
        if mean_ap > best_val_ap:
            best_val_ap = mean_ap
            best_state  = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:03d}  loss={avg_loss:.4f}  val_AP={mean_ap:.4f}", end="")
        for et, m in val_res.items():
            print(f"  {et[1]}: AUC={m['AUC']:.3f} AP={m['AP']:.3f} F1={m['F1']:.3f}", end="")
        print()

if best_state is not None:
    model.load_state_dict(best_state)
print("Training done.")


# ══════════════════════════════════════════════════════════════════════════════
# 13.  Test Evaluation + Recall@K
# ══════════════════════════════════════════════════════════════════════════════
valid_test_ets = [et for et in TASK_EDGE_TYPES
                  if et in test_data.edge_types
                  and hasattr(test_data[et], "edge_label_index")]
test_res = evaluate(model, test_loader, valid_test_ets, DEVICE)
print("\n=== Test Results ===")
for et, m in test_res.items():
    print(f"  {et}: AUC={m['AUC']:.4f}  AP={m['AP']:.4f}  F1={m['F1']:.4f}")

K_recall = args.top_k
for et in TASK_EDGE_TYPES:
    r = recall_at_k(model, test_data, et, k=K_recall)
    print(f"  Recall@{K_recall} [{et[1]}] = {r:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 14.  Metabolite Ranking for Disease (ALOPECIA)
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def rank_metabolites_for_disease(model, full_data, disease_id="ALOPECIA", top_k=100):
    model.eval()
    full_data = full_data.to(DEVICE)
    z = model(full_data)
    if disease_id not in disease2idx:
        raise ValueError(f"{disease_id} not in disease2idx")
    d_vec = z["disease"][disease2idx[disease_id]]
    logits = torch.matmul(z["metabolite"], d_vec)
    probs  = torch.sigmoid(logits)
    top    = torch.topk(probs, k=min(top_k, probs.size(0)))
    return pd.DataFrame([
        {"rank": r+1, "metabolite_name": idx2met[i],
         "disease_id": disease_id, "metabolite_gnn_score": float(s)}
        for r, (i, s) in enumerate(zip(top.indices.tolist(), top.values.tolist()))
    ])

metabolite_rank_df = rank_metabolites_for_disease(model, data, "ALOPECIA", top_k=100)
print("\n=== Top-10 Metabolites for ALOPECIA (GNN score) ===")
print(metabolite_rank_df.head(10).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 15.  Path Explanation  (gene → metabolite → drug → target → disease)
# ══════════════════════════════════════════════════════════════════════════════
def build_path_explanation():
    rows = []
    for _, gm_row in gene_metabolite_edges.iterrows():
        gene, met = gm_row["gene_id"], gm_row["metabolite_name"]
        for _, sim in metabolite_drug_edges[
                metabolite_drug_edges["metabolite_name"] == met].iterrows():
            drug, tan = sim["drug_name"], sim["tanimoto"]
            for _, dt_row in drug_target_edges[
                    drug_target_edges["drug_name"] == drug].iterrows():
                tgt, dtc = dt_row["target_id"], dt_row["confidence"]
                for _, td_row in target_disease_edges[
                        target_disease_edges["target_id"] == tgt].iterrows():
                    rows.append({
                        "gene_id":             gene,
                        "metabolite_name":     met,
                        "similar_drug":        drug,
                        "target":              tgt,
                        "disease":             td_row["disease_id"],
                        "tanimoto":            tan,
                        "drug_target_confidence": dtc,
                        "target_disease_score":   td_row["association_score"],
                        "path_score":          tan * dtc * td_row["association_score"],
                    })
    return pd.DataFrame(rows)

path_df = build_path_explanation()
print(f"\nPath explanation rows: {len(path_df)}")


# ══════════════════════════════════════════════════════════════════════════════
# 16.  Combined Ranking  (GNN score + path score)
# ══════════════════════════════════════════════════════════════════════════════
def combine_gnn_and_path(metabolite_rank_df, path_df,
                         disease_id="ALOPECIA",
                         w_gnn=0.5, w_path=0.5):
    alopecia_path = path_df[path_df["disease"] == disease_id].copy()
    met_path = (alopecia_path.groupby("metabolite_name")
                .agg(path_score=("path_score","max"),
                     num_supporting_genes=("gene_id","nunique"),
                     num_paths=("path_score","count"))
                .reset_index())
    if len(met_path) > 0 and met_path["path_score"].max() > 0:
        met_path["path_score_scaled"] = met_path["path_score"] / met_path["path_score"].max()
    else:
        met_path["path_score_scaled"] = 0.0

    df = metabolite_rank_df.merge(
        met_path[["metabolite_name","path_score","path_score_scaled",
                  "num_supporting_genes","num_paths"]],
        on="metabolite_name", how="left")
    for col in ["path_score","path_score_scaled"]:
        df[col] = df[col].fillna(0.0)
    df["num_supporting_genes"] = df["num_supporting_genes"].fillna(0).astype(int)
    df["num_paths"]             = df["num_paths"].fillna(0).astype(int)
    df["final_score"] = w_gnn * df["metabolite_gnn_score"] + w_path * df["path_score_scaled"]
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df["final_rank"] = np.arange(1, len(df)+1)
    return df

final_metabolite_rank_df = combine_gnn_and_path(metabolite_rank_df, path_df)
print("\n=== Final Combined Ranking (Top-10) ===")
print(final_metabolite_rank_df[
    ["final_rank","metabolite_name","metabolite_gnn_score",
     "path_score_scaled","num_supporting_genes","final_score"]
].head(10).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 17.  MMR Re-ranking  (drug-profile diversity)
# ══════════════════════════════════════════════════════════════════════════════
def build_drug_profile_sim_matrix(candidates, metabolite_drug_edges):
    pivot = (metabolite_drug_edges
             .pivot_table(index="metabolite_name", columns="drug_name",
                          values="tanimoto", aggfunc="max", fill_value=0.0)
             .reindex(candidates["metabolite_name"].tolist()).fillna(0.0))
    P = pivot.to_numpy(dtype=float)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    P_norm = np.divide(P, norms, out=np.zeros_like(P), where=(norms != 0))
    return P_norm @ P_norm.T

def mmr_rerank(rank_df, metabolite_drug_edges,
               score_col="final_score", top_k=10,
               candidate_k=100, lambda_mmr=0.7):
    cands = (rank_df.sort_values(score_col, ascending=False)
             .head(min(candidate_k, len(rank_df)))
             .copy().reset_index(drop=True))
    if len(cands) == 0:
        return cands
    sim = build_drug_profile_sim_matrix(cands, metabolite_drug_edges)
    rel = cands[score_col].astype(float).to_numpy()
    if rel.max() > rel.min():
        rel = (rel - rel.min()) / (rel.max() - rel.min())
    else:
        rel = np.ones_like(rel)

    selected, remaining, scores = [], list(range(len(cands))), []
    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda i: rel[i])
        else:
            best = max(remaining,
                       key=lambda i: lambda_mmr * rel[i]
                                     - (1 - lambda_mmr) * max(sim[i, j] for j in selected))
        mmr_s = (rel[best] if not selected else
                 lambda_mmr * rel[best]
                 - (1 - lambda_mmr) * max(sim[best, j] for j in selected))
        selected.append(best); scores.append(mmr_s); remaining.remove(best)

    out = cands.iloc[selected].copy().reset_index(drop=True)
    out["mmr_rank"]  = np.arange(1, len(out)+1)
    out["mmr_score"] = scores
    return out

mmr_rank_df = mmr_rerank(final_metabolite_rank_df, metabolite_drug_edges,
                          top_k=10, candidate_k=100, lambda_mmr=0.7)
print("\n=== MMR Re-ranked Top-10 ===")
print(mmr_rank_df[["mmr_rank","metabolite_name","metabolite_gnn_score",
                    "path_score_scaled","final_score","mmr_score"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 18.  Path Explanation for MMR-selected Metabolites
# ══════════════════════════════════════════════════════════════════════════════
def show_path_explanation(rank_df, path_df, top_n=5, disease_id="ALOPECIA"):
    alopecia = path_df[path_df["disease"] == disease_id].copy()
    if "metabolite_name" not in alopecia.columns and "metabolite" in alopecia.columns:
        alopecia = alopecia.rename(columns={"metabolite": "metabolite_name"})
    for met in rank_df["metabolite_name"].head(top_n):
        print("=" * 80)
        print(f"  Metabolite: {met}")
        sub = alopecia[alopecia["metabolite_name"] == met].sort_values(
            "path_score", ascending=False)
        if len(sub) == 0:
            print("  No interpretable path found.")
        else:
            print(sub[["gene_id","similar_drug","target","disease",
                        "tanimoto","drug_target_confidence",
                        "target_disease_score","path_score"]].head(5).to_string(index=False))

print("\n=== Path Explanation (MMR top-5) ===")
show_path_explanation(mmr_rank_df, path_df, top_n=5)


# ══════════════════════════════════════════════════════════════════════════════
# 19.  Inductive Inference  –  New Drug SMILES → Metabolite Candidates
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def infer_new_drug(model, base_data, new_drug_smiles,
                   target_edges=None, metabolite_edges=None):
    """
    Add a new drug node (Morgan FP only, 2048-d) to the graph and
    compute its embedding. Metabolite features (FP+RWPE) remain unchanged.
    """
    model.eval()
    fp = smiles_to_fp(new_drug_smiles)
    if fp is None:
        raise ValueError(f"Invalid SMILES: {new_drug_smiles}")
    new_x = fp_to_tensor(fp).unsqueeze(0).to(DEVICE)
    assert new_x.size(1) == base_data["drug"].x.size(1), \
        f"Drug FP dim mismatch: {new_x.size(1)} vs {base_data['drug'].x.size(1)}"

    infer_data = base_data.clone().to(DEVICE)
    new_idx = infer_data["drug"].x.size(0)
    infer_data["drug"].x = torch.cat([infer_data["drug"].x, new_x], dim=0)
    infer_data["drug"].num_nodes = infer_data["drug"].x.size(0)

    def _add_edges(src_list, dst_list, et_key, rev_et_key):
        if not src_list: return
        e = torch.tensor([src_list, dst_list], dtype=torch.long, device=DEVICE)
        infer_data[et_key].edge_index = torch.cat(
            [infer_data[et_key].edge_index, e], dim=1)
        re = e.flip(0)
        infer_data[rev_et_key].edge_index = torch.cat(
            [infer_data[rev_et_key].edge_index, re], dim=1)

    # Optional known drug→target edges
    if target_edges:
        _add_edges([new_idx]*len([t for t in target_edges if t in target2idx]),
                   [target2idx[t] for t in target_edges if t in target2idx],
                   ("drug","inhibits","target"), ("target","rev_inhibits","drug"))

    # Optional known metabolite→drug edges
    if metabolite_edges:
        _add_edges([met2idx[m] for m in metabolite_edges if m in met2idx],
                   [new_idx]*len([m for m in metabolite_edges if m in met2idx]),
                   ("metabolite","similar_to","drug"),
                   ("drug","rev_similar_to","metabolite"))

    z = model(infer_data)
    return {"infer_data": infer_data, "z_dict": z,
            "new_drug_idx": new_idx, "new_drug_emb": z["drug"][new_idx]}

@torch.no_grad()
def rank_metabolites_for_new_drug(model, base_data, new_drug_smiles,
                                   new_drug_name="NEW_DRUG",
                                   target_edges=None, metabolite_edges=None,
                                   top_k=100):
    res = infer_new_drug(model, base_data, new_drug_smiles,
                          target_edges, metabolite_edges)
    z    = res["z_dict"]
    emb  = res["new_drug_emb"]
    probs = torch.sigmoid(torch.matmul(z["metabolite"], emb))
    top   = torch.topk(probs, k=min(top_k, probs.size(0)))
    return pd.DataFrame([
        {"rank": r+1, "metabolite_name": idx2met[i],
         "query_drug": new_drug_name, "new_drug_score": float(s)}
        for r, (i, s) in enumerate(zip(top.indices.tolist(), top.values.tolist()))
    ])

# ── Example: Ibuprofen (NSAID) ─────────────────────────────────────────────
print("\n=== Inductive New Drug Inference (Ibuprofen) ===")
new_drug_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
ibu_df = rank_metabolites_for_new_drug(model, data, new_drug_smiles,
                                        new_drug_name="ibuprofen", top_k=100)
# Combine with path score
ibu_final = combine_gnn_and_path(
    ibu_df.rename(columns={"new_drug_score": "metabolite_gnn_score"}),
    path_df)
ibu_mmr = mmr_rerank(ibu_final, metabolite_drug_edges, top_k=10)
print(ibu_mmr[["mmr_rank","metabolite_name","metabolite_gnn_score",
               "final_score","mmr_score"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 20.  Plots  (save to files)
# ══════════════════════════════════════════════════════════════════════════════
if not args.no_plot:
    out_dir = SCRIPT_DIR

    # Training loss + Val AP
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(range(1, args.epochs+1), history["train_loss"], color="steelblue")
    axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].grid(alpha=0.4)
    if history["epochs_logged"]:
        axes[1].plot(history["epochs_logged"], history["val_ap"],
                     marker="o", color="darkorange")
        axes[1].set_title("Val Mean AP"); axes[1].set_xlabel("Epoch"); axes[1].grid(alpha=0.4)
    plt.tight_layout()
    fp_train = os.path.join(out_dir, "rwpe_training_curves.png")
    plt.savefig(fp_train, dpi=150); plt.close()
    print(f"\nSaved training curves → {fp_train}")

    # RWPE distribution: hub vs. terminal for metabolites that matched
    matched_mask = rwpe_met.sum(axis=1) > 0
    if matched_mask.any():
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, name in enumerate(metabolites_df["metabolite_name"]):
            if matched_mask[i]:
                ax.plot(range(1, WALK_LEN+1), rwpe_met[i], alpha=0.6, label=name)
        ax.set_xlabel("Walk step k"); ax.set_ylabel("Diagonal(T^k)[i]")
        ax.set_title("RWPE per metabolite (return probability at step k)")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(alpha=0.3)
        fp_rwpe = os.path.join(out_dir, "rwpe_metabolite_profiles.png")
        plt.savefig(fp_rwpe, dpi=150); plt.close()
        print(f"Saved RWPE profiles   → {fp_rwpe}")

    # ROC / PR for validation
    @torch.no_grad()
    def collect_probs(model, loaders, edge_types, device):
        model.eval()
        res = {}
        for ldr, et in zip(loaders, edge_types):
            yt, yp = [], []
            src_t, _, dst_t = et
            for batch in ldr:
                batch = batch.to(device)
                if et not in batch.edge_types: continue
                if not hasattr(batch[et], "edge_label_index"): continue
                z = model(batch)
                logits = model.decode(z, batch[et].edge_label_index, src_t, dst_t)
                yp.extend(torch.sigmoid(logits).cpu().numpy())
                yt.extend(batch[et].edge_label.float().cpu().numpy())
            res[et] = {"y_true": np.array(yt), "y_prob": np.array(yp)}
        return res

    val_pred = collect_probs(model, val_loader, valid_val_ets, DEVICE)
    for et, d in val_pred.items():
        yt, yp = d["y_true"], d["y_prob"]
        if len(yt) == 0 or len(np.unique(yt)) < 2: continue
        fpr, tpr, _ = roc_curve(yt, yp)
        prec, rec, _ = precision_recall_curve(yt, yp)
        auc  = roc_auc_score(yt, yp)
        ap   = average_precision_score(yt, yp)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(fpr, tpr, label=f"AUC={auc:.3f}"); axes[0].plot([0,1],[0,1],"--k",alpha=0.4)
        axes[0].set_title(f"ROC [{et[1]}]"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(rec, prec, color="darkorange", label=f"AP={ap:.3f}")
        axes[1].set_title(f"PR [{et[1]}]"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        plt.tight_layout()
        safe = et[1].replace(" ","_")
        fp_roc = os.path.join(out_dir, f"rwpe_roc_pr_{safe}.png")
        plt.savefig(fp_roc, dpi=150); plt.close()
        print(f"Saved ROC/PR          → {fp_roc}")

print("\nAll done.")