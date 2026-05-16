#!/usr/bin/env python3

"""
Inductive HAN + RWPE  ─  Yeast Metabolite → Alopecia Drug Candidate Ranking
=============================================================================
Node types  : metabolite, drug, target
Edge types  : metabolite ──similar_to──> drug
            drug       ──inhibits───> target
            (+ reverse edges for message passing)

Key design decisions
--------------------
1. Gene and disease nodes are *removed* from the GNN entirely.
target_disease_edges are still loaded to get OpenTargets association
weights for the final target-chain ranking.
2. Currency metabolites are auto-detected by degree (top-percentile in the
co-reaction graph) instead of a hard-coded list.
They are:
    (a) excluded from co-reaction graph EDGES when building RWPE
        → prevents spurious super-hub random-walk similarities
    (b) excluded from metabolites_df before any GNN/FP step
        → their 1-d constant features would add no signal anyway
3. Metabolite ranking uses a target-chain score:
    score(met_i) = Σ_t  w_t × sigmoid(met_emb_i · target_emb_t)
where w_t = OpenTargets association_score for target t.
This avoids the degenerate disease-1-node problem where the disease
embedding is a constant vector never seen in the training loss.
"""

# ── std-lib ────────────────────────────────────────────────────────────────
import os, re, time, random, argparse, warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

# ── 3rd-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import scipy.sparse as sp
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors

from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
# 0.  CLI / Config
# ══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Inductive HAN + RWPE for yeast metabolites")
    p.add_argument("--pathway_file",
                default="All-pathways-of-S-cerevisiae-S288c_cleaned_compounds.txt")
    p.add_argument("--drug_target_file",  default="real_drug_target_edges.csv")
    p.add_argument("--target_disease_file", default="real_target_disease_edges.csv")
    p.add_argument("--inchikey_file", default="compound-inchikey.txt",
                help="Mapping file containing Compound name and InChI-Key")
    p.add_argument("--walk_length",  type=int,   default=20)
    p.add_argument("--currency_pct", type=float, default=95.0,
                help="Degree percentile threshold for currency metabolite auto-detection")
    p.add_argument("--top_k_tanimoto", type=int, default=5)
    p.add_argument("--hidden",   type=int, default=256)
    p.add_argument("--heads",    type=int, default=4)
    p.add_argument("--dropout",  type=float, default=0.1)
    p.add_argument("--epochs",   type=int, default=30)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--top_k_rank",  type=int, default=100)
    p.add_argument("--mmr_top_k",   type=int, default=10)
    p.add_argument("--lambda_mmr",  type=float, default=0.7)
    p.add_argument("--mmr_alpha",   type=float, default=0.7,
                help="Weight of drug-profile vs RWPE-pathway similarity in MMR")

    # Evidence-calibrated final ranking knobs.
    # This is intentionally stricter than a simple thresholded Tanimoto boost:
    # a candidate should have chemical similarity AND interpretable target/path evidence,
    # while generic sugars, alcohols, cofactors, and currency-like compounds are penalized.
    p.add_argument("--alopecia_mode", choices=["AGA", "AA", "mixed"], default="AGA",
                help="Target-priority profile: androgenetic alopecia, alopecia areata, or mixed")
    p.add_argument("--candidate_min_tanimoto", type=float, default=0.25,
                help="Minimum max Tanimoto for path-only candidates to enter final ranking")
    p.add_argument("--tanimoto_midpoint", type=float, default=0.45,
                help="Soft midpoint for chemical-similarity scoring; around this value the Tanimoto score rises sharply")
    p.add_argument("--tanimoto_temperature", type=float, default=0.07,
                help="Slope of the soft Tanimoto score; smaller values make the cutoff stricter")
    p.add_argument("--artifact_penalty_strength", type=float, default=0.55,
                help="How strongly generic/toxic/currency-like metabolites are down-weighted")
    p.add_argument("--mmr_min_score_quantile", type=float, default=0.50,
                help="Before MMR, keep only candidates above this final-score quantile to avoid diversity-only low-quality hits")
    p.add_argument("--w_gnn", type=float, default=0.25)
    p.add_argument("--w_path", type=float, default=0.25)
    p.add_argument("--w_chem", type=float, default=0.20)
    p.add_argument("--w_target", type=float, default=0.20)
    p.add_argument("--w_breadth", type=float, default=0.10)
    p.add_argument("--new_drug_w_gnn", type=float, default=0.30)
    p.add_argument("--new_drug_w_path", type=float, default=0.25)
    p.add_argument("--new_drug_w_chem", type=float, default=0.15)
    p.add_argument("--new_drug_w_target", type=float, default=0.20)
    p.add_argument("--new_drug_w_breadth", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--new_drug_smiles", default="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                help="SMILES for inductive new-drug inference example")
    p.add_argument("--new_drug_name", default="ibuprofen")
    return p.parse_args()


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Known synonym map: toy / common names → pathway-file canonical names
COMPOUND_SYNONYMS = {
    "3-hydroxy-3-methylglutaryl-coa":  "(s)-3-hydroxy-3-methylglutaryl-coa",
    "mevalonic acid":                  "(r)-mevalonate",
    "mevalonate 5-phosphate":          "(r)-5-phosphomevalonate",
    "mevalonate 5-diphosphate":        "(r)-mevalonate diphosphate",
    "isopentenyl pyrophosphate":       "isopentenyl diphosphate",
    "dimethylallyl pyrophosphate":     "dimethylallyl diphosphate",
    "geranyl pyrophosphate":           "geranyl diphosphate",
    "farnesyl pyrophosphate":          "(2e,6e)-farnesyl diphosphate",
    "geranylgeranyl pyrophosphate":    "geranylgeranyl diphosphate",
    "2,3-oxidosqualene":               "(3s)-2,3-epoxy-2,3-dihydrosqualene",
    "farnesol":                        "(2e,6e)-farnesol",
}


# ══════════════════════════════════════════════════════════════════════════
# 1.  Currency metabolite auto-detection
# ══════════════════════════════════════════════════════════════════════════
def detect_currency_by_degree(pathway_file: str, percentile: float = 95.0) -> set:
    """
    Parse all reaction equations; count per-compound *reaction* appearances.
    Return the set of compound names (lowercase) whose degree exceeds the
    given percentile threshold.  These super-hubs are treated as currency
    metabolites and excluded from co-reaction graph edges.
    """
    def _parse_side(side):
        out = []
        # 단순 split("+") 대신 정규표현식을 사용하여 양옆에 공백이 있는 '+'만 분해합니다.
        # 이렇게 하면 'H+', 'NAD+', '(+)-allantoin' 내부의 +는 깨지지 않고 유지됩니다.
        for tok in re.split(r'\s+\+\s+', side.strip()):
            tok = re.sub(r"^\d+\s+", "", tok.strip()) # 계수(예: 2 pyruvate -> pyruvate) 제거
            if tok:
                out.append(tok.lower())
        return out

    compound_rxn_count = defaultdict(int)
    with open(pathway_file, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            for rxn in parts[2].split(" // "):
                rxn = rxn.strip()
                m = re.search(r"<-+>|<-|->", rxn)
                if not m:
                    continue
                sep = m.group()
                left_str, right_str = rxn.split(sep, 1)
                seen_in_rxn = set(_parse_side(left_str) + _parse_side(right_str))
                for c in seen_in_rxn:
                    compound_rxn_count[c] += 1

    if not compound_rxn_count:
        return set()

    degrees    = np.array(list(compound_rxn_count.values()), dtype=float)
    threshold  = np.percentile(degrees, percentile)
    currency   = {c for c, d in compound_rxn_count.items() if d > threshold}
    print(f"Currency auto-detection: threshold={threshold:.1f} rxns "
        f"(p{percentile}), {len(currency)} currency compounds identified")
    return currency


# ══════════════════════════════════════════════════════════════════════════
# 2.  Pathway graph  (frequency-weighted, currency-filtered edges)
# ══════════════════════════════════════════════════════════════════════════
def build_pathway_graph(pathway_file: str, currency_set: set):
    """
    Build a weighted co-reaction graph excluding currency metabolite edges.

    - comp2idx  : ALL compounds registered (including currency), so RWPE
                indices stay consistent with the full compound list.
    - edge_freq : only non-currency substrate-product pairs counted.
                weight = number of reactions the pair co-appears in.

    Returns
    -------
    comp2idx : dict[str, int]
    A        : scipy.sparse.csr_matrix  (N x N, float32, frequency weights)
    """
    def _parse_side(side):
        out = []
        # 단순 split("+") 대신 정규표현식을 사용하여 양옆에 공백이 있는 '+'만 분해합니다.
        # 이렇게 하면 'H+', 'NAD+', '(+)-allantoin' 내부의 +는 깨지지 않고 유지됩니다.
        for tok in re.split(r'\s+\+\s+', side.strip()):
            tok = re.sub(r"^\d+\s+", "", tok.strip()) # 계수(예: 2 pyruvate -> pyruvate) 제거
            if tok:
                out.append(tok.lower())
        return out

    comp2idx  = {}
    edge_freq = {}

    with open(pathway_file, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            for rxn in parts[2].split(" // "):
                rxn = rxn.strip()
                m = re.search(r"<-+>|<-|->", rxn)
                if not m:
                    continue
                sep = m.group()
                left_str, right_str = rxn.split(sep, 1)
                lefts  = _parse_side(left_str)
                rights = _parse_side(right_str)

                # Register ALL compounds (currency included) for index consistency
                for c in lefts + rights:
                    if c not in comp2idx:
                        comp2idx[c] = len(comp2idx)

                # Frequency-weighted edges — skip currency pairs
                for l in lefts:
                    if l in currency_set:
                        continue
                    for r in rights:
                        if r in currency_set or l == r:
                            continue
                        i, j = comp2idx[l], comp2idx[r]
                        edge_freq[(i, j)] = edge_freq.get((i, j), 0) + 1
                        edge_freq[(j, i)] = edge_freq.get((j, i), 0) + 1

    N = len(comp2idx)
    if N == 0 or not edge_freq:
        return comp2idx, sp.csr_matrix((N, N))

    rows = [k[0] for k in edge_freq]
    cols = [k[1] for k in edge_freq]
    vals = np.array([edge_freq[k] for k in edge_freq], dtype=np.float32)
    A    = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    print(f"Pathway graph: {N} compounds, {len(edge_freq)//2} undirected edges "
        f"(frequency-weighted, currency filtered)")
    return comp2idx, A


# ══════════════════════════════════════════════════════════════════════════
# 3.  RWPE computation
# ══════════════════════════════════════════════════════════════════════════
def compute_rwpe_matrix(A: sp.spmatrix, K: int) -> np.ndarray:
    """
    RWPE[i, k] = (T^{k+1})[i, i]  where  T = D_w^{-1} A  (row-stochastic).
    Returns (N, K) float32 array.
    """
    N      = A.shape[0]
    degree = np.array(A.sum(axis=1), dtype=np.float64).flatten()
    degree[degree == 0] = 1.0
    T      = (sp.diags(1.0 / degree) @ A).astype(np.float32)

    rwpe = np.zeros((N, K), dtype=np.float32)
    T_k  = T.copy()
    for k in range(K):
        rwpe[:, k] = np.array(T_k.diagonal()).flatten()
        if k < K - 1:
            T_k = T_k @ T
    return rwpe


def get_metabolite_rwpe(metabolite_names: list, comp2idx: dict,
                        rwpe_full: np.ndarray, K: int) -> np.ndarray:
    """Map metabolite display names to RWPE rows.  Unmatched rows stay zero."""
    out   = np.zeros((len(metabolite_names), K), dtype=np.float32)
    found = 0
    for i, name in enumerate(metabolite_names):
        key = name.lower()
        key = COMPOUND_SYNONYMS.get(key, key)
        if key in comp2idx:
            out[i] = rwpe_full[comp2idx[key]]
            found += 1
    print(f"RWPE lookup: {found}/{len(metabolite_names)} matched")
    return out


# ══════════════════════════════════════════════════════════════════════════
# 4.  SMILES helpers
# ══════════════════════════════════════════════════════════════════════════
def load_pathway_compound_smiles(pathway_file: str) -> dict:
    """Return {compound_name_lower: SMILES} from pathway TSV column 4."""
    comp_smiles = {}
    with open(pathway_file, encoding="utf-8") as fh:
        next(fh)
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

def load_compound_data(inchikey_file: str) -> tuple:
    """tsv 파일에서 {이름: InChIKey}와 {이름: SMILES} 매핑을 동시에 로드합니다."""
    comp_inchikey = {}
    comp_file_smiles = {}
    if not os.path.exists(inchikey_file):
        print(f"[WARNING] InChIKey file not found: {inchikey_file}")
        return comp_inchikey, comp_file_smiles
        
    with open(inchikey_file, encoding="utf-8") as fh:
        next(fh)  # 헤더 스킵 (Compound	InChI-Key	SMILES)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            name = parts[0].strip().lower()
            ikey = parts[1].strip()
            
            # 'InChIKey=' 프리픽스 제거 및 유효성 검사
            if ikey.startswith("InChIKey="):
                ikey = ikey.replace("InChIKey=", "")
            
            if name and ikey and ikey != "":
                comp_inchikey[name] = ikey
                
            # 3번째 컬럼에 SMILES 데이터가 실제로 존재하는지 확인
            if len(parts) >= 3:
                smi = parts[2].strip()
                if name and smi and smi != "":
                    comp_file_smiles[name] = smi
                    
    print(f"Loaded {len(comp_inchikey)} InChIKeys and {len(comp_file_smiles)} local SMILES from {inchikey_file}")
    return comp_inchikey, comp_file_smiles


def _fetch_smiles_by_inchikey(inchikey: str):
    """Fetch SMILES string from PubChem REST API via InChIKey."""
    # [수정] 옛날 규격(CanonicalSMILES,IsomericSMILES) 대신 최신 규격인 'SMILES' 하나만 요청합니다.
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/property/SMILES/JSON"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                props = r.json().get("PropertyTable", {}).get("Properties", [])
                if props:
                    # [수정] 가져오는 키값도 대문자 'SMILES'로 변경합니다.
                    smi = props[0].get("SMILES")
                    if smi:
                        return inchikey, smi
            elif r.status_code == 429: # Rate limit 제어
                time.sleep(2 ** attempt)
                continue
            break
        except Exception:
            break
    return inchikey, None

def fetch_metabolite_smiles_concurrently(metabolite_list: list, comp_inchikeys: dict, comp_file_smiles: dict, max_workers: int = 5) -> dict:
    """PubChem 검색을 기본(우선)으로 하되, InChIKey가 없는 경우에만 로컬 파일에서 SMILES를 바로 로드합니다."""
    unique_mets = list(dict.fromkeys(metabolite_list))
    
    met_smiles_map = {}
    inchikeys_to_fetch = []
    inchikey_to_met = {}  # PubChem 결과를 다시 대사물질 이름과 매핑하기 위한 역추적용
    
    for met in unique_mets:
        key = met.lower()
        key = COMPOUND_SYNONYMS.get(key, key)
        
        # [1순위 기본] InChIKey가 존재한다면 -> 무조건 PubChem 웹 검색 큐에 먼저 등록
        if key in comp_inchikeys:
            ikey = comp_inchikeys[key]
            inchikeys_to_fetch.append(ikey)
            inchikey_to_met[ikey] = met
            
        # [2순위 예외] InChIKey가 아예 없는 경우 -> 로컬 파일의 SMILES 데이터를 바로 로드
        elif key in comp_file_smiles:
            met_smiles_map[met] = comp_file_smiles[key]
            
    unique_inchikeys = list(dict.fromkeys(inchikeys_to_fetch))
    inchikey_to_smiles = {}
    
    # 1순위 타깃들 PubChem 비동기 웹 검색 진행
    if unique_inchikeys:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_smiles_by_inchikey, ik): ik for ik in unique_inchikeys}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching metabolite SMILES via PubChem (Default)"):
                ikey, smi = future.result()
                if smi:
                    inchikey_to_smiles[ikey] = smi
                time.sleep(0.1)
                
    # PubChem 웹 검색 성공 결과 데이터프레임 맵에 반영
    for ikey, smi in inchikey_to_smiles.items():
        met = inchikey_to_met.get(ikey)
        if met:
            met_smiles_map[met] = smi
            
    # [3순위 안전장치] InChIKey는 존재해서 PubChem 검색을 시도했으나, 
    # 서버 오류나 일시적 누락(None)으로 구조식을 못 가져온 경우 -> 로컬 파일 데이터가 있다면 최후의 수단으로 보완
    for met in unique_mets:
        if met not in met_smiles_map:
            key = met.lower()
            key = COMPOUND_SYNONYMS.get(key, key)
            if key in comp_file_smiles:
                met_smiles_map[met] = comp_file_smiles[key]
                
    return met_smiles_map
    
def _fetch_single_smiles(compound_name: str):
    """Try ChEMBL first (CHEMBL* IDs), fall back to PubChem REST."""
    if compound_name.upper().startswith("CHEMBL"):
        url = (f"https://www.ebi.ac.uk/chembl/api/data/molecule/"
            f"{compound_name.upper()}.json")
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                structs = r.json().get("molecule_structures") or {}
                smi = structs.get("canonical_smiles") or structs.get("standard_smiles")
                if smi:
                    return compound_name, smi
        except Exception:
            pass
    safe = quote(compound_name)
    url  = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{safe}/property/CanonicalSMILES,IsomericSMILES/JSON")
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                props = r.json().get("PropertyTable", {}).get("Properties", [])
                if props:
                    smi = (props[0].get("CanonicalSMILES") or
                        props[0].get("IsomericSMILES"))
                    if smi:
                        return compound_name, smi
            elif r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            break
        except Exception:
            break
    return compound_name, None


def fetch_smiles_concurrently(compound_list: list, max_workers: int = 5) -> pd.DataFrame:
    unique_names = list(dict.fromkeys(compound_list))
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_single_smiles, n): n
                for n in unique_names}
        for future in tqdm(as_completed(futures), total=len(futures),
                        desc="Fetching drug SMILES"):
            name, smi = future.result()
            results.append({"Name": name, "smiles": smi})
            time.sleep(0.3)
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════
# 5.  Morgan fingerprints
# ══════════════════════════════════════════════════════════════════════════
_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def smiles_to_fp(smiles: str):
    if not smiles or (isinstance(smiles, float) and np.isnan(smiles)):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return _morgan_gen.GetFingerprint(mol)


def fp_to_tensor(fp) -> torch.Tensor:
    arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return torch.tensor(arr, dtype=torch.float)


# ══════════════════════════════════════════════════════════════════════════
# 6.  Index / edge helpers
# ══════════════════════════════════════════════════════════════════════════
def make_mapping(values):
    vals = sorted(pd.Series(values).dropna().astype(str).unique().tolist())
    id2idx = {v: i for i, v in enumerate(vals)}
    idx2id = {i: v for v, i in id2idx.items()}
    return id2idx, idx2id


def build_edge_index(df, src_col, dst_col, src_map, dst_map) -> torch.Tensor:
    src_list, dst_list = [], []
    for _, row in df.iterrows():
        s, d = str(row[src_col]), str(row[dst_col])
        if s in src_map and d in dst_map:
            src_list.append(src_map[s])
            dst_list.append(dst_map[d])
    if not src_list:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src_list, dst_list], dtype=torch.long)


# ══════════════════════════════════════════════════════════════════════════
# 7.  Model
# ══════════════════════════════════════════════════════════════════════════
class InductiveHANLinkPredictor(nn.Module):
    """
    Inductive HAN for heterogeneous graph.
    Uses nn.Linear projections (not nn.Embedding) -- works on unseen nodes.
    """
    def __init__(self, metadata, in_channels_dict,
                hidden_channels=256, out_channels=256,
                heads=4, dropout=0.1):
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
        # Initial projection
        x_proj = {nt: self.node_proj[nt](data.x_dict[nt])
                for nt in data.x_dict}

        # First HANConv + residual
        out1   = self.conv1(x_proj, data.edge_index_dict)
        x_dict = {}
        for k in x_proj:
            c = out1.get(k)
            x_dict[k] = F.elu(c + x_proj[k]) if c is not None else F.elu(x_proj[k])

        # Second HANConv + residual (no activation at output)
        out2       = self.conv2(x_dict, data.edge_index_dict)
        final_dict = {}
        for k in x_dict:
            c = out2.get(k)
            final_dict[k] = (c + x_dict[k]) if c is not None else x_dict[k]

        return final_dict

    def decode(self, z_dict, edge_label_index, src_type, dst_type):
        src_z = z_dict[src_type][edge_label_index[0]]
        dst_z = z_dict[dst_type][edge_label_index[1]]
        return (src_z * dst_z).sum(dim=-1)


# ══════════════════════════════════════════════════════════════════════════
# 8.  Training helpers
# ══════════════════════════════════════════════════════════════════════════
def compute_loss(model, batch, task_edge_types, device):
    z_dict = model(batch)
    losses = []
    for etype in task_edge_types:
        if etype not in batch.edge_types:
            continue
        if not hasattr(batch[etype], "edge_label_index"):
            continue
        src_type, _, dst_type = etype
        logits = model.decode(z_dict,
                            batch[etype].edge_label_index,
                            src_type, dst_type)
        loss   = F.binary_cross_entropy_with_logits(
                    logits, batch[etype].edge_label.float())
        losses.append(loss)
    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate_auc_ap(model, loaders, edge_types, device):
    model.eval()
    y_true_d, y_prob_d = defaultdict(list), defaultdict(list)
    for loader, etype in zip(loaders, edge_types):
        for batch in loader:
            batch = batch.to(device)
            if etype not in batch.edge_types:
                continue
            if not hasattr(batch[etype], "edge_label_index"):
                continue
            src_type, _, dst_type = etype
            z    = model(batch)
            logits = model.decode(z, batch[etype].edge_label_index,
                                src_type, dst_type)
            y_true_d[etype].extend(batch[etype].edge_label.float().cpu().numpy())
            y_prob_d[etype].extend(torch.sigmoid(logits).cpu().numpy())

    results = {}
    for etype in edge_types:
        yt = np.array(y_true_d.get(etype, []))
        yp = np.array(y_prob_d.get(etype, []))
        if len(yt) == 0 or len(np.unique(yt)) < 2:
            continue
        auc = roc_auc_score(yt, yp)
        ap  = average_precision_score(yt, yp)
        f1  = f1_score((yp > 0.5).astype(int), (yt > 0.5).astype(int),
                    zero_division=0)
        results[etype] = {"AUC": auc, "AP": ap, "F1": f1}
    return results


# ══════════════════════════════════════════════════════════════════════════
# 9.  Ranking  --  target-chain score
#     score(met_i) = sum_t  w_t * sigmoid(met_emb_i . target_emb_t)
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def rank_metabolites_via_target_chain(model, full_data,
                                    target_disease_edges: pd.DataFrame,
                                    target2idx: dict, idx2met: dict,
                                    device, top_k: int = 100) -> pd.DataFrame:
    """
    Rank metabolites by their proximity to alopecia-associated targets.

    Uses target-chain scoring instead of disease inner-product, because
    with a single disease node the disease embedding is a constant vector
    that never appears in the training loss (degenerate).

    score(met_i) = sum_t  w_t * sigmoid(met_emb_i . target_emb_t)
    w_t = OpenTargets association_score (max over multiple rows per target)
    """
    model.eval()
    full_data = full_data.to(device)
    z         = model(full_data)
    met_emb = F.normalize(z["metabolite"], p=2, dim=1)
    tgt_emb = F.normalize(z["target"], p=2, dim=1)

    # Aggregate target weights (max association score per target)
    tgt_scores = (target_disease_edges
                .groupby("target_id")["association_score"]
                .max()
                .reset_index())

    score = torch.zeros(met_emb.size(0), device=device)
    total_weigth = 0.0

    for _, row in tgt_scores.iterrows():
        t_name = str(row["target_id"])
        w      = float(row["association_score"])
        if t_name not in target2idx:
            continue
        t_vec  = tgt_emb[target2idx[t_name]]       # (D,)
        #sim    = torch.sigmoid(met_emb @ t_vec)     # (N_met,)
        sim = torch.sigmoid(5*F.cosine_similarity(met_emb, t_vec.unsqueeze(0), dim=1))
        #score  += w * sim
        #sim = (sim+1)/2
        score += w * sim
        total_weigth += w

    score = score / total_weigth

    top = torch.topk(score, k=min(top_k, score.size(0)))
    rows = []
    for rank, (idx, s) in enumerate(
            zip(top.indices.cpu().tolist(), top.values.cpu().tolist()), 1):
        rows.append({
            "rank": rank,
            "metabolite_name": idx2met[idx],
            "metabolite_gnn_score": round(float(s), 6),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# 10.  Path explanation
# ══════════════════════════════════════════════════════════════════════════
def build_path_explanation(metabolites_df, metabolite_drug_edges,
                            drug_target_edges, target_disease_edges) -> pd.DataFrame:
    rows = []
    for met in metabolites_df["metabolite_name"].unique():
        sim_hits = metabolite_drug_edges[metabolite_drug_edges["metabolite_name"] == met]
        for _, sim in sim_hits.iterrows():
            drug    = sim["drug_name"]
            tanimoto = sim["tanimoto"]
            dt_hits = drug_target_edges[drug_target_edges["drug_name"] == drug]
            for _, dt in dt_hits.iterrows():
                target  = dt["target_id"]
                dt_conf = dt["confidence"]
                td_hits = target_disease_edges[target_disease_edges["target_id"] == target]
                for _, td in td_hits.iterrows():
                    disease       = td.get("disease_id", "ALOPECIA")
                    disease_score = td.get("association_score", 1.0)
                    path_score    = tanimoto * float(dt_conf) * float(disease_score)
                    rows.append({
                        "metabolite_name":        met,
                        "similar_drug":           drug,
                        "target":                 target,
                        "disease":                disease,
                        "tanimoto":               tanimoto,
                        "drug_target_confidence": dt_conf,
                        "target_disease_score":   disease_score,
                        "path_score":             path_score,
                    })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# 11.  Final ranking  (GNN score + path score)
# ══════════════════════════════════════════════════════════════════════════

def _robust_minmax(series, lower_q: float = 0.05, upper_q: float = 0.95,
                neutral: float = 0.5) -> pd.Series:
    """Quantile-clipped min-max scaling. More stable than raw min-max."""
    x = pd.Series(series).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(x) == 0:
        return x
    lo = x.quantile(lower_q)
    hi = x.quantile(upper_q)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = x.min(), x.max()
    if hi > lo:
        return ((x.clip(lo, hi) - lo) / (hi - lo)).clip(0.0, 1.0)
    return pd.Series(np.full(len(x), neutral), index=x.index, dtype=float)


def _sigmoid_score(x, midpoint: float = 0.45, temperature: float = 0.07) -> pd.Series:
    """Soft threshold: values near midpoint rise sharply without a hard jump."""
    x = pd.Series(x).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    temp = max(float(temperature), 1e-6)
    return 1.0 / (1.0 + np.exp(-(x - float(midpoint)) / temp))


def _canonical_target_symbol(target) -> str:
    """Normalize target symbols such as 'srd5a2', 'SRD5A2_HUMAN', etc."""
    t = str(target).upper().strip()
    t = re.sub(r"_HUMAN$", "", t)
    t = re.sub(r"[^A-Z0-9]", "", t)
    return t


def get_alopecia_target_priority(mode: str = "AGA") -> dict:
    """
    Conservative target-priority map for final ranking.

    AGA focuses on androgen metabolism/receptor signaling, Wnt-beta-catenin,
    prostaglandin signaling, dermal papilla growth-factor biology, and
    perifollicular inflammation/oxidative-stress related nodes.

    AA focuses on JAK-STAT/interferon/IL-15 immune-axis targets.
    """
    mode = str(mode).upper()

    aga_high = {
        "AR", "SRD5A1", "SRD5A2", "PTGDR2", "GPR44", "PTGDS",
        "CTNNB1", "WNT10A", "WNT10B", "GSK3B", "DKK1", "LEF1"
    }
    aga_mid = {
        "VEGFA", "KDR", "FLT1", "HIF1A", "IGF1", "IGF1R",
        "FGF5", "FGF7", "FGFR1", "FGFR2", "TGFB1", "TGFB2", "TGFB3",
        "TGFBR1", "TGFBR2", "BMP2", "BMP4", "BMPR1A", "SHH", "SMO", "GLI1",
        "PTGS1", "PTGS2", "PTGES", "CYP19A1", "ESR1", "ESR2"
    }
    aga_low = {
        "IL6", "TNF", "NFKB1", "RELA", "PPARG", "AKT1", "MTOR",
        "MAPK1", "MAPK3", "EGFR", "CASP3", "BCL2", "SOD1", "CAT"
    }

    aa_high = {
        "JAK1", "JAK2", "JAK3", "TYK2", "STAT1", "STAT3", "STAT5A", "STAT5B",
        "IFNG", "IL15", "IL15RA", "IL2RG", "CXCL9", "CXCL10", "CXCL11"
    }
    aa_mid = {"IL2", "IL7", "IL21", "IL12A", "IL12B", "TNF", "NFKB1", "RELA", "CTLA4"}

    priority = {}
    if mode in {"AGA", "MIXED"}:
        priority.update({t: 1.00 for t in aga_high})
        priority.update({t: max(priority.get(t, 0), 0.75) for t in aga_mid})
        priority.update({t: max(priority.get(t, 0), 0.45) for t in aga_low})
    if mode in {"AA", "MIXED"}:
        priority.update({t: max(priority.get(t, 0), 1.00) for t in aa_high})
        priority.update({t: max(priority.get(t, 0), 0.70) for t in aa_mid})
    return priority


def _artifact_penalty_from_name(name: str) -> float:
    """Penalize generic metabolites that often rank high because of graph/FP artefacts."""
    n = str(name).lower()
    penalty = 0.0

    severe_patterns = [
        r"\b(isobutanol|butanol|ethanol|methanol|propanol)\b",
        r"\bacetaldehyde\b", r"phenylacetaldehyde", r"\baldehyde\b",
    ]
    sugar_patterns = [
        r"glucopyranose", r"galactopyranose", r"mannopyranose", r"fructofuranose",
        r"\bglucose\b", r"\bgalactose\b", r"\bmannose\b", r"\bfructose\b",
        r"\bxylose\b", r"\barabinose\b", r"\bribose\b", r"trehalose", r"sucrose"
    ]
    cofactor_currency_patterns = [
        r"\batp\b", r"\badp\b", r"\bamp\b", r"\bgtp\b", r"\bgdp\b",
        r"\bnad\b", r"\bnadh\b", r"\bnadp\b", r"\bnadph\b",
        r"coenzyme a", r"\bcoa\b", r"thiamine diphosphate", r"tetrahydrofolate",
        r"tetrahydropteroyl", r"phosphate", r"diphosphate", r"pyrophosphate"
    ]

    if any(re.search(p, n) for p in severe_patterns):
        penalty = max(penalty, 0.90)
    if any(re.search(p, n) for p in sugar_patterns):
        penalty = max(penalty, 0.75)
    if any(re.search(p, n) for p in cofactor_currency_patterns):
        penalty = max(penalty, 0.65)

    # Do not punish sterol/quinone classes that are biologically plausible in yeast lipid biology.
    if any(k in n for k in ["ergosterol", "zymosterol", "lanosterol", "squalene", "ubiquinone", "ubiquinol"]):
        penalty = min(penalty, 0.15)
    return float(penalty)


def _artifact_penalty_from_smiles(smiles) -> float:
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return 0.0
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return 0.0
    mw = Descriptors.MolWt(mol)
    heavy = mol.GetNumHeavyAtoms()
    hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
    hetero_ratio = hetero / max(1, heavy)

    penalty = 0.0
    if mw < 85 or heavy < 6:
        penalty = max(penalty, 0.55)
    # Very hetero-atom rich small molecules are often sugars/phosphorylated currency metabolites.
    if heavy <= 20 and hetero_ratio >= 0.55:
        penalty = max(penalty, 0.45)
    return float(penalty)


def _candidate_class(row) -> str:
    if row["artifact_penalty"] >= 0.75 and row["max_tanimoto"] < 0.55:
        return "likely_artifact"
    if (row["target_relevance_score"] >= 0.75 and
        row["max_tanimoto"] >= 0.45 and
        row["artifact_penalty"] < 0.45 and
        row["best_weighted_path_score"] > 0):
        return "priority_candidate"
    if (row["target_relevance_score"] >= 0.45 and
        row["max_tanimoto"] >= 0.35 and
        row["artifact_penalty"] < 0.65):
        return "moderate_candidate"
    return "exploratory_only"


def build_final_ranking(metabolite_rank_df: pd.DataFrame,
                        path_df: pd.DataFrame,
                        metabolites_df: pd.DataFrame = None,
                        disease_id: str = "ALOPECIA",
                        alopecia_mode: str = "AGA",
                        candidate_min_tanimoto: float = 0.25,
                        tanimoto_midpoint: float = 0.45,
                        tanimoto_temperature: float = 0.07,
                        w_gnn: float = 0.25,
                        w_path: float = 0.25,
                        w_chem: float = 0.20,
                        w_target: float = 0.20,
                        w_breadth: float = 0.10,
                        artifact_penalty_strength: float = 0.55,
                        include_path_candidates: bool = True) -> pd.DataFrame:
    """
    Evidence-calibrated final ranking.

    This replaces a simple Tanimoto boost with five interpretable components:
    1) GNN target-chain score
    2) target-weighted path score
    3) soft chemical-similarity score from max/mean-top Tanimoto
    4) alopecia-target relevance score
    5) evidence breadth across drugs and targets

    Then it penalizes generic sugars, volatile alcohols/aldehydes, cofactors,
    and currency-like metabolites. This is deliberately conservative: it tries
    to rank plausible follow-up candidates, not merely high-Tanimoto artefacts.
    """
    if path_df is None:
        path_df = pd.DataFrame()
    if "metabolite_name" not in path_df.columns and "metabolite" in path_df.columns:
        path_df = path_df.rename(columns={"metabolite": "metabolite_name"})

    target_priority_map = get_alopecia_target_priority(alopecia_mode)
    required_cols = {"metabolite_name", "disease", "path_score", "tanimoto", "similar_drug", "target"}

    if len(path_df) == 0 or not required_cols.issubset(path_df.columns):
        met_path = pd.DataFrame(columns=[
            "metabolite_name", "best_path_score", "best_weighted_path_score",
            "sum_top3_weighted_path_score", "num_paths", "num_unique_drugs", "num_unique_targets",
            "max_tanimoto", "mean_top3_tanimoto", "target_relevance_score",
            "high_quality_paths", "best_similar_drug", "best_target"
        ])
    else:
        alopecia_df = path_df[path_df["disease"] == disease_id].copy()
        if len(alopecia_df) == 0:
            met_path = pd.DataFrame(columns=[
                "metabolite_name", "best_path_score", "best_weighted_path_score",
                "sum_top3_weighted_path_score", "num_paths", "num_unique_drugs", "num_unique_targets",
                "max_tanimoto", "mean_top3_tanimoto", "target_relevance_score",
                "high_quality_paths", "best_similar_drug", "best_target"
            ])
        else:
            alopecia_df["target_symbol_norm"] = alopecia_df["target"].apply(_canonical_target_symbol)
            alopecia_df["target_priority"] = alopecia_df["target_symbol_norm"].map(target_priority_map).fillna(0.0).astype(float)
            alopecia_df["tanimoto"] = alopecia_df["tanimoto"].astype(float)
            alopecia_df["path_score"] = alopecia_df["path_score"].astype(float)
            alopecia_df["weighted_path_score"] = alopecia_df["path_score"] * alopecia_df["target_priority"]
            alopecia_df["is_high_quality_path"] = (
                (alopecia_df["target_priority"] >= 0.45) &
                (alopecia_df["tanimoto"] >= 0.35)
            ).astype(int)

            def _summarize_group(g: pd.DataFrame) -> pd.Series:
                g = g.copy()
                top_tani = g["tanimoto"].sort_values(ascending=False).head(3)
                top_weighted = g["weighted_path_score"].sort_values(ascending=False).head(3)
                weights = g["path_score"].clip(lower=0.0).to_numpy(dtype=float)
                priorities = g["target_priority"].to_numpy(dtype=float)
                if weights.sum() > 0:
                    target_rel = float(np.average(priorities, weights=weights + 1e-12))
                else:
                    target_rel = float(np.max(priorities)) if len(priorities) else 0.0
                best_idx = g["weighted_path_score"].idxmax() if g["weighted_path_score"].max() > 0 else g["path_score"].idxmax()
                best_row = g.loc[best_idx]
                return pd.Series({
                    "best_path_score": float(g["path_score"].max()),
                    "best_weighted_path_score": float(g["weighted_path_score"].max()),
                    "sum_top3_weighted_path_score": float(top_weighted.sum()),
                    "num_paths": int(len(g)),
                    "num_unique_drugs": int(g["similar_drug"].nunique()),
                    "num_unique_targets": int(g["target"].nunique()),
                    "max_tanimoto": float(g["tanimoto"].max()),
                    "mean_top3_tanimoto": float(top_tani.mean()) if len(top_tani) else 0.0,
                    "target_relevance_score": float(np.clip(target_rel, 0.0, 1.0)),
                    "high_quality_paths": int(g["is_high_quality_path"].sum()),
                    "best_similar_drug": best_row.get("similar_drug", ""),
                    "best_target": best_row.get("target", ""),
                })

            met_path = alopecia_df.groupby("metabolite_name", group_keys=False).apply(_summarize_group).reset_index()

    # Candidate universe: initial GNN candidates + path-supported candidates.
    base_candidates = metabolite_rank_df.copy()
    if include_path_candidates and len(met_path) > 0:
        path_candidates = met_path.loc[
            (met_path["max_tanimoto"].astype(float) >= float(candidate_min_tanimoto)) |
            (met_path["best_weighted_path_score"].astype(float) > 0),
            ["metabolite_name"]
        ].copy()
        if len(path_candidates) > 0:
            base_candidates = pd.concat([base_candidates, path_candidates], ignore_index=True, sort=False)

    base_candidates = base_candidates.drop_duplicates(subset=["metabolite_name"], keep="first").reset_index(drop=True)

    final_df = base_candidates.merge(met_path, on="metabolite_name", how="left")

    numeric_cols = [
        "best_path_score", "best_weighted_path_score", "sum_top3_weighted_path_score",
        "num_paths", "num_unique_drugs", "num_unique_targets", "max_tanimoto",
        "mean_top3_tanimoto", "target_relevance_score", "high_quality_paths"
    ]
    for col in numeric_cols:
        if col not in final_df.columns:
            final_df[col] = 0.0
        final_df[col] = final_df[col].fillna(0.0).astype(float)
    for col in ["best_similar_drug", "best_target"]:
        if col not in final_df.columns:
            final_df[col] = ""
        final_df[col] = final_df[col].fillna("")

    if "metabolite_gnn_score" not in final_df.columns:
        final_df["metabolite_gnn_score"] = np.nan
    min_gnn = final_df["metabolite_gnn_score"].min(skipna=True)
    if pd.isna(min_gnn):
        min_gnn = 0.0
    final_df["added_by_path_evidence"] = final_df["metabolite_gnn_score"].isna()
    final_df["metabolite_gnn_score"] = final_df["metabolite_gnn_score"].fillna(min_gnn).astype(float)

    # Optional chemical-property annotations for artefact control.
    if metabolites_df is not None and "metabolite_name" in metabolites_df.columns:
        cols = ["metabolite_name"] + (["smiles"] if "smiles" in metabolites_df.columns else [])
        final_df = final_df.merge(metabolites_df[cols].drop_duplicates("metabolite_name"),
                                on="metabolite_name", how="left")
    if "smiles" not in final_df.columns:
        final_df["smiles"] = np.nan

    name_pen = final_df["metabolite_name"].apply(_artifact_penalty_from_name).astype(float)
    smiles_pen = final_df["smiles"].apply(_artifact_penalty_from_smiles).astype(float)
    final_df["artifact_penalty"] = np.maximum(name_pen, smiles_pen).clip(0.0, 1.0)

    # Component scores.
    final_df["gnn_score_scaled"] = _robust_minmax(final_df["metabolite_gnn_score"], neutral=0.5)
    final_df["path_score_scaled"] = _robust_minmax(np.log1p(final_df["sum_top3_weighted_path_score"]), neutral=0.0)

    max_tani_soft = _sigmoid_score(final_df["max_tanimoto"], midpoint=tanimoto_midpoint,
                                temperature=tanimoto_temperature)
    mean_tani_soft = _sigmoid_score(final_df["mean_top3_tanimoto"],
                                    midpoint=max(0.30, tanimoto_midpoint - 0.10),
                                    temperature=tanimoto_temperature)
    final_df["chem_similarity_score"] = (0.70 * max_tani_soft + 0.30 * mean_tani_soft).clip(0.0, 1.0)
    final_df.loc[final_df["max_tanimoto"] < 0.20, "chem_similarity_score"] = 0.0

    breadth_drug = np.log1p(final_df["num_unique_drugs"]) / np.log1p(4)
    breadth_tgt = np.log1p(final_df["num_unique_targets"]) / np.log1p(4)
    final_df["evidence_breadth_score"] = (0.50 * breadth_drug + 0.50 * breadth_tgt).clip(0.0, 1.0)

    weight_sum = max(1e-12, float(w_gnn) + float(w_path) + float(w_chem) + float(w_target) + float(w_breadth))
    final_df["raw_evidence_score"] = (
        float(w_gnn)    * final_df["gnn_score_scaled"] +
        float(w_path)   * final_df["path_score_scaled"] +
        float(w_chem)   * final_df["chem_similarity_score"] +
        float(w_target) * final_df["target_relevance_score"] +
        float(w_breadth)* final_df["evidence_breadth_score"]
    ) / weight_sum

    final_df["final_score"] = (
        final_df["raw_evidence_score"] *
        (1.0 - float(artifact_penalty_strength) * final_df["artifact_penalty"])
    ).clip(0.0, 1.0)

    final_df["candidate_class"] = final_df.apply(_candidate_class, axis=1)

    final_df = (final_df
                .sort_values([
                    "final_score", "candidate_class", "target_relevance_score",
                    "max_tanimoto", "path_score_scaled", "gnn_score_scaled"
                ], ascending=[False, True, False, False, False, False])
                .reset_index(drop=True))
    final_df["final_rank"] = np.arange(1, len(final_df) + 1)
    return final_df

# ══════════════════════════════════════════════════════════════════════════
# 12.  MMR re-ranking
# ══════════════════════════════════════════════════════════════════════════
def build_similarity_matrix(candidates: pd.DataFrame,
                            metabolite_drug_edges: pd.DataFrame,
                            metabolites_df: pd.DataFrame,
                            rwpe_met: np.ndarray,
                            alpha: float = 0.7) -> np.ndarray:
    candidate_mets = candidates["metabolite_name"].tolist()

    pivot = (metabolite_drug_edges
            .pivot_table(index="metabolite_name", columns="drug_name",
                        values="tanimoto", aggfunc="max", fill_value=0.0)
            .reindex(candidate_mets).fillna(0.0))
    profile = pivot.to_numpy(dtype=float)
    norms   = np.linalg.norm(profile, axis=1, keepdims=True)
    profile_n = np.divide(profile, norms,
                        out=np.zeros_like(profile), where=norms != 0)
    drug_sim  = profile_n @ profile_n.T

    if rwpe_met is not None and rwpe_met.shape[1] > 0:
        met_indices = [
            metabolites_df.index[
                metabolites_df["metabolite_name"] == m].tolist()
            for m in candidate_mets]
        rwpe_vecs = np.zeros((len(candidate_mets), rwpe_met.shape[1]))
        for i, idx_list in enumerate(met_indices):
            if idx_list:
                rwpe_vecs[i] = rwpe_met[idx_list[0]]
        rnorms = np.linalg.norm(rwpe_vecs, axis=1, keepdims=True)
        rwpe_n = np.divide(rwpe_vecs, rnorms,
                            out=np.zeros_like(rwpe_vecs), where=rnorms != 0)
        path_sim = rwpe_n @ rwpe_n.T
        sim = alpha * drug_sim + (1.0 - alpha) * path_sim
    else:
        sim = drug_sim
    return sim


def mmr_rerank(rank_df: pd.DataFrame,
            metabolite_drug_edges: pd.DataFrame,
            metabolites_df: pd.DataFrame,
            rwpe_met: np.ndarray,
            score_col: str = "final_score",
            top_k: int = 10,
            candidate_k: int = 100,
            lambda_mmr: float = 0.7,
            alpha: float = 0.7,
            min_score_quantile: float = 0.50) -> pd.DataFrame:
    candidates = (rank_df
                .sort_values(score_col, ascending=False)
                .head(min(candidate_k, len(rank_df)))
                .copy().reset_index(drop=True))

    # MMR can otherwise select weak candidates just because they are dissimilar.
    # Keep it as a diversity step among already-good candidates, not a rescue step
    # for low-evidence metabolites.
    if len(candidates) > top_k and 0.0 < float(min_score_quantile) < 1.0:
        cutoff = candidates[score_col].astype(float).quantile(float(min_score_quantile))
        filtered = candidates[candidates[score_col].astype(float) >= cutoff].copy()
        if len(filtered) >= top_k:
            candidates = filtered.reset_index(drop=True)
    if len(candidates) == 0:
        return candidates

    sim_matrix = build_similarity_matrix(
        candidates, metabolite_drug_edges, metabolites_df, rwpe_met, alpha)

    relevance = candidates[score_col].astype(float).to_numpy()
    if relevance.max() > relevance.min():
        relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min())
    else:
        relevance = np.ones_like(relevance)

    selected, remaining, mmr_scores = [], list(range(len(candidates))), []

    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
            best_score = relevance[best]
        else:
            def _mmr(i, _sel=selected):
                return (lambda_mmr * relevance[i]
                        - (1.0 - lambda_mmr) * max(sim_matrix[i, j] for j in _sel))
            best = max(remaining, key=_mmr)
            best_score = _mmr(best)
        selected.append(best)
        mmr_scores.append(best_score)
        remaining.remove(best)

    out = candidates.iloc[selected].copy().reset_index(drop=True)
    out["mmr_rank"]  = np.arange(1, len(out) + 1)
    out["mmr_score"] = mmr_scores
    return out


# ══════════════════════════════════════════════════════════════════════════
# 13.  Inductive new-drug inference
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def infer_new_drug(model, base_data, new_drug_smiles: str,
                met2idx: dict, target2idx: dict,
                device,
                target_edges=None, metabolite_edges=None):
    if target_edges    is None: target_edges    = []
    if metabolite_edges is None: metabolite_edges = []

    fp = smiles_to_fp(new_drug_smiles)
    if fp is None:
        raise ValueError(f"Invalid SMILES: {new_drug_smiles}")

    new_x    = fp_to_tensor(fp).unsqueeze(0).to(device)
    expected = base_data["drug"].x.size(1)
    if new_x.size(1) != expected:
        raise ValueError(f"FP dim mismatch: {new_x.size(1)} vs {expected}")

    infer_data = base_data.clone().to(device)
    new_idx    = infer_data["drug"].x.size(0)
    infer_data["drug"].x         = torch.cat([infer_data["drug"].x, new_x], dim=0)
    infer_data["drug"].num_nodes = infer_data["drug"].x.size(0)

    # Optional drug->target edges
    new_dt = [(new_idx, target2idx[t]) for t in target_edges if t in target2idx]
    if new_dt:
        ne = torch.tensor(new_dt, dtype=torch.long, device=device).t()
        infer_data["drug", "inhibits", "target"].edge_index = torch.cat(
            [infer_data["drug", "inhibits", "target"].edge_index, ne], dim=1)
        infer_data["target", "rev_inhibits", "drug"].edge_index = torch.cat(
            [infer_data["target", "rev_inhibits", "drug"].edge_index, ne.flip(0)], dim=1)

    # Optional metabolite->drug edges
    new_md = [(met2idx[m], new_idx) for m in metabolite_edges if m in met2idx]
    if new_md:
        ne = torch.tensor(new_md, dtype=torch.long, device=device).t()
        infer_data["metabolite", "similar_to", "drug"].edge_index = torch.cat(
            [infer_data["metabolite", "similar_to", "drug"].edge_index, ne], dim=1)
        infer_data["drug", "rev_similar_to", "metabolite"].edge_index = torch.cat(
            [infer_data["drug", "rev_similar_to", "metabolite"].edge_index, ne.flip(0)], dim=1)

    z_dict       = model(infer_data)
    new_drug_emb = z_dict["drug"][new_idx]
    return {"infer_data": infer_data, "z_dict": z_dict,
            "new_drug_idx": new_idx, "new_drug_emb": new_drug_emb}


@torch.no_grad()
def rank_metabolites_for_new_drug(model, base_data, new_drug_smiles: str,
                                    new_drug_name: str,
                                    met2idx: dict, target2idx: dict,
                                    idx2met: dict, device,
                                    target_edges=None, metabolite_edges=None,
                                    top_k: int = 100) -> pd.DataFrame:
    result       = infer_new_drug(model, base_data, new_drug_smiles,
                                met2idx, target2idx, device,
                                target_edges, metabolite_edges)
    z_dict       = result["z_dict"]
    new_drug_emb = result["new_drug_emb"]
    met_emb      = z_dict["metabolite"]

    met_emb_n  = F.normalize(met_emb, p=2, dim=-1)
    drug_emb_n = F.normalize(new_drug_emb, p=2, dim=-1)
    scores     = met_emb_n @ drug_emb_n

    top  = torch.topk(scores, k=min(top_k, scores.size(0)))
    rows = []
    for rank, (idx, s) in enumerate(
            zip(top.indices.cpu().tolist(), top.values.cpu().tolist()), 1):
        rows.append({"rank": rank,
                    "metabolite_name": idx2met[idx],
                    "query_drug": new_drug_name,
                    "new_drug_score": round(float(s), 4)})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# 14.  Path explanation printing
# ══════════════════════════════════════════════════════════════════════════
def show_path_explanation(rank_df: pd.DataFrame,
                        path_df: pd.DataFrame,
                        top_n: int = 5,
                        disease_id: str = "ALOPECIA"):
    if "metabolite_name" not in path_df.columns and "metabolite" in path_df.columns:
        path_df = path_df.rename(columns={"metabolite": "metabolite_name"})
    dis_df = path_df[path_df["disease"] == disease_id]

    for met in rank_df["metabolite_name"].head(top_n):
        print("=" * 90)
        print(f"Metabolite: {met}")
        hits = dis_df[dis_df["metabolite_name"] == met].sort_values(
            "path_score", ascending=False)
        if len(hits) == 0:
            print("  No interpretable path found.")
        else:
            print(hits[["metabolite_name", "similar_drug", "target",
                        "disease", "tanimoto",
                        "drug_target_confidence", "target_disease_score",
                        "path_score"]].head(5).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════
# 15.  ROC / PR plots
# ══════════════════════════════════════════════════════════════════════════
def plot_roc_pr(pred_results: dict, title_prefix: str = "Validation",
                out_dir: str = "."):
    from sklearn.metrics import roc_curve, precision_recall_curve
    for etype, res in pred_results.items():
        yt, yp = res["y_true"], res["y_prob"]
        if len(yt) == 0 or len(np.unique(yt)) < 2:
            continue
        roc_auc = roc_auc_score(yt, yp)
        ap      = average_precision_score(yt, yp)
        fpr, tpr, _  = roc_curve(yt, yp)
        prec, rec, _ = precision_recall_curve(yt, yp)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        axes[0].plot([0,1],[0,1],"--")
        axes[0].set(title=f"{title_prefix} ROC - {etype[1]}",
                    xlabel="FPR", ylabel="TPR")
        axes[0].legend()
        axes[1].plot(rec, prec, label=f"AP={ap:.3f}")
        axes[1].set(title=f"{title_prefix} PR - {etype[1]}",
                    xlabel="Recall", ylabel="Precision")
        axes[1].legend()
        plt.tight_layout()
        fname = os.path.join(out_dir,
                            f"roc_pr_{etype[1]}_{title_prefix.lower()}.png")
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  {etype[1]} | AUC={roc_auc:.4f} | AP={ap:.4f}")


# ══════════════════════════════════════════════════════════════════════════
# 16.  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    WALK_LEN = args.walk_length
    FP_DIM   = 2048

    pathway_file        = os.path.join(SCRIPT_DIR, args.pathway_file)
    drug_target_file    = os.path.join(SCRIPT_DIR, args.drug_target_file)
    target_disease_file = os.path.join(SCRIPT_DIR, args.target_disease_file)
    inchikey_file       = os.path.join(SCRIPT_DIR, args.inchikey_file)  # 추가

    for f in [pathway_file, drug_target_file, target_disease_file, inchikey_file]:  # 검증 대상에 추가
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing required file: {f}")

    # ── 16.1  Load edge CSVs ──────────────────────────────────────────────
    print("\n[1/9] Loading real edge data ...")
    _dt_raw = pd.read_csv(drug_target_file)
    drug_target_edges = (_dt_raw[["drug_id", "target_symbol", "confidence"]]
                        .rename(columns={"drug_id": "drug_name",
                                        "target_symbol": "target_id"})
                        .copy())
    print(f"  drug-target: {len(drug_target_edges)} rows, "
        f"{drug_target_edges['drug_name'].nunique()} drugs, "
        f"{drug_target_edges['target_id'].nunique()} targets")

    _td_raw = pd.read_csv(target_disease_file)
    target_disease_edges = (_td_raw[["target_symbol", "disease_id",
                                    "disease_name", "association_score"]]
                            .rename(columns={"target_symbol": "target_id"})
                            .copy())
    target_disease_edges["disease_id"]   = "ALOPECIA"
    target_disease_edges["disease_name"] = "Alopecia"
    print(f"  target-disease: {len(target_disease_edges)} rows, "
        f"{target_disease_edges['target_id'].nunique()} targets")

    # ── 16.2  Currency auto-detection ─────────────────────────────────────
    print(f"\n[2/9] Currency metabolite detection (p{args.currency_pct}) ...")
    currency_set = detect_currency_by_degree(pathway_file, args.currency_pct)

    # ── 16.3  Pathway compound list ───────────────────────────────────────
    print("\n[3/9] Loading pathway compound list ...")
    pathway_compound_names = set()
    with open(pathway_file, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            for c in parts[1].split(" // "):
                c = c.strip()
                if c:
                    pathway_compound_names.add(c)
    print(f"  Total pathway compounds: {len(pathway_compound_names)}")

    # Exclude currency metabolites from GNN metabolite node set
    metabolites_df = pd.DataFrame({"metabolite_name": sorted(pathway_compound_names)})
    before_n = len(metabolites_df)
    metabolites_df = metabolites_df[
        ~metabolites_df["metabolite_name"].str.lower().isin(currency_set)
    ].reset_index(drop=True)
    print(f"  Metabolite nodes after currency exclusion: "
        f"{before_n} -> {len(metabolites_df)} "
        f"(-{before_n - len(metabolites_df)} removed)")

    # Drug / target node tables
    drugs_df = pd.DataFrame({
        "drug_name": sorted(drug_target_edges["drug_name"].astype(str).unique())
    })
    targets_df = pd.DataFrame({
        "target_id": sorted(
            set(drug_target_edges["target_id"].astype(str)) |
            set(target_disease_edges["target_id"].astype(str))
        )
    })
    print(f"  Pre-SMILES: metabolites={len(metabolites_df)}, "
        f"drugs={len(drugs_df)}, targets={len(targets_df)}")

    # ── 16.4 SMILES Lookup via InChIKey & Local File ───────────────────
    print("\n[4/9] Loading SMILES via Local file & PubChem fallback ...")
    # 1. 파일에서 대명사 - InChIKey 및 로컬 SMILES 데이터 동시 로드
    comp_inchikeys, comp_file_smiles = load_compound_data(inchikey_file)
    
    # 2. 로컬 데이터 우선 매칭 후, 누락본만 PubChem 비동기 쿼리
    print(f"  Resolving SMILES for {len(metabolites_df)} metabolites ...")
    met_smiles_map = fetch_metabolite_smiles_concurrently(
        metabolites_df["metabolite_name"].tolist(), 
        comp_inchikeys, 
        comp_file_smiles, 
        max_workers=5
    )
    
    # 3. 데이터프레임에 매핑 결과 반영
    metabolites_df["smiles"] = metabolites_df["metabolite_name"].map(met_smiles_map)
    print(f"  Metabolites successfully matched with SMILES: {metabolites_df['smiles'].notna().sum()}/{len(metabolites_df)}")
    print(f"  Fetching drug SMILES for {len(drugs_df)} ChEMBL drugs ...")
    drug_smiles_df = fetch_smiles_concurrently(drugs_df["drug_name"].tolist(),
                                            max_workers=5)
    drug_smiles_df["merge_name"] = drug_smiles_df["Name"].str.lower()
    drugs_df["merge_name"]       = drugs_df["drug_name"].str.lower()
    drugs_df = (drugs_df
                .merge(drug_smiles_df[["merge_name", "smiles"]],
                    on="merge_name", how="left")
                .drop(columns=["merge_name"]))

    # Drop rows without SMILES
    metabolites_df = metabolites_df.dropna(subset=["smiles"]).reset_index(drop=True)
    drugs_df       = drugs_df.dropna(subset=["smiles"]).reset_index(drop=True)
    drug_target_edges = drug_target_edges[
        drug_target_edges["drug_name"].isin(drugs_df["drug_name"])
    ].reset_index(drop=True)
    print(f"  After SMILES filter: metabolites={len(metabolites_df)}, "
        f"drugs={len(drugs_df)}, drug-target={len(drug_target_edges)}")

    # ── 16.5  Morgan fingerprints ─────────────────────────────────────────
    print("\n[5/9] Computing Morgan fingerprints ...")
    metabolites_df["fp"] = metabolites_df["smiles"].apply(smiles_to_fp)
    drugs_df["fp"]       = drugs_df["smiles"].apply(smiles_to_fp)
    metabolites_df = metabolites_df.dropna(subset=["fp"]).reset_index(drop=True)
    drugs_df       = drugs_df.dropna(subset=["fp"]).reset_index(drop=True)
    print(f"  After FP filter: metabolites={len(metabolites_df)}, "
        f"drugs={len(drugs_df)}")

    # ── 16.6  Tanimoto metabolite->drug edges ─────────────────────────────
    print(f"\n[6/9] Tanimoto TOP-{args.top_k_tanimoto} metabolite->drug edges ...")
    drug_names = drugs_df["drug_name"].tolist()
    drug_fps   = drugs_df["fp"].tolist()
    sim_rows   = []
    for met_row in tqdm(metabolites_df.itertuples(index=False),
                        total=len(metabolites_df)):
        sims    = DataStructs.BulkTanimotoSimilarity(met_row.fp, drug_fps)
        top_idx = np.argsort(sims)[::-1][:args.top_k_tanimoto]
        for j in top_idx:
            sim_rows.append({"metabolite_name": met_row.metabolite_name,
                            "drug_name": drug_names[j],
                            "tanimoto": float(sims[j])})
    metabolite_drug_edges = pd.DataFrame(sim_rows)
    print(f"  Metabolite-drug edges: {len(metabolite_drug_edges)}")

    # ── 16.7  Index mappings (NO disease, NO gene) ────────────────────────
    met2idx,    idx2met    = make_mapping(metabolites_df["metabolite_name"])
    drug2idx,   idx2drug   = make_mapping(drugs_df["drug_name"])
    target2idx, idx2target = make_mapping(targets_df["target_id"])

    # ── 16.8  Build HeteroData ────────────────────────────────────────────
    print("\n[7/9] Building HeteroData (metabolite / drug / target only) ...")
    data = HeteroData()
    data["metabolite"].num_nodes = len(met2idx)
    data["drug"].num_nodes       = len(drug2idx)
    data["target"].num_nodes     = len(target2idx)

    md_edge = build_edge_index(metabolite_drug_edges,
                                "metabolite_name", "drug_name",
                                met2idx, drug2idx)
    data["metabolite", "similar_to",     "drug"      ].edge_index = md_edge
    data["drug",       "rev_similar_to", "metabolite"].edge_index = md_edge.flip(0)

    dt_edge = build_edge_index(drug_target_edges,
                                "drug_name", "target_id",
                                drug2idx, target2idx)
    data["drug",   "inhibits",     "target"].edge_index = dt_edge
    data["target", "rev_inhibits", "drug"  ].edge_index = dt_edge.flip(0)

    print(data)

    # ── 16.9  RWPE (currency-filtered graph, PE only) ─────────────────────
    print("\n[8/9] Building pathway co-reaction graph & RWPE ...")
    comp2idx, A_pathway = build_pathway_graph(pathway_file, currency_set)

    if A_pathway.shape[0] > 0:
        print(f"  Computing RWPE (K={WALK_LEN}) ...")
        rwpe_full = compute_rwpe_matrix(A_pathway, WALK_LEN)
    else:
        print("  [WARNING] Empty pathway graph -- RWPE = zeros")
        rwpe_full = np.zeros((0, WALK_LEN), dtype=np.float32)

    rwpe_met = get_metabolite_rwpe(
        metabolites_df["metabolite_name"].tolist(),
        comp2idx, rwpe_full, WALK_LEN)

    # Node features
    data["drug"].x = torch.stack([fp_to_tensor(fp) for fp in drugs_df["fp"]])
    met_fps  = torch.stack([fp_to_tensor(fp) for fp in metabolites_df["fp"]])
    met_rwpe = torch.tensor(rwpe_met, dtype=torch.float32)
    data["metabolite"].x = torch.cat([met_fps, met_rwpe], dim=-1)
    data["target"].x = torch.ones((data["target"].num_nodes, 1), dtype=torch.float)

    print(f"  metabolite.x : {data['metabolite'].x.shape}  "
        f"(Morgan {FP_DIM} + RWPE {WALK_LEN})")
    print(f"  drug.x       : {data['drug'].x.shape}")
    print(f"  target.x     : {data['target'].x.shape}")

    # ── 16.10  Train/Val/Test split ───────────────────────────────────────
    TASK_EDGE_TYPES = [
        ("metabolite", "similar_to", "drug"),
        ("drug",       "inhibits",   "target"),
    ]
    REV_EDGE_TYPES = [
        ("drug",   "rev_similar_to", "metabolite"),
        ("target", "rev_inhibits",   "drug"),
    ]

    transform = RandomLinkSplit(
        num_val=0.2,
        num_test=0.2,
        is_undirected=False,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
        edge_types=TASK_EDGE_TYPES,
        rev_edge_types=REV_EDGE_TYPES,
    )
    train_data, val_data, test_data = transform(data)

    def make_loaders(split_data, shuffle):
        return [
            LinkNeighborLoader(
                split_data,
                num_neighbors=[10, 5],
                edge_label_index=(et, split_data[et].edge_label_index),
                edge_label=split_data[et].edge_label,
                batch_size=args.batch_size,
                shuffle=shuffle,
            )
            for et in TASK_EDGE_TYPES
            if et in split_data.edge_types
            and hasattr(split_data[et], "edge_label_index")
        ]

    train_loader = make_loaders(train_data, shuffle=True)
    val_loader   = make_loaders(val_data,   shuffle=False)
    test_loader  = make_loaders(test_data,  shuffle=False)
    print(f"\n  Loaders -- train:{len(train_loader)} "
        f"val:{len(val_loader)} test:{len(test_loader)}")

    # ── 16.11  Model ──────────────────────────────────────────────────────
    print("\n[9/9] Building model ...")
    in_channels_dict = {nt: data[nt].x.size(-1) for nt in data.node_types}
    print(f"  in_channels_dict: {in_channels_dict}")

    model = InductiveHANLinkPredictor(
        metadata=data.metadata(),
        in_channels_dict=in_channels_dict,
        hidden_channels=args.hidden,
        out_channels=args.hidden,
        heads=args.heads,
        dropout=args.dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.lr, weight_decay=1e-4)

    # ── 16.12  Training ───────────────────────────────────────────────────
    print(f"\n[Training] {args.epochs} epochs ...")
    valid_train_etypes = [et for et in TASK_EDGE_TYPES
                        if et in train_data.edge_types
                        and hasattr(train_data[et], "edge_label_index")]
    valid_val_etypes   = [et for et in TASK_EDGE_TYPES
                        if et in val_data.edge_types
                        and hasattr(val_data[et], "edge_label_index")]

    best_val_ap, best_state = -1, None
    history = {"train_loss": [], "val_ap": [], "val_f1": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        iterators = [(et, iter(loader))
                    for et, loader in zip(valid_train_etypes, train_loader)]
        active = iterators[:]

        while active:
            et, it = active.pop(0)
            try:
                batch = next(it)
                active.append((et, it))
            except StopIteration:
                continue

            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            loss = compute_loss(model, batch, [et], DEVICE)
            if loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

        avg_loss = total_loss / max(1, n_batches)
        history["train_loss"].append(avg_loss)

        if epoch % 5 == 0:
            val_res = evaluate_auc_ap(model, val_loader, valid_val_etypes, DEVICE)
            if not val_res:
                print(f"  Epoch {epoch:03d} | Loss {avg_loss:.4f} | (no val results)")
                continue
            mean_ap = np.nanmean([v["AP"] for v in val_res.values()])
            mean_f1 = np.nanmean([v["F1"] for v in val_res.values()])

            if mean_ap > best_val_ap:
                best_val_ap = mean_ap
                best_state  = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}

            history["val_ap"].append((epoch, mean_ap))
            history["val_f1"].append((epoch, mean_f1))
            print(f"  Epoch {epoch:03d} | Loss {avg_loss:.4f} | "
                f"Val AP {mean_ap:.4f} | Val F1 {mean_f1:.4f}")
            for et, m in val_res.items():
                print(f"    {et[1]}: AUC={m['AUC']:.4f} "
                    f"AP={m['AP']:.4f} F1={m['F1']:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    print(f"\nTraining finished. Best Val AP: {best_val_ap:.4f}")

    # ── 16.13  Test evaluation ────────────────────────────────────────────
    valid_test_etypes = [et for et in TASK_EDGE_TYPES
                        if et in test_data.edge_types
                        and hasattr(test_data[et], "edge_label_index")]
    test_res = evaluate_auc_ap(model, test_loader, valid_test_etypes, DEVICE)
    print("\n[Test results]")
    for et, m in test_res.items():
        print(f"  {et[1]}: AUC={m['AUC']:.4f} AP={m['AP']:.4f} F1={m['F1']:.4f}")

    # ── 16.14  Metabolite ranking via target-chain ─────────────────────────
    print(f"\n[Ranking] Metabolites via target-chain score (top {args.top_k_rank}) ...")
    metabolite_rank_df = rank_metabolites_via_target_chain(
        model, data, target_disease_edges,
        target2idx, idx2met, DEVICE, top_k=args.top_k_rank)
    print(metabolite_rank_df.head(20).to_string(index=False))

    # ── 16.15  Path explanation ────────────────────────────────────────────
    print("\n[Path explanation] Building metabolite -> drug -> target -> disease paths ...")
    path_df = build_path_explanation(
        metabolites_df, metabolite_drug_edges,
        drug_target_edges, target_disease_edges)
    print(f"  Path records: {len(path_df)}")

    # ── 16.16  Evidence-calibrated final ranking ──────────────────────────
    final_df = build_final_ranking(
        metabolite_rank_df, path_df, metabolites_df=metabolites_df,
        disease_id="ALOPECIA",
        alopecia_mode=args.alopecia_mode,
        candidate_min_tanimoto=args.candidate_min_tanimoto,
        tanimoto_midpoint=args.tanimoto_midpoint,
        tanimoto_temperature=args.tanimoto_temperature,
        w_gnn=args.w_gnn, w_path=args.w_path, w_chem=args.w_chem,
        w_target=args.w_target, w_breadth=args.w_breadth,
        artifact_penalty_strength=args.artifact_penalty_strength,
        include_path_candidates=True)
    print("\n[Evidence-calibrated final ranking - top 20]")
    print(final_df[["final_rank", "metabolite_name", "candidate_class",
                    "final_score", "raw_evidence_score", "artifact_penalty",
                    "metabolite_gnn_score", "gnn_score_scaled",
                    "max_tanimoto", "mean_top3_tanimoto", "chem_similarity_score",
                    "best_weighted_path_score", "path_score_scaled",
                    "target_relevance_score", "evidence_breadth_score",
                    "num_unique_drugs", "num_unique_targets",
                    "best_similar_drug", "best_target"]].head(20).to_string(index=False))

    # ── 16.17  MMR re-ranking among already-good candidates ───────────────
    mmr_df = mmr_rerank(
        final_df, metabolite_drug_edges, metabolites_df, rwpe_met,
        score_col="final_score",
        top_k=args.mmr_top_k,
        candidate_k=100,
        lambda_mmr=args.lambda_mmr,
        alpha=args.mmr_alpha,
        min_score_quantile=args.mmr_min_score_quantile)
    print("\n[MMR re-ranked top metabolites - quality-filtered]")
    print(mmr_df[["mmr_rank", "metabolite_name", "candidate_class",
                "final_score", "mmr_score", "artifact_penalty",
                "max_tanimoto", "target_relevance_score",
                "best_similar_drug", "best_target"]].to_string(index=False))

    # ── 16.18  Path explanation for top MMR hits ──────────────────────────
    print("\n[Path explanation for MMR top hits]")
    show_path_explanation(mmr_df, path_df, top_n=5, disease_id="ALOPECIA")

    # ── 16.19  Inductive new-drug inference ───────────────────────────────
    print(f"\n[Inductive inference] new drug: {args.new_drug_name}")
    try:
        new_drug_met_df = rank_metabolites_for_new_drug(
            model, data,
            new_drug_smiles=args.new_drug_smiles,
            new_drug_name=args.new_drug_name,
            met2idx=met2idx, target2idx=target2idx,
            idx2met=idx2met, device=DEVICE,
            top_k=args.top_k_rank)

        new_drug_final_df = build_final_ranking(
            new_drug_met_df.rename(columns={"new_drug_score": "metabolite_gnn_score"}),
            path_df, metabolites_df=metabolites_df,
            disease_id="ALOPECIA",
            alopecia_mode=args.alopecia_mode,
            candidate_min_tanimoto=args.candidate_min_tanimoto,
            tanimoto_midpoint=args.tanimoto_midpoint,
            tanimoto_temperature=args.tanimoto_temperature,
            w_gnn=args.new_drug_w_gnn, w_path=args.new_drug_w_path,
            w_chem=args.new_drug_w_chem, w_target=args.new_drug_w_target,
            w_breadth=args.new_drug_w_breadth,
            artifact_penalty_strength=args.artifact_penalty_strength,
            include_path_candidates=True)

        new_drug_mmr_df = mmr_rerank(
            new_drug_final_df, metabolite_drug_edges, metabolites_df, rwpe_met,
            score_col="final_score",
            top_k=args.mmr_top_k,
            candidate_k=100,
            lambda_mmr=args.lambda_mmr,
            alpha=args.mmr_alpha,
            min_score_quantile=args.mmr_min_score_quantile)

        print(f"\n  '{args.new_drug_name}' -> top metabolites (MMR, evidence-calibrated):")
        print(new_drug_mmr_df[["mmr_rank", "metabolite_name", "candidate_class",
                                "final_score", "mmr_score", "artifact_penalty",
                                "max_tanimoto", "target_relevance_score",
                                "best_similar_drug", "best_target"]].to_string(index=False))
        new_drug_final_out_csv = os.path.join(
            SCRIPT_DIR, f"{args.new_drug_name}_metabolite_final_ranking_evidence_calibrated.csv")
        new_drug_mmr_out_csv = os.path.join(
            SCRIPT_DIR, f"{args.new_drug_name}_metabolite_mmr_evidence_calibrated.csv")
        new_drug_final_df.to_csv(new_drug_final_out_csv, index=False)
        new_drug_mmr_df.to_csv(new_drug_mmr_out_csv, index=False)
        print(f"  New-drug final ranking saved -> {new_drug_final_out_csv}")
        print(f"  New-drug MMR ranking saved -> {new_drug_mmr_out_csv}")
        show_path_explanation(new_drug_mmr_df, path_df,
                            top_n=3, disease_id="ALOPECIA")
    except Exception as e:
        print(f"  [WARNING] Inductive inference failed: {e}")

    # ── 16.20  Save results ────────────────────────────────────────────────
    final_out_csv = os.path.join(SCRIPT_DIR, "metabolite_alopecia_final_ranking_evidence_calibrated.csv")
    out_csv = os.path.join(SCRIPT_DIR, "metabolite_alopecia_mmr_evidence_calibrated.csv")
    final_df.to_csv(final_out_csv, index=False)
    mmr_df.to_csv(out_csv, index=False)
    print(f"\nFinal ranking saved -> {final_out_csv}")
    print(f"MMR ranking saved -> {out_csv}")

    # Training curve
    if history["train_loss"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history["train_loss"])
        axes[0].set(title="Train Loss", xlabel="Epoch", ylabel="Loss")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        if history["val_ap"]:
            ep, ap = zip(*history["val_ap"])
            _,  f1 = zip(*history["val_f1"])
            axes[1].plot(ep, ap, "s--", label="Val AP")
            axes[1].plot(ep, f1, "o-",  label="Val F1")
            axes[1].set(title="Validation Metrics", xlabel="Epoch")
            axes[1].legend()
            axes[1].grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        curve_path = os.path.join(SCRIPT_DIR, "training_curve.png")
        plt.savefig(curve_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Training curve saved -> {curve_path}")


if __name__ == "__main__":
    main()