from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from scipy.sparse import hstack
import pandas as pd
import numpy as np
from typing import List
from scipy.sparse import hstack, csr_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import NearestNeighbors

# 🔁 Load and cache your dataset once
df_claims = pd.read_csv("claims_with_notes.csv")

# ---- Global, reusable components & caches ----
_HASH = HashingVectorizer(n_features=4096, alternate_sign=False, norm="l2")
_num_imputer = None
_num_scaler  = None
_X_cache     = None
_nn_cache    = None
_row_index_by_claim = None
_cache_fingerprint  = None

def _build_or_get_cache(df: pd.DataFrame, text_cols: List[str], num_cols: List[str]):
    """
    Build (once) and cache:
      - numeric imputer/scaler
      - sparse feature matrix X (L2-normalized)
      - NearestNeighbors index (cosine)
      - claim_number -> row index map
    Reuse on subsequent calls if df hasn't changed shape/columns.
    """
    global _num_imputer, _num_scaler, _X_cache, _nn_cache, _row_index_by_claim, _cache_fingerprint

    # crude but effective fingerprint: shape + column names tuple
    fp = (df.shape, tuple(df.columns))
    if _X_cache is not None and _nn_cache is not None and _cache_fingerprint == fp:
        return  # cache is warm

    # Ensure columns exist (subset in case some are missing)
    text_cols = [c for c in text_cols if c in df.columns]
    num_cols  = [c for c in num_cols  if c in df.columns]

    # ---- Numeric block: impute + scale -> CSR
    Xn = None
    if num_cols:
        _num_imputer = SimpleImputer(strategy="mean")
        _num_scaler  = StandardScaler(with_mean=True, with_std=True)
        num_arr = _num_imputer.fit_transform(df[num_cols])
        num_arr = _num_scaler.fit_transform(num_arr)
        Xn = csr_matrix(num_arr)

    # ---- Text block: join cols -> HashingVectorizer (no fit)
    Xt = None
    if text_cols:
        text_joined = df[text_cols].fillna("").agg(" | ".join, axis=1)
        Xt = _HASH.transform(text_joined)

    # ---- Combine & normalize
    if Xn is not None and Xt is not None:
        X = hstack([Xn, Xt], format="csr")
    elif Xn is not None:
        X = Xn
    elif Xt is not None:
        X = Xt
    else:
        X = csr_matrix((len(df), 1), dtype=float)

    X = normalize(X, norm="l2", copy=False)

    # ---- Build NN index (cosine) for top-K queries
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(X)

    # ---- Cache everything
    _X_cache = X
    _nn_cache = nn
    _row_index_by_claim = {str(cn): i for i, cn in enumerate(df["Claim Number"].astype(str).values)}
    _cache_fingerprint = fp

@tool
def similar_claims_tool(claim_number: str) -> str:
    """
    Returns the top 5 most similar claims to a given claim number, including a breakdown
    of which features contributed to similarity or difference.
    """
    print("💾 [Tool] Calling Similarity Tool (cached+fast)...")
    df = df_claims

    if "Claim Number" not in df.columns:
        return "❌ 'Claim Number' column missing."
    claim_number_str = str(claim_number)
    if claim_number_str not in set(df["Claim Number"].astype(str).values):
        return f"❌ Claim Number {claim_number} not found."

    # ---- Static relevant columns (as in your original)
    text_cols = [
        'Loss cause', 'Loss Location State', 'Vehicle Make', 'Vehicle Model',
        'Damage Description', 'Claim Status', 'Litigation',
        'Medical & Injury Documentation', 'Medical Reports', 'Hospital Records',
        'Third-Party Information', 'Subro Opportunity', 'Third-Party Insurance',
        'Third-Party Claim Form','witness_available'
    ]
    num_cols = [
        'Vehicle Year', 'Repair Estimate', 'Repair Bill', 'Medical bill',
        'Total Claim Bill', 'fault_rating', 'Time_to_Report', 'subrogation_score',
        'recovery_amount', 'recovery_rate', 'pursuit_cost','recovery_gap_amount'
    ]

    # Keep your dtype re-check logic (object -> text, else numeric) but do it once
    keep_cols = [c for c in text_cols + num_cols if c in df.columns]
    df_features = df[keep_cols].copy()
    text_cols = [c for c in df_features.columns if df_features[c].dtype == 'object']
    num_cols  = [c for c in df_features.columns if c not in text_cols]

    # ---- Build or reuse cached feature matrix + NN index
    _build_or_get_cache(df, text_cols, num_cols)

    # ---- Locate target row
    idx = _row_index_by_claim[str(claim_number)]

    # ---- Top-K neighbors (no full NxN)
    # n_neighbors = self + 5 matches => ask for 6, drop self
    distances, indices = _nn_cache.kneighbors(_X_cache[idx], n_neighbors=min(6, len(df)))
    distances = distances.ravel()
    indices = indices.ravel()

    # Drop self
    mask = indices != idx
    indices = indices[mask][:5]
    distances = distances[mask][:5]

    target_row = df.iloc[idx]

    # Precompute std for numeric explanations
    num_std = {}
    for col in num_cols:
        try:
            s = float(df[col].dropna().std(ddof=0))
            num_std[col] = s if s > 0 else 1.0
        except Exception:
            num_std[col] = 1.0

    output_blocks = []
    for j, d in zip(indices, distances):
        row = df.iloc[j]
        sim_score = round(1 - float(d), 2)  # cosine similarity

        matches, diffs = [], []

        # Numeric comparisons (same logic as yours)
        for col in num_cols:
            try:
                v1, v2 = target_row[col], row[col]
                if pd.isna(v1) or pd.isna(v2):
                    continue
                diff = abs(float(v1) - float(v2))
                std = num_std.get(col, 1.0)
                similarity_score = 1 - (diff / (3 * std))
                if similarity_score >= 0.85:
                    matches.append(f"{col} (close: {v1} vs {v2})")
                elif similarity_score <= 0.5:
                    diffs.append(f"{col} (diff: {v1} vs {v2})")
            except Exception:
                continue

        # Text comparisons (exact match)
        for col in text_cols:
            val1 = str(target_row[col]).strip().lower()
            val2 = str(row[col]).strip().lower()
            if val1 == val2 and val1 != "":
                matches.append(f"{col} (match: {target_row[col]})")
            elif val1 != val2:
                diffs.append(f"{col} (target: {target_row[col]} vs match: {row[col]})")

        block = f"""📄 **Claim {row.get('Claim Number','N/A')}** | Policy: {row.get('Policy Number','N/A')}
Similarity Score: {sim_score}

✅ Top Matching Features:
{chr(10).join(['• ' + m for m in matches[:5]]) if matches else '• None'}

⚠️ Differences:
{chr(10).join(['• ' + d for d in diffs[:5]]) if diffs else '• None'}
"""
        output_blocks.append(block)

    return "\n\n".join(output_blocks)