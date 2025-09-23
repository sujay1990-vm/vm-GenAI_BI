from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from scipy.sparse import hstack
import pandas as pd
import numpy as np

import numpy as np
import pandas as pd
from typing import List, Optional
from scipy.sparse import hstack, csr_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import NearestNeighbors

# Reuse your existing dataset load
df_claims = pd.read_csv("claims_with_notes.csv")

_HASH = HashingVectorizer(n_features=2048, alternate_sign=False, norm="l2")

def _features(df, text_cols, num_cols):
    Xn = Xt = None
    if num_cols:
        arr = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
        arr = StandardScaler(with_mean=True, with_std=True).fit_transform(arr)
        Xn = csr_matrix(arr)
    if text_cols:
        txt = df[text_cols].fillna("").agg(" | ".join, axis=1)
        Xt = _HASH.transform(txt)
    X = Xt if Xn is None else Xn if Xt is None else hstack([Xn, Xt], format="csr")
    return normalize(X, norm="l2", copy=False)

def _prefilter(df, idx, cols, min_pool=150):
    if not cols: return None
    mask = pd.Series(True, index=df.index)
    t = df.loc[idx]
    for c in cols:
        if c in df.columns and pd.notna(t[c]) and str(t[c]).strip():
            mask &= (df[c].astype(str) == str(t[c]))
        if mask.sum() < min_pool: return None
    i = np.flatnonzero(mask.values)
    return i if len(i) >= min_pool else None

@tool
def dynamic_similar_claims_tool(
    claim_number: str,
    text_cols: List[str],
    num_cols: List[str],
    k: int = 5,
    prefilter_cols: Optional[List[str]] = None
) -> str:
    """
    LLM-driven similarity. The LLM chooses text_cols/num_cols (after get_schema_tool).
    No defaults, no heuristics. Fast KNN over combined numeric+text features.
    """
    print("💾 [Tool] Calling Similarity Tool...")
    df = df_claims
    if "Claim Number" not in df.columns:
        return "❌ 'Claim Number' column missing."
    m = df["Claim Number"].astype(str) == str(claim_number)
    if not m.any():
        return f"❌ Claim Number {claim_number} not found."
    if not text_cols and not num_cols:
        return "❌ Provide text_cols and/or num_cols (chosen by LLM)."

    # strict: only keep existing columns
    text_cols = [c for c in text_cols if c in df.columns]
    num_cols  = [c for c in num_cols  if c in df.columns]
    if not text_cols and not num_cols:
        return "❌ None of the provided columns exist in the dataset."

    idx = int(df.index[m][0])

    cand = _prefilter(df, idx, [c for c in (prefilter_cols or []) if c in df.columns])
    if cand is not None:
        sub = df.iloc[cand].reset_index(drop=True)
        target_sub = int(np.where(cand == idx)[0][0])
        X = _features(sub, text_cols, num_cols)
        nn = NearestNeighbors(metric="cosine", algorithm="brute").fit(X)
        dist, nbr = nn.kneighbors(X[target_sub], n_neighbors=min(k+1, len(sub)))
        nbr = nbr.ravel(); dist = dist.ravel()
        full_idx = df.index[cand[nbr]]
        mask = full_idx != idx
        neigh_idx, dist = full_idx[mask][:k], dist[mask][:k]
    else:
        X = _features(df, text_cols, num_cols)
        nn = NearestNeighbors(metric="cosine", algorithm="brute").fit(X)
        dist, nbr = nn.kneighbors(X[idx], n_neighbors=min(k+1, len(df)))
        nbr = nbr.ravel(); dist = dist.ravel()
        mask = nbr != idx
        neigh_idx, dist = nbr[mask][:k], dist[mask][:k]

    # lightweight explanation (same style you had)
    tgt = df.loc[idx]
    num_std = {c: (df[c].dropna().std(ddof=0) or 1.0) for c in num_cols}
    blocks = []
    for j, d in zip(neigh_idx, dist):
        row = df.loc[j]
        sim = round(1 - float(d), 2)
        matches, diffs = [], []
        for c in num_cols:
            v1, v2 = tgt.get(c), row.get(c)
            if pd.isna(v1) or pd.isna(v2): continue
            s = 1 - (abs(float(v1)-float(v2))/(3*num_std[c]))
            (matches if s>=0.85 else diffs if s<=0.5 else matches).append(
                f"{c} ({'close' if s>=0.85 else 'diff' if s<=0.5 else 'near'}: {v1} vs {v2})"
            )
        for c in text_cols:
            a = str(tgt.get(c,"")).strip().lower()
            b = str(row.get(c,"")).strip().lower()
            if a and a==b: matches.append(f"{c} (match: {row.get(c,'')})")
            elif a!=b:     diffs.append(f"{c} (target: {tgt.get(c,'')} vs match: {row.get(c,'')})")
        blocks.append(
f"""📄 **Claim {row.get('Claim Number','N/A')}** | Policy: {row.get('Policy Number','N/A')}
Similarity Score: {sim}

✅ Top Matching Features:
{chr(10).join('• '+m for m in matches[:5]) if matches else '• None'}

⚠️ Differences:
{chr(10).join('• '+d for d in diffs[:5]) if diffs else '• None'}
""".rstrip()
        )
    return "\n\n".join(blocks) if blocks else "No similar claims found."
