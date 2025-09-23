from langchain_core.tools import tool
from typing import TypedDict, List, Optional, List, Literal, Annotated
from llm import get_llm
import json
from langchain.prompts import ChatPromptTemplate

llm = get_llm()

@tool
def llm_similarity_explainer_tool(
    claim_rows: List[dict],
    used_text_cols: Optional[List[str]] = None,
    used_num_cols: Optional[List[str]] = None
) -> str:
    """
    Natural-language explanation of why the given claims are similar.
    Works with dynamic columns selected by the LLM.

    Args:
      claim_rows: List of claim dicts (output rows for the neighbors).
      used_text_cols / used_num_cols: Columns actually used to compute similarity.
                                      If omitted, the tool infers common columns.
    """
    print("💾 [Tool] Calling Similarity Explainer Tool...")

    if not claim_rows:
        return "❌ No claims provided to explain."

    # Columns to focus on = provided by caller, else intersect of keys across claims
    if used_text_cols or used_num_cols:
        focus_cols = list(dict.fromkeys((used_text_cols or []) + (used_num_cols or [])))
        # Keep only columns that exist in *all* rows
        row_key_sets = [set(r.keys()) for r in claim_rows]
        focus_cols = [c for c in focus_cols if all(c in ks for ks in row_key_sets)]
    else:
        common_keys = set(claim_rows[0].keys())
        for r in claim_rows[1:]:
            common_keys &= set(r.keys())
        # Filter out obvious identifiers unless caller asked for them
        id_like = {"Claim Number", "Policy Number", "Claimant Name"}
        focus_cols = [c for c in common_keys if c not in id_like]

    if not focus_cols:
        return "❌ No overlapping columns to analyze across the provided claims."

    system_prompt = """You are an insurance domain expert. The user provided a set of similar claims.
Explain, in clear English:
- Key common patterns across the focus columns (categorical alignments, close numeric ranges, recurring flags)
- Notable differences and what they might imply
- A concise reason these claims likely clustered together given the focus columns
Avoid merely listing values; synthesize patterns and relationships. Keep it succinct and actionable.
"""

    # Keep payload tight: include only focus columns and a small subset of identifiers for reference
    compact_rows = []
    for r in claim_rows:
        row_view = {k: r.get(k) for k in focus_cols}
        # Optional lightweight IDs for reference if present
        for idk in ["Claim Number", "Policy Number"]:
            if idk in r:
                row_view[idk] = r[idk]
        compact_rows.append(row_view)

    focus_info = {
        "focus_columns": focus_cols,
        "claims": compact_rows,
    }

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Focus columns:\n{cols}\n\nClaims (subset):\n{claims_json}\n\nExplain.")
    ])

    input_data = {
        "cols": ", ".join(focus_cols),
        "claims_json": json.dumps(focus_info, indent=2)
    }

    chain = prompt | llm
    return chain.invoke(input_data).content
