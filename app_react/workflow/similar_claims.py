from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from scipy.sparse import hstack
import pandas as pd
import numpy as np

# 🔁 Load and cache your dataset once
df_claims = pd.read_csv("claims_with_notes.csv")

@tool
def similar_claims_tool(claim_number: str) -> str:
    """
    Returns the top 5 most similar claims to a given claim number, including a breakdown
    of which features contributed to similarity or difference.
    """
    print("💾 [Tool] Calling Similarity Tool...")
    df = df_claims.copy()
    if claim_number not in df['Claim Number'].values:
        return f"❌ Claim Number {claim_number} not found."

    # Updated relevant columns
    text_cols = [
        'Loss cause', 'Loss Location State', 'Vehicle Make', 'Vehicle Model',
        'Damage Description', 'Claim Status', 'Litigation',
        'Medical & Injury Documentation', 'Medical Reports', 'Hospital Records',
        'Third-Party Information', 'Subro Opportunity', 'Third-Party Insurance',
        'Third-Party Claim Form'
    ]

    num_cols = ['Vehicle Year', 'Repair Estimate', 'Repair Bill', 'Medical bill', 'Total Claim Bill']

    df_features = df[text_cols + num_cols].copy()

    text_cols = [col for col in df_features.columns if df_features[col].dtype == 'object']
    num_cols = [col for col in df_features.columns if col not in text_cols]

    # print(f"\n🧾 Text Columns Used ({len(text_cols)}): {text_cols}")
    # print(f"🔢 Numeric Columns Used ({len(num_cols)}): {num_cols}\n")

    # Numeric matrix
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    num_matrix = scaler.fit_transform(imputer.fit_transform(df_features[num_cols]))

    # Text matrix
    text_combined = df_features[text_cols].fillna('').agg(' '.join, axis=1)
    tfidf = TfidfVectorizer(max_features=500)
    text_matrix = tfidf.fit_transform(text_combined)

    combined = hstack([num_matrix, text_matrix]).tocsr()

    idx = df[df['Claim Number'] == claim_number].index[0]
    similarities = cosine_similarity(combined[idx], combined).flatten()
    df['Similarity'] = similarities

    top_matches = df.sort_values(by='Similarity', ascending=False).iloc[1:6]  # Exclude self
    target_row = df.iloc[idx]

    output_blocks = []

    for _, row in top_matches.iterrows():
        claim_id = row["Claim Number"]
        policy_id = row["Policy Number"]
        sim_score = round(row["Similarity"], 2)

        matches = []
        diffs = []

        # Numeric comparisons
        for col in num_cols:
            try:
                val1, val2 = target_row[col], row[col]
                diff = abs(val1 - val2)
                std = np.std(df[col].dropna())
                if std == 0: continue
                similarity_score = 1 - (diff / (3 * std))
                if similarity_score >= 0.85:
                    matches.append(f"{col} (close: {val1} vs {val2})")
                elif similarity_score <= 0.5:
                    diffs.append(f"{col} (diff: {val1} vs {val2})")
            except:
                continue

        # Text comparisons
        for col in text_cols:
            val1 = str(target_row[col]).strip().lower()
            val2 = str(row[col]).strip().lower()
            if val1 == val2 and val1 != "":
                matches.append(f"{col} (match: {target_row[col]})")
            elif val1 != val2:
                diffs.append(f"{col} (target: {target_row[col]} vs match: {row[col]})")

        block = f"""📄 **Claim {claim_id}** | Policy: {policy_id}
Similarity Score: {sim_score}

✅ Top Matching Features:
{chr(10).join(['• ' + m for m in matches[:5]]) if matches else '• None'}

⚠️ Differences:
{chr(10).join(['• ' + d for d in diffs[:5]]) if diffs else '• None'}
"""
        output_blocks.append(block)

    return "\n\n".join(output_blocks)