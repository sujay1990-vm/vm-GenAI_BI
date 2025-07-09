from langchain_core.tools import tool
from typing import TypedDict, List, Optional, List, Literal, Annotated
from llm import get_llm
import json
from langchain.prompts import ChatPromptTemplate
import pandas as pd
import joblib
import numpy as np
import random
from typing import Dict

llm = get_llm()

# Static setup
text_cols = [
    'Loss cause', 'Loss Location State', 'Vehicle Make', 'Vehicle Model',
    'Damage Description', 'Claim Status', 'Medical & Injury Documentation',
    'Medical Reports', 'Hospital Records', 'Third-Party Information',
    'Subro Opportunity', 'Third-Party Insurance', 'Third-Party Claim Form'
]
num_cols = ['Vehicle Year', 'Repair Estimate', 'Repair Bill', 'Medical bill', 'Total Claim Bill']
feature_cols = text_cols + num_cols

def convert_risk_score_to_percent(score: float) -> int:
    if score < 0.05:
        # Scale 0–0.05 to 0–39
        return int((score / 0.05) * 39)
    elif score < 0.10:
        # Scale 0.05–0.10 to 40–69
        return int(((score - 0.05) / 0.05) * 29 + 40)
    else:
        # Scale >0.10 to 70–100 (clip at 1.0)
        return int(min(((score - 0.10) / 0.90) * 30 + 70, 100))


@tool
def get_litigation_risk_score_tool(claim_id: str) -> Dict:
    """
    Returns litigation risk score and explanation for a given claim ID.
    Output includes score (0–1), top positive/negative features, and natural language explanation.
    """
    print("📚 [Tool] Invoking Litigation Risk model...")
    # Load artifacts
    model = joblib.load("logreg_litigation_model.pkl")
    
    df = pd.read_csv("claims_with_notes.csv")
    
    row = df[df["Claim Number"] == claim_id].copy()
    if row.empty:
        return {"error": f"Claim ID {claim_id} not found."}
    
    # Fill missing values (same as training)
    for col in num_cols:
        row[col] = row[col].fillna(df[col].median())
    for col in text_cols:
        choices = df[col].dropna().unique()
        row[col] = row[col].apply(lambda x: random.choice(choices) if pd.isna(x) else x)

    # Keep raw row for prediction
    row_raw = row[feature_cols].copy()

    # Predict
    risk_score = model.predict_proba(row_raw)[0][1]
    risk_score_percentage = convert_risk_score_to_percent(risk_score)
    # Extract internal model components
    preprocess = model.named_steps["preprocess"]
    classifier = model.named_steps["clf"]

    # Transform row using preprocessor
    row_vector = preprocess.transform(row_raw)
    feature_names = preprocess.get_feature_names_out()
    weights = classifier.coef_[0]
    values = row_vector[0]

    # Calculate contributions
    contributions = [
        (name, val * weight)
        for name, val, weight in zip(feature_names, values, weights)
        if val != 0
    ]
    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)

    top_pos = [f"{k}: {v:.2f}" for k, v in contributions if v > 0][:5]
    top_neg = [f"{k}: {v:.2f}" for k, v in contributions if v < 0][:5]

    # Prepare JSON
    claim_json = {
        "claim_id": claim_id,
        "risk_score_percentage": round(risk_score_percentage, 3),
        "top_positive_features": top_pos,
        "top_negative_features": top_neg
    }
    # LLM explanation
    system_prompt = """
    You are an insurance domain expert. A litigation prediction tool has analyzed a claim and produced a risk score and key feature contributions.

    - Scores < 40% = low risk
    - 40-69% = moderate risk
    - > 70% = high risk
    Based on the JSON output, explain:
    - What the risk score means
    - What factors increase or reduce litigation risk
    - Whether this claim is concerning
    Keep your tone concise, clear, and professional.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Here is the tool output with percentage-based risk score:\n\n{claim_json}")
    ])
    chain = prompt | llm
    explanation = chain.invoke({"claim_json": json.dumps(claim_json, indent=2)}).content

    # Final response
    return {
        "claim_id": claim_id,
        "risk_score_percentage": round(risk_score_percentage, 3),
        "top_positive_features": top_pos,
        "top_negative_features": top_neg,
        "explanation": explanation
    }