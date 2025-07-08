from langchain_core.tools import tool
from typing import TypedDict, List, Optional, List, Literal, Annotated
from llm import get_llm
import json
from langchain.prompts import ChatPromptTemplate

llm = get_llm()

@tool
def llm_similarity_explainer_tool(claim_rows: List[dict]) -> str:
    """
    Explains in natural language why a list of 5 claims are similar.
    Each claim must be a dictionary with the following keys:
    
    - 'Loss cause'
    - 'Loss Location State'
    - 'Vehicle Make'
    - 'Vehicle Model'
    - 'Damage Description'
    - 'Claim Status'
    - 'Litigation'
    - 'Medical & Injury Documentation'
    - 'Medical Reports'
    - 'Hospital Records'
    - 'Third-Party Information'
    - 'Subro Opportunity'
    - 'Third-Party Insurance'
    - 'Third-Party Claim Form'
    - 'Vehicle Year'
    - 'Repair Estimate'
    - 'Repair Bill'
    - 'Medical bill'
    - 'Total Claim Bill'

    These are the same fields used in the cosine similarity calculation.
    """
    print("💾 [Tool] Calling Similarity Explainer Tool...")
    system_prompt = """You are an insurance domain expert. The user has provided 5 claims that are considered similar. 
            Analyze the data across all claims and explain in English:
            - What key features or patterns are common across these claims
            - What are the notable differences
            - Why they might have been grouped together
            Avoid just repeating values — look for patterns, trends, and correlations.
            """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Here are the claims:\n\n{claims_json}")
    ])

    input_data = {
        "claims_json": json.dumps(claim_rows, indent=2)
    }

    chain = prompt | llm
    return chain.invoke(input_data).content