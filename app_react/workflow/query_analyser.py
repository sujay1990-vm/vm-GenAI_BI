from pydantic import BaseModel, Field
from typing import List, Literal
from langgraph.store.base import BaseStore
from llm import get_llm

llm = get_llm()

class SubQuery(BaseModel):
    query: str = Field(
        description="An atomic sub-question derived from the user's original query."
    )
    intent: Literal["sql", "rag", "other"] = Field(
        description="The type of execution this sub-query requires: 'sql' for structured database queries, 'rag' for document retrieval and summarization, or 'other' for fallback/general."
    )

class SubQueryList(BaseModel):
    subqueries: List[SubQuery] = Field(
        description="List of parsed and intent-classified sub-queries from the original user question."
    )

from langchain.prompts import ChatPromptTemplate

analyzer_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are a query analysis agent. Your job is to:
1. Understand the user query.
2. Break it into smaller subqueries if it contains multiple parts.
3. Classify each subquery into one of: 
   - 'sql' (structured data, e.g., counts, trends, metrics),
   - 'rag' (policy/guidelines/instructions from unstructured documents),
   - 'other' (if it doesn't fit the above).
Output a list of subqueries as structured JSON.
"""),
    ("human", "User Query:\n{user_query}\n\nReturn the parsed and classified subqueries:")
])   

structured_llm_query_analysis = llm.with_structured_output(
    schema=SubQueryList,
    method="function_calling"  # Optional but good for structured output
)

query_analysis_chain = analyzer_prompt | structured_llm_query_analysis

from langchain_core.tools import tool

@tool
def query_analyzer_tool(user_query: str) -> str:
    """
    Analyzes a user query to extract subqueries and their intents.
    Returns a formatted string summary of the extracted subqueries.
    """
    print("🔍 Running Query Analyzer Tool")

    structured = query_analysis_chain.invoke({"user_query": user_query})

    print("🧠 Subqueries extracted:")
    for sq in structured.subqueries:
        print(f"- [{sq.intent.upper()}] {sq.query}")

    if not structured.subqueries:
        return "No subqueries identified."

    # Return a readable summary string (ReAct tools expect string outputs)
    summary = "\n".join([f"[{sq.intent}] {sq.query}" for sq in structured.subqueries])
    return summary