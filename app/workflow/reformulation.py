from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from llm import get_llm

llm = get_llm()
reformulation_prompt = """
Given the chat history and the latest user question, which might reference context in the chat history, 
reformulate the question into a standalone question that can be understood without the chat history. 
If the question is related to the most recent queries or requires context from the chat history to be understood, 
include that context in the reformulated question. Do NOT answer the question; just provide the reformulated version.
"""

reformulation_decision_prompt = """
You are an assistant that determines whether the user query requires reformulation using prior chat history.
Answer True only if the query is ambiguous, vague, or clearly follows from prior conversation.
Answer False if the query is already self-contained and understandable on its own.

Respond using the function format below.
"""

class ReformulatedQuery(BaseModel):
    """Rewritten query that includes necessary context."""
    reformulated_query: str = Field(
        description="A rewritten version of the user's question that includes necessary context from prior conversation."
    )

class ReformulationDecision(BaseModel):
    """Boolean flag for whether reformulation is required."""
    requires_reformulation: bool = Field(
        description="True if query depends on chat history or is ambiguous."
    )



# Prompt templates
clarity_prompt = ChatPromptTemplate.from_messages([
    ("system", reformulation_decision_prompt.strip()),
    ("human", "Chat History:\n{memory}\n\nCurrent User Question:\n{user_query}\n\nDoes this query require reformulation?")
])

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", reformulation_prompt.strip()),
    ("human", "Chat History:\n{memory}\n\nCurrent User Question:\n{user_query}\n\nRewritten Question:")
])

# LLM Structured Output
structured_clarity_resolution = llm.with_structured_output(
    schema=ReformulationDecision,
    method="function_calling"
)

structured_llm_reformulation_resolution = llm.with_structured_output(
    schema=ReformulatedQuery,
    method="function_calling"
)

# Composable chains
chain_query_clarity_check = clarity_prompt | structured_clarity_resolution
chain_reformulation = contextualize_prompt | structured_llm_reformulation_resolution


def query_clarity_and_reformulation_node(state: dict, config: dict) -> dict:
    print("🧐 Checking query clarity and possibly reformulating...")

    user_query = state["user_query"]
    memory = state.get("memory", "")
    # Step 1: Determine if reformulation is needed
    decision_result = chain_query_clarity_check.invoke({
        "user_query": user_query,
        "memory": memory,
    })

    print("🔍 Reformulation required:", decision_result.requires_reformulation)

    if decision_result.requires_reformulation:
        # Step 2: Reformulate the user query
        reformulated_result = chain_reformulation.invoke({
            "user_query": user_query,
            "memory": memory,
        })

        return {
            "reformulated_query": reformulated_result.reformulated_query,
            "reformulation_required": True
        }

    return {
        "reformulated_query": user_query,
        "reformulation_required": False
    }
