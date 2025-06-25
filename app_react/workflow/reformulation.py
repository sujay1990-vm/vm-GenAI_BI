from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from llm import get_llm
from langgraph.store.base import BaseStore
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

llm = get_llm()
reformulation_prompt = """
Given the chat history and the latest user question, which might reference context in the chat history, 
reformulate the question into a standalone question that can be understood without the chat history. 
If the question is related to the most recent queries or requires context from the chat history to be understood, 
include that context in the reformulated question. Do NOT answer the question; just provide the reformulated version.
"""

reformulation_decision_prompt = """
You are an assistant that determines whether the user query requires reformulation using prior chat history.

Answer True only if the query:
- contains vague references like "there", "that", "those", or
- depends on unresolved terms from prior chat history.

Answer False if the query is:
- grammatically clear,
- semantically complete, and
- understandable on its own.

Do not assume the user wants helpful context from memory unless ambiguity is explicitly present.

Respond using the function format below.
"""

class ReformulationDecision(BaseModel):
    """
    Determines whether the query should be reformulated based on chat history context.
    """
    requires_reformulation: bool = Field(
        description="True if the user query depends on chat history or is ambiguous. False if it is already clear and self-contained."
    )

class ReformulatedQuery(BaseModel):
    """
    Reformulated user query that is self-contained and does not require chat history for understanding.
    """
    reformulated_query: str = Field(
        description="A rewritten version of the user's question that includes necessary context from prior conversation."
    )
contextualize_prompt = ChatPromptTemplate.from_messages([ 
    ("system", reformulation_prompt.strip()),
    ("human", 
     "Chat History:\n{memory}\n\nCurrent User Question:\n{user_query}\n\nRewritten Question:")
        ])

clarity_prompt = ChatPromptTemplate.from_messages([
    ("system", reformulation_decision_prompt.strip()),
    ("human",
     "Chat History:\n{memory}\n\nCurrent User Question:\n{user_query}\n\nDoes this query require reformulation?")
])


structured_llm_reformulation_resolution = llm.with_structured_output(
    schema=ReformulatedQuery,
    method="function_calling"
)

structured_clarity_resolution = llm.with_structured_output(
    schema=ReformulationDecision,
    method="function_calling"
)

chain_reformulation = contextualize_prompt | structured_llm_reformulation_resolution
chain_query_clarity_check = clarity_prompt | structured_clarity_resolution


def query_reformulator_node(state: MessagesState):
    print("🔁 [Node] Reformulating query using structured LLM...")

    user_msg = next((m for m in reversed(state["messages"]) if m.type == "human"), None)
    user_query = user_msg.content if user_msg else ""
    memory = next(
            (m.content for m in reversed(state["messages"])
            if m.type == "system" and "past memory retrieved" in m.content),
            ""
        )
    
    clarity_response: ReformulationDecision = chain_query_clarity_check.invoke({
            "user_query": user_query,
            "memory": memory
        })
    if clarity_response.requires_reformulation:
        print("🧠 Not self-contained → Reformulating based on memory")
        
        output: ReformulatedQuery = chain_reformulation.invoke({
            "user_query": user_query,
            "memory": memory
        })
    
        print("✍️ Reformulated Query:", output.reformulated_query)

    # Step 2: Replace the latest HumanMessage with the reformulated query
        if user_msg:
            user_msg.content = output.reformulated_query
    else:
        print("✅ Query is already self-contained. No reformulation needed.")
        
    return state
