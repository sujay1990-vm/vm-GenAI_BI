from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from typing import Annotated, TypedDict
import operator
from langchain_core.documents import Document
from llm import get_embedding_model
from query_analyser import query_analyzer_tool
from retrieve_memory import make_retrieve_recent_memory_tool
from rag_worker import make_rag_worker_tool
from sql_worker import sql_worker_tool
from save_memory_node import make_save_memory_tool
from synthesizer import synthesizer_tool
from reformulation import query_reformulator_tool
from llm import get_llm, get_embedding_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, START, END

llm = get_llm()
embeddings = get_embedding_model()

store = InMemoryStore(index={"embed": embeddings, "dims": 1536})
query_reformulator_tool.description = "Reformulates the user's question using prior memory if needed, making the query clearer for downstream reasoning."
memory_tool = make_retrieve_recent_memory_tool(store, user_id)
memory_tool.description = "Retrieve the top 3 relevant past memories for the user's query based on semantic similarity."
query_analyzer_tool.description = "Analyze the user query to identify subqueries and their intent for targeted processing (e.g., turnover + staffing)."
rag_tool = make_rag_worker_tool(retriever)
rag_tool.description = "Retrieve relevant context from unstructured documents using semantic search (RAG). Returns top 3 relevant chunks."
synthesizer_tool.description = "Combine SQL results and document context into a clear natural language answer for the user query."
get_schema_tool.description = "Load the full database schema and metric definitions from disk for use in SQL generation or metadata reasoning."
sql_worker_tool.description = "Generate and execute SQL based on the user query, schema, and metric definitions. Returns raw result or error messages."
save_tool = make_save_memory_tool(store, user_id)
save_tool.description = "Store the user's query, reformulated query, and final response into memory for future reference."
tools = [
    query_reformulator_tool,
    query_analyzer_tool,
    get_schema_tool,
    make_rag_worker_tool(retriever),
    sql_worker_tool,
    synthesizer_tool,
    make_retrieve_recent_memory_tool(store, user_id),
    make_save_memory_tool(store, user_id),
]

tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

tool_usage_prompt = """
You are an intelligent assistant designed to help users query and analyze insurance data using tools like SQL, RAG, schema metadata, and memory.

Use the tools when needed. Follow this reasoning pattern:

---
🧠 **Query Understanding & Reformulation**

If the user query is ambiguous, vague, or context-dependent:
→ Use: `query_reformulator_tool`  
Tool call format:  
`tool_choice: {"type": "tool", "name": "query_reformulator_tool"}`

If you need to split or identify sub-intents in the query:
→ Use: `query_analyzer_tool`  
Tool call format:  
`tool_choice: {"type": "tool", "name": "query_analyzer_tool"}`

---

📚 **Retrieval (Unstructured Context)**

If the query refers to clinical definitions, policy language, or concepts not directly in SQL:
→ Use: `rag_worker_tool`  
Tool call format:  
`tool_choice: {"type": "tool", "name": "rag_worker_tool"}`

---

🗂️ **Schema and Metric Metadata**

If you need to understand the structure of the data or metric definitions:
→ Use: `get_schema_tool`  
Tool call format:  
`tool_choice: {"type": "tool", "name": "get_schema_tool"}`

---

🛠️ **SQL Generation and Execution**

If you're ready to generate SQL using schema + metric definitions:
→ Use: `sql_worker_tool`  
Tool call format:  
`tool_choice: {"type": "tool", "name": "sql_worker_tool"}`

---

🧠 **Memory Tools**

To fetch recent relevant questions from user history:
→ Use: `retrieve_recent_memory_tool`  
Tool call format:  
`tool_choice: {"type": "tool", "name": "retrieve_recent_memory_tool"}`

To save the current question and final response for future reuse:
→ Use: `save_memory_tool`  
Tool call format:  
`tool_choice: {"type": "tool", "name": "save_memory_tool"}`

---

🧪 **Final Answer Synthesis**

To combine SQL + RAG outputs and produce a coherent response:
→ Use: `synthesizer_tool`  
Tool call format:  
`tool_choice: {"type": "tool", "name": "synthesizer_tool"}`

---

Always think step-by-step and only call the tools when needed. If no tools are required, return a final answer directly.
"""


def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or just reply, using Anthropic-style prompt."""
    return {
        "messages": [
            llm_with_tools.invoke(
                [SystemMessage(content=tool_usage_prompt)] + state["messages"]
            )
        ]
    }

def tool_node(state: MessagesState):
    """Executes the tool requested by the LLM."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

from langgraph.graph import END
from typing import Literal

def should_continue(state: MessagesState) -> Literal["Action", END]:
    """Decides whether LLM made a tool call."""
    last_message = state["messages"][-1]
    return "Action" if last_message.tool_calls else END





agent_builder = StateGraph(MessagesState)
checkpointer = InMemorySaver()
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_edge("environment", "llm_call")

agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "Action": "environment",
        END: END,
    },
)

agent = agent_builder.compile(checkpointer=checkpointer, store=store)
