from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from typing import Annotated, TypedDict
import operator
from langchain_core.documents import Document
from llm import get_embedding_model
from assign_workers import assign_workers
from query_analyser import query_analysis_node
from retrieve_memory import retrieve_recent_memory_node
from rag_worker import rag_worker
from sql_worker import sql_worker
from save_memory_node import save_memory_node
from synthesizer import synthesizer
from reformulation import query_clarity_and_reformulation_node


class GraphState(TypedDict, total=False):
    user_query: str
    reformulated_query: str
    reformulation_required: bool
    subqueries: list  # includes SQL and RAG subqueries
    rag_outputs: Annotated[list[Document], operator.add]  # ✅ RAG chunks accumulated
    sql_outputs: Annotated[list[str], operator.add]       # ✅ SQL results accumulated
    final_response: str
    memory: list


class RAGWorkerState(TypedDict):
    query: str
    rag_outputs: List[Document] 


class SQLWorkerState(TypedDict):
    query: str
    sql_outputs: list[str]

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore


rag_graph = StateGraph(GraphState)

checkpointer = InMemorySaver()

embeddings = get_embedding_model()

store = InMemoryStore(index={"embed": embeddings, "dims": 1536})
rag_graph.add_node("retrieve_memory", retrieve_recent_memory_node)
rag_graph.add_node("query_clarity_and_reformulation_node", query_clarity_and_reformulation_node)
rag_graph.add_node("query_analysis_node", query_analysis_node)
rag_graph.add_node("rag_worker", rag_worker)
rag_graph.add_node("sql_worker", sql_worker)
rag_graph.add_node("synthesizer", synthesizer)
rag_graph.add_node("save_memory_node", save_memory_node)

# Edges
rag_graph.add_edge(START, "retrieve_memory")
rag_graph.add_edge("retrieve_memory", "query_clarity_and_reformulation_node")
rag_graph.add_edge("query_clarity_and_reformulation_node", "query_analysis_node")
# Conditional branching to RAG or SQL workers
rag_graph.add_conditional_edges("query_analysis_node", assign_workers, ["rag_worker", "sql_worker"])
rag_graph.add_edge("rag_worker", "synthesizer")
rag_graph.add_edge("sql_worker", "synthesizer")
rag_graph.add_edge("synthesizer", "save_memory_node")
rag_graph.add_edge("save_memory_node", END)


rag_app = rag_graph.compile(checkpointer=checkpointer, store=store)
