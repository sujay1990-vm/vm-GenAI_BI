from .llm import get_llm, get_embedding_model
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from .nl_response import nl_response_node
from .visualization import identify_visualization_goals_from_state, visualization_generation_node
from .report import report_generation_node
import pandas as pd
from .domain_detection import metric_resolution_node
from .metadata_loader import metadata_loader_node
from .save_memory import save_memory_node
from .sql_generator import sql_generation_node
from .sql_validator import sql_query_validation_node
from .final_output import final_output_node
from .sql_executor import execute_query_node
from .error_handling import handle_metric_resolution, handle_query_checks, handle_execution_result


llm = get_llm()
embeddings = get_embedding_model()

# --- Define Data Models and Workflow Nodes (as in your notebook code) ---
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END

class GraphState(TypedDict):
    user_query: str 
    resolved_metrics: List[Dict[str, Any]]
    report: bool
    visualize: bool
    visual_instructions: str
    metadata: Dict[str, Dict[str, str]]
    sql_queries: List[str]
    sql_result_str: str
    sql_result_df: List[pd.DataFrame]
    sql_results: List[Dict[str, Any]]
    execution_error: bool
    error_history: List[str]
    nl_response: str
    csv_files: Dict[str, bytes]
    report_response: str
    report_generation_error: bool
    final_response: Any
    goals: List[str]
    generated_visualization_code: str
    visualization_files: List[bytes]
    visualization_output: str
    visualization_response: str
    sql_cache_key: Optional[str]
    query_check_flag: bool  # True if invalid SQL pattern was detected
    query_check_msg: str   # Message explaining the validation issue
    fact_join_violation : bool
    fact_join_msg: str
    cte_flag : bool
    cte_disclaimer : str
    filters: Optional[Dict[str, Any]]
    hard_exit: Optional[bool]  # True if the workflow should terminate early
    exit_reason: Optional[str]  # Reason for termination if hard_exit is triggered
    flow_exit_flag: Optional[bool]
    exit_reason: Optional[str]
    memory: List[Dict[str, Any]]  # ✅ New field to hold memory history
    access_token: Optional[str]  # ✅ Add this line


# Create memory-based checkpointing
checkpointer = InMemorySaver()

# Define memory store with embedding support
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,  # text-embedding-3-small = 1536 dimensions
        "fields": ["user_query", "sql_queries", "nl_response"]  # fields to embed/search
    }
)

# --- Assemble the Workflow using your StateGraph ---
workflow = StateGraph(GraphState)

workflow.add_node("MetricResolver", metric_resolution_node)
# workflow.add_node("UserQueryConfirmation", user_query_confirmation_node)
workflow.add_node("MetadataLoader", metadata_loader_node)
workflow.add_node("SQL_generator", sql_generation_node)
workflow.add_node("sql_validator", sql_query_validation_node)
workflow.add_node("sql_executor", execute_query_node)
workflow.add_node("llm_response", nl_response_node)
workflow.add_node("visualization_generation", visualization_generation_node)
workflow.add_node("report_generation", report_generation_node)
workflow.add_node("identify_visualization_goals", identify_visualization_goals_from_state)
workflow.add_node("final_output", final_output_node)
workflow.add_node("save_memory", save_memory_node)


# Define edges:
workflow.add_edge(START, "MetricResolver")
workflow.add_conditional_edges("MetricResolver", handle_metric_resolution)
# workflow.add_conditional_edges(
#     "MetricResolver",  # or wherever reformulation happens
#     lambda state: "UserQueryConfirmation" if state.get("reformulated_query") else "MetadataLoader"
# )
# workflow.add_edge("UserQueryConfirmation", "MetadataLoader")
workflow.add_edge("MetadataLoader", "SQL_generator")
workflow.add_edge("SQL_generator", "sql_validator")
workflow.add_conditional_edges("sql_validator", handle_query_checks)
workflow.add_conditional_edges("sql_executor", handle_execution_result)
workflow.add_edge("final_output", "save_memory")
workflow.add_edge("save_memory", END)
app = workflow.compile(checkpointer=checkpointer, store=store)