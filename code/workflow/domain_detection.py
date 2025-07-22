import pandas as pd
import os
import sys
from pydantic import BaseModel, Field
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from typing import List, Dict, Any, Union, Optional, Literal

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

    
from prompts.kpi_metrics import kpi_lookup
from prompts.common_prompts import targetted_domain_prompt, reformulation_prompt, metric_resolver_prompt, reformulation_decision_prompt
from langgraph.store.base import BaseStore
import json
from typing import get_args
import re
from .llm import get_llm, get_embedding_model

llm = get_llm()
embeddings = get_embedding_model()


class ResolvedMetric(BaseModel):
    """
    A single business metric or jargon item resolved from the user query.
    """
    metric: str = Field(description="Metric or business jargon identified in the user query.")
    definition: str = Field(description="Plain English definition of the metric.")
    formula: str = Field(description="How the metric would be calculated using schema logic.")
    domain_name: Literal["Census", "Quality"] = Field(
        description="Name of the domain this metric belongs to. Must be one of: Census, Quality"
    )
    confidence: float = Field(description="Confidence between 0 and 1 indicating how relevant this domain is.")

class MetricResolutionOutput(BaseModel):
    resolved_metrics: List[ResolvedMetric] = Field(
        description="List of all metrics resolved from the user query."
    )
    report: bool = Field(default=False,
        description="True if the user explicitly requests a report."
    )
    visualize: bool = Field(default=False,
        description="True if the user explicitly requests a visualization."
    )
    visual_instructions: str = Field(
        default="",
        description="Any specific instructions provided by the user for visualization."
    )


class ReformulatedQuery(BaseModel):
    """
    Reformulated user query that is self-contained and does not require chat history for understanding.
    """
    reformulated_query: str = Field(
        description="A rewritten version of the user's question that includes necessary context from prior conversation."
    )

class ReformulationDecision(BaseModel):
    """
    Determines whether the query should be reformulated based on chat history context.
    """
    requires_reformulation: bool = Field(
        description="True if the user query depends on chat history or is ambiguous. False if it is already clear and self-contained."
    )

# domain_list_str = "\n".join(f"- {domain}" for domain in domain_list)

metric_resolution_prompt = ChatPromptTemplate.from_messages([
    ("system", metric_resolver_prompt ),
    ("human", 
     """
User query: {user_query}

Domain information:
{targetted_domain_prompt}

Available KPI dictionary:
{kpi_metrics}
""")
])

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", reformulation_prompt.strip()),
    ("human", 
     "Chat History:\n{memory}\n\nCurrent User Question:\n{user_query}\n\nRewritten Question:")
        ])

# Prompt setup
clarity_prompt = ChatPromptTemplate.from_messages([
    ("system", reformulation_decision_prompt.strip()),
    ("human",
     "Chat History:\n{memory}\n\nCurrent User Question:\n{user_query}\n\nDoes this query require reformulation?")
])

formatted_kpis = "\n".join(f"- {k}: {v}" for k, v in kpi_lookup.items())

structured_llm_metric_resolution = llm.with_structured_output(
    schema=MetricResolutionOutput,
    method="function_calling"
)

structured_llm_reformulation_resolution = llm.with_structured_output(
    schema=ReformulatedQuery,
    method="function_calling"
)

structured_clarity_resolution = llm.with_structured_output(
    schema=ReformulationDecision,
    method="function_calling"
)

def sanitize_label(label: str) -> str:
    # Replace any character that is not alphanumeric, hyphen or underscore
    return re.sub(r'[^0-9A-Za-z_-]', '_', label)

# Chain together prompt and structured LLM
chain = metric_resolution_prompt | structured_llm_metric_resolution
chain_reformulation = contextualize_prompt | structured_llm_reformulation_resolution
chain_query_clarity_check = clarity_prompt | structured_clarity_resolution

def metric_resolution_node(state: dict, config: dict, *, store: BaseStore) -> dict:
    """Resolves business metrics with domain classification and confidence scoring."""
    print("---METRIC RESOLVER NODE---")
    print(f"User Query: {state['user_query']}")
    state.update({
    "flow_exit_flag": False,
    "hard_exit": False,
    "query_check_flag": False,
    "fact_join_violation": False,
    "cte_flag": False,
    "execution_error": False,
    "error_history": [],
    "report_generation_error": False
        })

    user_id = config["configurable"]["user_id"]
    print(f"User ID: {user_id}")
    thread_id = config["configurable"]["thread_id"]
    print(f"Thread ID: {thread_id}")
    user_label= sanitize_label(config["configurable"]["user_id"])
    namespace = (user_label, "memories")

    # Pull memory from vector store
    recent_memories = store.search(
        namespace,
        query=state["user_query"],
        limit=3
    )

    if recent_memories:
        memory_snippets = [
            {
                "user_query": m.value.get("user_query")
                # "final_response": m.value.get("final_response")
                # "sql_queries": m.value.get("sql_queries", [])
            }
            for m in recent_memories
        ]

        clarity_response: ReformulationDecision = chain_query_clarity_check.invoke({
            "user_query": state["user_query"],
            "memory": json.dumps(memory_snippets, indent=2)
        })

        if clarity_response.requires_reformulation:
            print("🧠 Not self-contained → Reformulating based on memory")

            reformulation_response = chain_reformulation.invoke({
                "user_query": state["user_query"],
                "memory": json.dumps(memory_snippets, indent=2)
            })
            reformulated = reformulation_response.reformulated_query.strip()
            state["user_query"] = reformulated
            print(f"🔁 Reformulated Query: {reformulated}")
        else:
            print("✅ Query is self-contained. Reformulation skipped.")


    response: MetricResolutionOutput = chain.invoke({
        "user_query": state["user_query"],
        "targetted_domain_prompt": targetted_domain_prompt,
        "kpi_metrics" : formatted_kpis
    })

    # Convert Pydantic objects to dictionaries
    state["resolved_metrics"] = [metric.model_dump() for metric in response.resolved_metrics]
    state["report"] = response.report
    state["visualize"] = response.visualize
    state["visual_instructions"] = response.visual_instructions
    # state["filters"] = response.filters.model_dump() if response.filters else {}

    # 🚨 Domain check — ensure at least one known domain is found
    # Dynamically extract allowed domains from ResolvedMetric.domain_name
    known_domains = [d.lower() for d in get_args(ResolvedMetric.__annotations__['domain_name'])]

    has_valid_domain = any(
        metric.domain_name.lower() in known_domains
        for metric in response.resolved_metrics
    )
    # known_domains = ["HR", "Census", "Quality", "PPD", "Turnover"]
    # has_valid_domain = any(
    #     metric.domain_name.lower() in [d.lower() for d in known_domains]
    #     for metric in response.resolved_metrics
    # )

    if not has_valid_domain:
        print("🛑 No valid domain detected — triggering clarification response.")
        state["exit_reason"] = "No domain detected"
        state["flow_exit_flag"] = True  # Signal exit to graph

        # ✅ LLM clarification prompt
        clarification_prompt = f"""
            The user submitted a question, but the metric resolution step could not identify a valid business domain from the known list {known_domains}

            Below is the information:
            - User query: "{state['user_query']}"
            - Resolved metrics (if any): {state['resolved_metrics']}

            As a direct and helpful assistant:
            1. Inform the user that their question is too vague or missing key context — do not try to guess or make up an answer.
            2. Clearly explain what a good question typically includes:
            - A measurable **metric** (e.g., 'turnover rate', 'daily census')
            - A clear **time frame** (e.g., 'March 2023', 'Q1 2024')
            - Optionally, a **location** (e.g., 'Somerset')
            3. Keep the tone direct, efficient, and constructive — no fluff or over-politeness.
            """

        clarification = llm.invoke(clarification_prompt).content.strip()
        state["final_response"] = clarification
        return state

    print("✅ Domain detected — continuing flow.")
    return state