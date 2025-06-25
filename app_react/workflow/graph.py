from typing import TypedDict, List, Optional, List, Literal, Annotated
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from langchain_core.documents import Document
import operator
from langchain_core.documents import Document
from llm import get_embedding_model
from query_analyser import query_analyzer_tool
from retrieve_memory import make_retrieve_memory_node
from rag_worker import make_rag_worker_tool
from get_schema import get_schema_tool
from sql_worker import sql_worker_tool
from save_memory_node import make_save_memory_node
from synthesizer import synthesizer_tool
from handle_irrelevant_query import handle_irrelevant_query
from llm import get_llm, get_embedding_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import copy
from langgraph.graph.message import add_messages
from follow_up_questions import make_follow_up_node
from reformulation import query_reformulator_node
from date_diff_tool import calculate_date_diff
from datetime import datetime

llm = get_llm()
embeddings = get_embedding_model()


from langchain_core.prompts import ChatPromptTemplate

tool_usage_prompt = """

You are a reasoning-first AI assistant that answers questions about insurance data and policy guidelines. 
You have access to tools for SQL, RAG, and schema/metric metadata. Your job is to decide which tools are needed — and only respond with a final answer once you've used the necessary tools and received their outputs.

You must not assume definitions, thresholds, or policy logic — always retrieve such information using the appropriate tool before proceeding.

**Strict Rules**:
1. Never respond with a final answer until you have invoked the appropriate tool(s) and received their outputs.
    - If a user query cannot be answered using any of the tools, respond with: "I'm only able to assist with data-related questions using available tools. Please ask relevant questions"
2. If you use `rag_worker_tool`, the final answer must include the source filename(s) from the retrieved documents in bulleted format.
3. If a SQL query depends on a concept, threshold, or definition that is not directly present in the structured data, you **must first use** `rag_worker_tool` to retrieve the exact value or definition before attempting SQL.
4. To prevent hallucinations or irrelevant answers, always use `handle_irrelevant_query` for vague, off-topic, or non-data-related questions.
    - Tool call format: {"type": "tool", "name": "handle_irrelevant_query"}
5. Use `calculate_date_diff` exclusively for calculating the number of days between two specific dates.
    - This tool supports formats like `YYYY-MM-DD` and `MM/DD/YYYY`.
    - Tool call format: {"type": "tool", "name": "calculate_date_diff", "arguments": {"start_date": "...", "end_date": "..."}}
6. Generate SQL that is compatible with SQLite.
- **DO NOT** generate SQL to calc date diff, use tool "calculate_date_diff"

---

📚 **Retrieval (Unstructured Context)**

If the query refers to clinical definitions, policy language, thresholds, concepts, or guidelines not directly in SQL:
→ Use: `rag_worker_tool`  
Tool call format:  
`tool_choice: {"type": "tool", "name": "rag_worker_tool"}`

Examples of concepts that must be looked up via RAG:  
"soft threshold", "total loss criteria", "eligibility rule", "high severity claim", "policy language", etc.

---

🗂️ **Schema and Metric Metadata**

If you need to understand the structure of the data or metric definitions:
→ Use: `get_schema_tool`  
Tool call format:  
`tool_choice: {"type": "tool", "name": "get_schema_tool"}`

---

🛠️ **SQL Generation and Execution**

If you're ready to generate SQL using schema + metric definitions and/or information retrieved from the RAG tool:
→ Use: `sql_worker_tool`  
Tool call format:  
`tool_choice: {"type": "tool", "name": "sql_worker_tool"}`

---

Think step-by-step. Only call tools when needed. Do not guess any domain-specific concepts — retrieve them explicitly.

"""

def get_claims_overview_injection(user_query: str) -> str:
    if "claims overview" in user_query.lower():
        return """
            If the user asks for a "claims overview", structure the output like this:

            - Policy Number
            - Claim Number
            - Date of Loss
            - Claim Type [PD, BI, Total Loss, Cargo, Subro]
            - Claim Status
            - Current Claim Phase [e.g., investigation, negotiation, litigation]
            - Total Incurred / Paid / Reserved
            - Days Open - The number of days between the date the loss was reported and when the claim was paid
            - Days to Close - Time from FNOL to when the last payment (medical or repair) was made

            Use calculation logic from the schema definitions returned by `get_schema_tool`.
            Make necessary calculations
            """
    return ""


def build_graph(user_id: str, store, retriever, llm, embeddings):
    # query_reformulator_tool.description = "Reformulates the user's question using prior memory if needed, making the query clearer for downstream reasoning."
    # memory_tool = make_retrieve_recent_memory_tool(store, user_id)
    # memory_tool.description = "Retrieve the top 3 relevant past memories for the user's query based on semantic similarity."
    calculate_date_diff.description = "Given two dates, returns the number of days between them (supports MM/DD/YYYY and YYYY-MM-DD formats)."
    rag_tool = make_rag_worker_tool(retriever)
    rag_tool.description = "Retrieve relevant context from unstructured documents using semantic search (RAG). Returns top 3 relevant chunks."
    # synthesizer_tool.description = "Combine SQL results and document context into a clear natural language answer for the user query."
    get_schema_tool.description = "Load the full database schema and metric definitions from disk for use in SQL generation or metadata reasoning."
    sql_worker_tool.description = "Generate and execute SQL based on the user query, schema, and metric definitions. Returns raw result or error messages."
    # save_tool = make_save_memory_tool(store, user_id)
    
    
    # save_tool.description = "Store the user's query, reformulated query, and final response into memory for future reference."
    handle_irrelevant_query.description = (
    "Detects unrelated, vague, or non-data-related queries (e.g., jokes, greetings, personal questions) "
    "and returns a message explaining that this assistant only handles data-related questions using tools.")
    memory_node = make_retrieve_memory_node(store, user_id)
    save_memory_node = make_save_memory_node(store, user_id)
    tools = [
        get_schema_tool,
        rag_tool,
        sql_worker_tool,
        # synthesizer_tool,
        handle_irrelevant_query,
        calculate_date_diff
    ]

    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)

    # 2. Define tool-aware LLM node
    def llm_call(state: MessagesState):
        """LLM decides whether to call a tool or just reply, using Anthropic-style prompt."""

        memory_messages = []

        # Rebuild memory as Human/AI turns
        if "retrieved_memory" in state and state["retrieved_memory"]:
            memory_lines = state["retrieved_memory"].split("\n")
            for i in range(0, len(memory_lines), 3):
                if i + 1 < len(memory_lines):
                    user_line = memory_lines[i]
                    response_line = memory_lines[i + 1]
                    if user_line.startswith("- User:") and response_line.startswith("- Final Response:"):
                        user_msg = user_line.replace("- User:", "").strip()
                        assistant_msg = response_line.replace("- Final Response:", "").strip()
                        memory_messages.append(HumanMessage(content=user_msg))
                        memory_messages.append(AIMessage(content=assistant_msg))

        # Extract current user query
        user_msg = next((m for m in reversed(state["messages"]) if m.type == "human"), None)
        user_query = user_msg.content if user_msg else ""

        # Prompt injection only for claims overview
        injected_prompt = tool_usage_prompt + get_claims_overview_injection(user_query)

        return {
            "messages": [
                llm_with_tools.invoke(
                    [SystemMessage(content=injected_prompt)] + memory_messages + state["messages"]
                )
            ]
        }


    # 3. Tool execution node
    def tool_node(state: MessagesState):
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}


    # 4. Conditional routing
    def should_continue(state: MessagesState) -> Literal["Action", END]:
        last_message = state["messages"][-1]
        return "Action" if last_message.tool_calls else END

    # 5. Build LangGraph
    agent_builder = StateGraph(MessagesState)
    checkpointer = InMemorySaver()

    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("retrieve_memory_node", memory_node)
    agent_builder.add_node("query_reformulator", query_reformulator_node)
    agent_builder.add_node("save_memory_node", save_memory_node)
    agent_builder.add_node("follow_up_node", make_follow_up_node())

        # agent_builder.add_node(
    # "environment",
    # lambda state, config=None: {"messages": tool_node(state, config=config)["messages"]}
    #     )
    agent_builder.add_node("environment", tool_node)

    agent_builder.add_edge(START, "retrieve_memory_node")
    agent_builder.add_edge("retrieve_memory_node", "query_reformulator")
    agent_builder.add_edge("query_reformulator", "llm_call")
    # agent_builder.add_edge("retrieve_memory_node", "llm_call")
    agent_builder.add_edge("environment", "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, {"Action": "environment", END: "save_memory_node"})
    agent_builder.add_edge("save_memory_node", "follow_up_node")
    agent_builder.add_edge("follow_up_node", END)

    # 6. Compile and return agent
    return agent_builder.compile(checkpointer=checkpointer, store=store)
