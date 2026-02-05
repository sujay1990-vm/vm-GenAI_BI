from typing import TypedDict, List, Optional, List, Literal, Annotated
from langchain_core.messages import BaseMessage, AIMessage
from pydantic import BaseModel, Field
from langchain_core.documents import Document
import operator
from langchain_core.documents import Document
from llm import get_embedding_model
from query_analyser import query_analyzer_tool
from retrieve_memory import make_retrieve_memory_node
from rag_worker import make_rag_worker_tool
from get_schema import get_schema_tool
from get_table_schema import get_schema_table_tool
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
from similar_claims import similar_claims_tool
# from dynamic_similar_claims import dynamic_similar_claims_tool
from similarity_explain import llm_similarity_explainer_tool
from litigation_risk import get_litigation_risk_score_tool

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
7. For similarity-related questions such as "Why is Claim X similar to these claims?" or "Explain what makes these claims related," use the `llm_similarity_explainer_tool`.
    - This tool accepts a list of 5 full claim records (as dictionaries) and returns a natural language explanation of key similarities and differences.
    - It must be used **after retrieving the claims via the SQL tool**.
    - Tool call format: {"type": "tool", "name": "llm_similarity_explainer_tool", "arguments": {"claim_rows": [...]}}

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

If you need to understand the structure of the data or metric definitions, you have TWO options:

**Option 1 (RECOMMENDED - Use by default):** 
Use `get_schema_table_tool` to retrieve only the top 3 most relevant table schemas based on your query.
This is more efficient and reduces token usage while still providing the necessary context.
→ Tool call format: `tool_choice: {"type": "tool", "name": "get_schema_table_tool"}`



**Default behavior:** Always try `get_schema_table_tool` first.
---

🛠️ **SQL Generation and Execution**

If you're ready to generate SQL using schema + metric definitions and/or information retrieved from the RAG tool:
→ Use: `sql_worker_tool`  
Tool call format:  
`tool_choice: {"type": "tool", "name": "sql_worker_tool"}`

**Important:** Make sure you have the necessary schema information before generating SQL. Use one of the schema tools first.

---

🔍 **Similarity Analysis**

For finding similar claims:
→ Use: `similar_claims_tool`
This tool finds the top 5 most similar claims using combined structured and textual features.

For explaining why claims are similar:
→ Use: `llm_similarity_explainer_tool` (only AFTER retrieving the claims via SQL)
This tool provides natural language explanations of claim similarities.

---

⚖️ **Litigation Risk Assessment**

For predicting litigation likelihood:
→ Use: `get_litigation_risk_score_tool`
This tool provides a risk score (0-1) and explains contributing factors.

---

📅 **Date Calculations**

For calculating days between dates:
→ Use: `calculate_date_diff`
Do NOT use SQL for date differences - always use this tool instead.

---

Think step-by-step. Only call tools when needed. Do not guess any domain-specific concepts — retrieve them explicitly.

"""

def get_claims_overview_injection(user_query: str) -> str:
    """
    Injects specific formatting instructions when user asks for claims overview.
    This ensures consistent output structure for overview requests.
    """
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

            Use calculation logic from the schema definitions returned by schema tools.
            Make necessary calculations using available data.
            """
    return ""


def build_graph(user_id: str, store, retriever, llm, embeddings):
    """
    Builds the LangGraph agent with all tools, nodes, and routing logic.
    
    Args:
        user_id: Unique identifier for the user (for memory management)
        store: Memory store for saving/retrieving conversation history
        retriever: Vector store retriever for RAG functionality
        llm: Language model instance
        embeddings: Embedding model instance
    
    Returns:
        Compiled LangGraph agent
    """
    
    # ============================================================================
    # TOOL DESCRIPTIONS - Define what each tool does for the LLM
    # ============================================================================
    
    calculate_date_diff.description = (
        "Given two dates, returns the number of days between them. "
        "Supports formats: MM/DD/YYYY and YYYY-MM-DD. "
        "Use this instead of SQL for date calculations."
    )
    
    rag_tool = make_rag_worker_tool(retriever)
    rag_tool.description = (
        "Retrieve relevant context from unstructured documents using semantic search (RAG). "
        "Returns top 3 relevant chunks with source information. "
        "Use for policy definitions, guidelines, thresholds, and domain-specific concepts."
    )
    
    get_schema_tool.description = (
        "Load the FULL database schema and metric definitions from disk. "
        "Use only when you need comprehensive schema information across ALL tables. "
        "For most queries, prefer 'get_schema_table_tool' instead for efficiency."
    )
    
    get_schema_table_tool.description = (
        "Retrieve only the top 3 most relevant table schemas based on the user's query. "
        "This is the RECOMMENDED schema tool - more efficient and focused. "
        "Use this by default unless you specifically need all schemas."
    )
    
    sql_worker_tool.description = (
        "Generate and execute SQL queries based on user query, schema, and metric definitions. "
        "Returns raw SQL results or error messages. "
        "Compatible with SQLite syntax only. "
        "Always obtain schema information first using a schema tool."
    )
    
    similar_claims_tool.description = (
        "Find the top 5 most similar claims to a given claim number. "
        "Uses combined structured and textual features for matching. "
        "Explains which columns contributed to similarity or differences, including top matching features."
    )
    
    llm_similarity_explainer_tool.description = (
        "Generates a natural language explanation for why a given set of claims are similar. "
        "Takes a list of 5 claims (as dictionaries) with selected similarity columns: "
        "'Loss cause', 'Loss Location State', 'Vehicle Make', 'Vehicle Model', 'Damage Description', "
        "'Claim Status', 'Litigation', 'Medical & Injury Documentation', 'Medical Reports', "
        "'Hospital Records', 'Third-Party Information', 'Subro Opportunity', 'Third-Party Insurance', "
        "'Third-Party Claim Form', 'Vehicle Year', 'Repair Estimate', 'Repair Bill', 'Medical bill', "
        "'Total Claim Bill', 'fault_rating', 'Time_to_Report', 'subrogation_score', 'recovery_amount', "
        "'recovery_rate', 'witness_available', 'pursuit_cost', 'recovery_gap_amount'. "
        "This tool analyzes common patterns and differences and returns a human-readable explanation. "
        "Must be used AFTER retrieving claims via SQL tool."
    )
    
    get_litigation_risk_score_tool.description = (
        "Predicts the likelihood of litigation for a given claim ID using a logistic regression model. "
        "Returns a risk score between 0 and 1 (higher = more likely to result in litigation). "
        "Identifies top positive and negative contributing features affecting the prediction. "
        "Generates a natural language explanation interpreting the result in litigation risk context."
    )
    
    handle_irrelevant_query.description = (
        "Detects unrelated, vague, or non-data-related queries. "
        "Examples: jokes, greetings, personal questions, off-topic requests. "
        "Returns a polite message explaining this assistant only handles data-related questions using tools."
    )
    
    # ============================================================================
    # TOOLS LIST - All available tools for the LLM to use
    # ============================================================================
    
    memory_node = make_retrieve_memory_node(store, user_id)
    save_memory_node = make_save_memory_node(store, user_id)
    
    tools = [
        # get_schema_tool,                    # Full schema (fallback)
        get_schema_table_tool,              # Smart schema (recommended)
        rag_tool,                           # Document retrieval
        sql_worker_tool,                    # SQL generation & execution
        handle_irrelevant_query,            # Off-topic detection
        calculate_date_diff,                # Date calculations
        similar_claims_tool,                # Claim similarity search
        llm_similarity_explainer_tool,      # Explain claim similarities
        get_litigation_risk_score_tool      # Litigation risk prediction
    ]

    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)

    # ============================================================================
    # LLM CALL NODE - Main reasoning node that decides which tools to use
    # ============================================================================
    
    def llm_call(state: MessagesState):
        """
        LLM decides whether to call a tool or provide final response.
        
        This node:
        1. Retrieves conversation memory if available
        2. Checks for special formatting needs (e.g., claims overview)
        3. Constructs the full prompt with tool usage instructions
        4. Invokes the LLM with tool-calling capabilities
        
        Returns:
            Updated state with LLM's response (either tool calls or final answer)
        """
        
        memory_messages = []

        # Rebuild memory from retrieved_memory (if exists)
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

        # Only inject overview format prompt if this exact query is a "claims overview" request,
        # and if it's not already in the current state messages
        format_injection = ""
        if "claims overview" in user_query.lower():
            already_injected = any(
                m.type == "system" and "structure the output like this" in getattr(m, "content", "")
                for m in state["messages"]
            )
            if not already_injected:
                format_injection = get_claims_overview_injection(user_query)

        # Final prompt = base tool usage instructions + any special formatting
        injected_prompt = tool_usage_prompt + format_injection

        return {
            "messages": [
                llm_with_tools.invoke(
                    [SystemMessage(content=injected_prompt)] + memory_messages + state["messages"]
                )
            ]
        }

    # ============================================================================
    # TOOL EXECUTION NODE - Executes the tools that LLM decides to call
    # ============================================================================
    
    def tool_node(state: MessagesState):
        """
        Executes tool calls requested by the LLM.
        
        Processes each tool call from the last LLM message:
        1. Extracts tool name and arguments
        2. Invokes the corresponding tool
        3. Wraps results in ToolMessage for LLM to process
        
        Returns:
            Updated state with tool execution results
        """
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}

    # ============================================================================
    # ROUTING LOGIC - Decides whether to continue with tools or end
    # ============================================================================
    
    def should_continue(state: MessagesState) -> Literal["Action", END]:
        """
        Determines if the agent should continue calling tools or finish.
        
        Logic:
        - If last message has tool_calls → route to "Action" (tool execution)
        - If last message has no tool_calls → route to END (final response ready)
        
        Returns:
            "Action" to execute tools, or END to finish the conversation turn
        """
        last_message = state["messages"][-1]
        return "Action" if last_message.tool_calls else END

    # ============================================================================
    # GRAPH CONSTRUCTION - Build the LangGraph workflow
    # ============================================================================
    
    agent_builder = StateGraph(MessagesState)
    checkpointer = InMemorySaver()

    # Add all nodes to the graph
    agent_builder.add_node("llm_call", llm_call)                        # Main reasoning
    agent_builder.add_node("retrieve_memory_node", memory_node)         # Load conversation history
    agent_builder.add_node("query_reformulator", query_reformulator_node)  # Reformulate query if needed
    agent_builder.add_node("save_memory_node", save_memory_node)        # Save conversation to memory
    agent_builder.add_node("follow_up_node", make_follow_up_node())     # Generate follow-up questions
    agent_builder.add_node("environment", tool_node)                    # Execute tools

    # Define the flow: START → memory → reformulate → llm → tools → llm → save → follow-up → END
    agent_builder.add_edge(START, "retrieve_memory_node")
    agent_builder.add_edge("retrieve_memory_node", "query_reformulator")
    agent_builder.add_edge("query_reformulator", "llm_call")
    agent_builder.add_edge("environment", "llm_call")  # After tool execution, go back to LLM
    
    # Conditional edge: LLM decides to call tools or finish
    agent_builder.add_conditional_edges(
        "llm_call", 
        should_continue, 
        {
            "Action": "environment",      # Call tools
            END: "save_memory_node"       # Finish and save memory
        }
    )
    
    agent_builder.add_edge("save_memory_node", "follow_up_node")
    agent_builder.add_edge("follow_up_node", END)

    # ============================================================================
    # COMPILE AND RETURN - Create the final executable agent
    # ============================================================================
    
    return agent_builder.compile(checkpointer=checkpointer, store=store)