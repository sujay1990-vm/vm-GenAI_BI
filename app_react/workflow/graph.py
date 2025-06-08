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
from reformulation import *
from handle_irrelevant_query import handle_irrelevant_query
from llm import get_llm, get_embedding_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import copy
from langgraph.graph.message import add_messages
from follow_up_questions import suggest_follow_up_questions_tool


llm = get_llm()
embeddings = get_embedding_model()


from langchain_core.prompts import ChatPromptTemplate

tool_usage_prompt = """
You are an intelligent assistant designed to help users query and analyze insurance data using tools like SQL, RAG, schema metadata, and memory.

**Strict Rules**:
1. Do NOT provide a final answer unless you have first used a tool and received its observation.
    - If a user query cannot be answered using any of the tools, respond with: "I'm only able to assist with data-related questions using available tools. Please ask relevant questions"
2. Always mention the source/ filename in the final answer when 'rag_worker_tool' is invoked. 

---
🧠 **Query Understanding & Reformulation**

If the user query is ambiguous, vague, or context-dependent, use this whenver possible to maintain the integrity of user's query:
→ Use: `query_reformulator_tool`  
Tool call format:  
`tool_choice: {"type": "tool", "name": "query_reformulator_tool"}`

If you need to split the user query, any other queries into simpler sub queries or identify sub-intents in the query:
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

To fetch recent relevant questions from user history, use this tool , ** ALWAYS USE THIS TOOL **:
→ Use: `memory_tool`  
Tool call format:  
`tool_choice: {"type": "tool", "name": "memory_tool"}`

To save the current question and final response for future reuse, **ALWAYS USE THIS TOOL**:
→ Use: `save_tool`  
Tool call format:  
`tool_choice: {"type": "tool", "name": "save_tool"}`

*Prevent Halucinations or irrelevant answers*

To handle vague, non-data-related, or off-topic questions:
→ Use: handle_irrelevant_query
Tool call format:
tool_choice: {"type": "tool", "name": "handle_irrelevant_query"}

*Follow up questions for users*

To generate helpful follow-up questions for the user:  
→ Use: memory_tool, `suggest_follow_up_questions_tool`  
Tool call format:  
`tool_choice: {"type": "tool", "name": "memory_tool"}`
`tool_choice: {"type": "tool", "name": "suggest_follow_up_questions_tool"}`

When calling the tool `suggest_follow_up_questions_tool`, provide the `chat_history` argument as a dictionary with a `turns` key.

The format example is :

{
  "turns": [
    {"role": "user", "content": "How many claims were litigated in California?"},
    {"role": "assistant", "content": "There were 12 litigated claims."},
    {"role": "user", "content": "Compare that to New York."},
    {"role": "assistant", "content": "New York had 15 litigated claims."}
  ]
}


---

Always think step-by-step and only call the tools when needed. If no tools are required, return a final answer directly.
"""

def build_graph(user_id: str, store, retriever, llm, embeddings):
    query_reformulator_tool.description = "Reformulates the user's question using prior memory if needed, making the query clearer for downstream reasoning."
    # memory_tool = make_retrieve_recent_memory_tool(store, user_id)
    # memory_tool.description = "Retrieve the top 3 relevant past memories for the user's query based on semantic similarity."
    query_analyzer_tool.description = "Analyze the user query to identify subqueries and their intent for targeted processing (e.g., turnover + staffing)."
    rag_tool = make_rag_worker_tool(retriever)
    rag_tool.description = "Retrieve relevant context from unstructured documents using semantic search (RAG). Returns top 3 relevant chunks."
    # synthesizer_tool.description = "Combine SQL results and document context into a clear natural language answer for the user query."
    get_schema_tool.description = "Load the full database schema and metric definitions from disk for use in SQL generation or metadata reasoning."
    sql_worker_tool.description = "Generate and execute SQL based on the user query, schema, and metric definitions. Returns raw result or error messages."
    # save_tool = make_save_memory_tool(store, user_id)
    save_memory_node = make_save_memory_node(store, user_id)
    
    # save_tool.description = "Store the user's query, reformulated query, and final response into memory for future reference."
    handle_irrelevant_query.description = (
    "Detects unrelated, vague, or non-data-related queries (e.g., jokes, greetings, personal questions) "
    "and returns a message explaining that this assistant only handles data-related questions using tools.")
    suggest_follow_up_questions_tool.description = (
    "Generates 3-5 helpful, concise follow-up questions for a claims adjuster based on the current query, chat history, data summary, and schema."
    "Inputs: "
    "user_input: The user's current question."
    "chat_history: A dict containing previous user questions and assistant responses relevant to the current question. Should be formatted as a short summary or transcript."
        )
    memory_node = make_retrieve_memory_node(store, user_id)
    tools = [
        get_schema_tool,
        rag_tool,
        sql_worker_tool,
        # synthesizer_tool,
        handle_irrelevant_query
    ]

    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)

    # 2. Define tool-aware LLM node
    def llm_call(state: MessagesState):
        return {
            "messages": [
                llm_with_tools.invoke(
                    [SystemMessage(content=tool_usage_prompt)] + state["messages"]
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
    agent_builder.add_node("save_memory_node", save_memory_node)
        # agent_builder.add_node(
    # "environment",
    # lambda state, config=None: {"messages": tool_node(state, config=config)["messages"]}
    #     )
    agent_builder.add_node("environment", tool_node)

    agent_builder.add_edge(START, "retrieve_memory_node")
    agent_builder.add_edge("retrieve_memory_node", "llm_call")
    agent_builder.add_edge("environment", "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, {"Action": "environment", END: "save_memory_node"})
    agent_builder.add_edge("save_memory_node", END)
    # 6. Compile and return agent
    return agent_builder.compile(checkpointer=checkpointer, store=store)
