from typing import TypedDict, List, Optional, List, Literal, Annotated
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from langchain_core.documents import Document
import operator
from langchain_core.documents import Document
from llm import get_embedding_model
from query_analyser import query_analyzer_tool
from retrieve_memory import make_retrieve_recent_memory_tool
from rag_worker import make_rag_worker_tool
from get_schema import get_schema_tool
from sql_worker import sql_worker_tool
from save_memory_node import make_save_memory_tool
from synthesizer import synthesizer_tool
from reformulation import query_reformulator_tool
from llm import get_llm, get_embedding_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import copy
from langgraph.graph.message import add_messages


llm = get_llm()
embeddings = get_embedding_model()


from langchain_core.prompts import ChatPromptTemplate

tool_usage_prompt = """
You are an intelligent assistant designed to help users query and analyze insurance data using tools like SQL, RAG, schema metadata, and memory.

🚫 You MUST NOT answer personal, casual, or general questions (e.g., jokes, stories, trivia).
✅ If the user's query is unrelated to the data or tools, respond with:
"I'm here to help with data-related questions only. Please ask something related to the dataset or guidelines."

Use the below tools as needed to answer the user question as accurately and precisely as possible. 
Use the tools when needed. Follow this reasoning pattern:
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of tools below
Action Input: the input to the action
Observation: the result of the action
When you have a response to say to the Human, or if you do not need to use a tool, respond using this format:
Thought: Do I need to use a tool? No
Final Answer: [your response here]
Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
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

To fetch recent relevant questions from user history, use this tool , ** ALWAYS USE THIS TOOL  **:
→ Use: `retrieve_recent_memory`  
Tool call format:  
`tool_choice: {"type": "tool", "name": "retrieve_recent_memory"}`

To save the current question and final response for future reuse, **ALWAYS USE THIS TOOL**:
→ Use: `save_memory_tool`  
Tool call format:  
`tool_choice: {"type": "tool", "name": "save_memory_tool"}`

---

Always think step-by-step and only call the tools when needed. If no tools are required, return a final answer directly.
"""

def build_graph(user_id: str, store, retriever, llm, embeddings):
    query_reformulator_tool.description = "Reformulates the user's question using prior memory if needed, making the query clearer for downstream reasoning."
    memory_tool = make_retrieve_recent_memory_tool(store, user_id)
    memory_tool.name = "retrieve_recent_memory" 
    memory_tool.description = "Retrieve the top 3 relevant past memories for the user's query based on semantic similarity."
    query_analyzer_tool.description = "Analyze the user query to identify subqueries and their intent for targeted processing (e.g., turnover + staffing)."
    rag_tool = make_rag_worker_tool(retriever)
    rag_tool.description = "Retrieve relevant context from unstructured documents using semantic search (RAG). Returns top 3 relevant chunks."
    # synthesizer_tool.description = "Combine SQL results and document context into a clear natural language answer for the user query."
    get_schema_tool.description = "Load the full database schema and metric definitions from disk for use in SQL generation or metadata reasoning."
    sql_worker_tool.description = "Generate and execute SQL based on the user query, schema, and metric definitions. Returns raw result or error messages."
    save_tool = make_save_memory_tool(store, user_id)
    save_tool.name = "save_memory_tool"
    save_tool.description = "Store the user's query, reformulated query, and final response into memory for future reference."
    tools = [
        query_reformulator_tool,
        query_analyzer_tool,
        get_schema_tool,
        rag_tool,
        sql_worker_tool,
        # synthesizer_tool,
        memory_tool,
        save_tool,
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
        # agent_builder.add_node(
    # "environment",
    # lambda state, config=None: {"messages": tool_node(state, config=config)["messages"]}
    #     )
    agent_builder.add_node("environment", tool_node)

    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_edge("environment", "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, {"Action": "environment", END: END})

    # 6. Compile and return agent
    return agent_builder.compile(checkpointer=checkpointer, store=store)
