import uuid
from langgraph.store.base import BaseStore
from langchain_core.tools import tool
from functools import partial
from typing import TypedDict, List, Optional, List, Literal, Annotated
from langgraph.graph import MessagesState

# def make_save_memory_tool(store, user_id: str):
#     @tool
#     def save_memory_tool(user_query: str, reformulated_query: str, final_response: str) -> str:
#         """
#         Saves memory to vector store for future retrieval. Stores user query, reformulated version, and final response.
#         """
#         print(f"💾 Saving memory for user_id: {user_id}")
#         namespace = (user_id, "memories")
#         memory_id = str(uuid.uuid4())

#         memory = {
#             "user_query": user_query,
#             "reformulated_query": reformulated_query,
#             "final_response": final_response,
#         }

#         try:
#             store.put(namespace, memory_id, memory)
#             print("✅ Memory saved.")
#             return f"✅ Memory saved: {user_query[:60]}..."
#         except Exception as e:
#             return f"❌ Failed to save memory: {str(e)}"

#     save_memory_tool.name = "save_memory_tool"
#     return save_memory_tool


import uuid
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState

def make_save_memory_node(store, user_id: str):
    def save_memory_node(state: MessagesState):
        print("💾 [Node] Saving memory...")

        messages = state["messages"]

        user_query = next(
            (m.content for m in reversed(messages) if m.type == "human"),
            ""
        )
        final_response = next(
            (m.content for m in reversed(messages) if m.type in {"ai", "assistant"}),
            ""
        )

        # Build and sanitize memory record
        memory_record = {
            "user_query": str(user_query or ""),
            "final_response": str(final_response or "")
        }

        # Store memory
        namespace = (user_id, "memories")
        memory_id = str(uuid.uuid4())
        store.put(namespace, memory_id, memory_record)

        return {
            "messages": [
                SystemMessage(content="✅ [Memory saved]")
            ]
        }

    return save_memory_node

