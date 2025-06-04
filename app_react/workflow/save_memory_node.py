import uuid
from langgraph.store.base import BaseStore
from langchain_core.tools import tool
from functools import partial
from typing import TypedDict, List, Optional, List, Literal, Annotated

def make_save_memory_tool(store, user_id: str):
    @tool
    def save_memory_tool(user_query: str, reformulated_query: str = "", final_response: str = "") -> str:
        """
        Saves memory to vector store for future retrieval. Stores user query, reformulated version, and final response.
        """
        print(f"💾 Saving memory for user_id: {user_id}")
        namespace = (user_id, "memories")
        memory_id = str(uuid.uuid4())

        memory = {
            "user_query": user_query,
            "reformulated_query": reformulated_query,
            "final_response": final_response,
        }

        try:
            store.put(namespace, memory_id, memory)
            print("✅ Memory saved.")
            return f"✅ Memory saved: {user_query[:60]}..."
        except Exception as e:
            return f"❌ Failed to save memory: {str(e)}"

    save_memory_tool.name = "save_memory_tool"
    return save_memory_tool


