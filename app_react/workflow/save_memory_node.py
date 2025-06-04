import uuid
from langgraph.store.base import BaseStore
from langchain_core.tools import tool
from functools import partial
from typing import TypedDict, List, Optional, List, Literal, Annotated

def make_save_memory_tool(store):
    @tool
    def save_memory_tool(**kwargs) -> str:
        """
        Saves memory to vector store for future retrieval. Stores user query, reformulated version, and final response.
        """
        config = kwargs.get("config")
        user_query = kwargs.get("user_query", "")
        reformulated_query = kwargs.get("reformulated_query", "")
        final_response = kwargs.get("final_response", "")

        print("🧠 Tool received config:", config)

        if not config or "configurable" not in config:
            return "❌ Missing config or user_id"

        user_id = config["configurable"]["user_id"]
        print("💾 Saving memory for user_id:", user_id)

        namespace = (user_id, "memories")
        memory_id = str(uuid.uuid4())

        memory = {
            "user_query": user_query,
            "reformulated_query": reformulated_query,
            "final_response": final_response
        }

        try:
            store.put(namespace, memory_id, memory)
            print("✅ Memory saved.")
            return "✅ Memory saved successfully."
        except Exception as e:
            return f"❌ Failed to save memory: {str(e)}"

    return save_memory_tool

