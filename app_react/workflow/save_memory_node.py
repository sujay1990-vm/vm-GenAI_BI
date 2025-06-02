import uuid
from langgraph.store.base import BaseStore
from langchain_core.tools import tool
from functools import partial

def make_save_memory_tool(store, user_id: str):
    @tool
    def save_memory_tool(user_query: str, reformulated_query: str = "", final_response: str = "") -> str:
        """
        Saves memory to vector store for future retrieval. Stores user query, reformulated version, and final response.
        """
        print("💾 Saving memory...")

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