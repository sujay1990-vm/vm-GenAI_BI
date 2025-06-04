from langgraph.store.base import BaseStore
from langchain_core.tools import tool
from typing import TypedDict, List, Optional, List, Literal, Annotated

def make_retrieve_recent_memory_tool(store: BaseStore):
    @tool
    def retrieve_recent_memory(**kwargs) -> str:
        print("📚 Retrieving recent memory...")
        config = kwargs.get("config")
        user_query = kwargs.get("user_query", "")

        print("🧠 Tool received config:", config)
        if not config or "configurable" not in config:
            return "❌ Missing config or user_id"

        user_id = config["configurable"]["user_id"]
        namespace = (user_id, "memories")

        recent_memories = store.search(namespace, query=user_query, limit=3)
        print(f"🧠 Retrieved {len(recent_memories)} past memories")

        memory_str = "\n\n".join([
            f"User: {m.get('user_query', '')}\nRewritten: {m.get('reformulated_query', '')}\nAnswer: {m.get('final_response', '')}"
            for m in recent_memories
        ])
        return memory_str or "(no relevant memory found)"

    return retrieve_recent_memory


