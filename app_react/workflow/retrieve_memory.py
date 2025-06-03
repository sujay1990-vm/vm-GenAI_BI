from langgraph.store.base import BaseStore
from langchain_core.tools import tool

def make_retrieve_recent_memory_tool(store: BaseStore):
    @tool
    def retrieve_recent_memory(user_query: str, config: Optional[dict] = None) -> str:
        """
        Retrieves the most relevant recent memories (up to 3) for the given user query.
        """
        print("📚 Retrieving recent memory...")
        user_id = config["configurable"]["user_id"]
        namespace = (user_id, "memories")
        recent_memories = store.search(
            namespace,
            query=user_query,
            limit=3
        )

        print(f"🧠 Retrieved {len(recent_memories)} past memories")

        # Join as text block for LLM usage
        memory_str = "\n".join([m.page_content for m in recent_memories])
        print("📥 RAG memory search input:", user_query)
        print("🧠 Namespace:", (user_id, "memories"))
        print("🔍 Retrieved memories:", recent_memories)

        return memory_str or "(no relevant memory found)"
        

    return retrieve_recent_memory_tool