
from langgraph.store.base import BaseStore

def make_retrieve_recent_memory_tool(store: BaseStore, user_id: str):
    @tool
    def retrieve_recent_memory_tool(user_query: str) -> str:
        """
        Retrieves the most relevant recent memories (up to 3) for the given user query.
        """
        print("📚 Retrieving recent memory...")

        namespace = (user_id, "memories")
        recent_memories = store.search(
            namespace,
            query=user_query,
            limit=3
        )

        print(f"🧠 Retrieved {len(recent_memories)} past memories")

        # Join as text block for LLM usage
        memory_str = "\n".join([m.page_content for m in recent_memories])
        return memory_str or "(no relevant memory found)"

    return retrieve_recent_memory_tool