from langgraph.store.base import BaseStore
from langchain_core.tools import tool

def make_retrieve_recent_memory_tool(store: BaseStore):
    def retrieve_recent_memory(user_query: str, config: Optional[dict] = None) -> str:
        print("📚 Retrieving recent memory...")
        user_id = config["configurable"]["user_id"]
        namespace = (user_id, "memories")
        results = store.search(namespace, query=user_query, limit=3)
        ...
        return formatted_results
    return Tool.from_function(
        retrieve_recent_memory,
        name="retrieve_recent_memory_tool",
        description="Retrieve top 3 past memory results relevant to the user query.",
    )
