from graph import GraphState
from langgraph.store.base import BaseStore

def retrieve_recent_memory_node(state: GraphState, config: dict, *, store: BaseStore) -> dict:
    print("📚 Retrieving recent memory...")

    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")

    recent_memories = store.search(
        namespace,
        query=state["user_query"],
        limit=3
    )

    print(f"🧠 Retrieved {len(recent_memories)} past memories")
    return {"memory": recent_memories}