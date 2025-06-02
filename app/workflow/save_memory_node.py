import uuid
from langgraph.store.base import BaseStore

def save_memory_node(state: dict, config: dict, *, store: BaseStore) -> dict:
    print("💾 Saving memory...")

    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")
    memory_id = str(uuid.uuid4())

    # Convert Document objects to just page content for storage
    rag_chunks = [doc.page_content for doc in state.get("rag_outputs", [])]

    memory = {
        "user_query": state.get("user_query", ""),
        "reformulated_query": state.get("reformulated_query", ""),
        "rag_outputs": rag_chunks,
        "final_response": state.get("final_response", "")
    }

    store.put(namespace, memory_id, memory)
    print("✅ Memory saved.")
    return state