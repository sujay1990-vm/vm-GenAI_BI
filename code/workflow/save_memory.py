from typing import List, Dict, Any
import re
from langgraph.store.base import BaseStore
import uuid
from .llm import get_embedding_model

embeddings = get_embedding_model()

def sanitize_label(label: str) -> str:
    # Replace any character that is not alphanumeric, hyphen or underscore
    return re.sub(r'[^0-9A-Za-z_-]', '_', label)



def save_memory_node(state: dict, config: dict, *, store: BaseStore) -> dict:
    """
    Saves relevant memory (user query, resolved metrics, SQL) into `state["memory"]`.
    """
    print("💾 Saving memory...")
    user_id = config["configurable"]["user_id"]
    user_label  = sanitize_label(user_id) 
    namespace   = (user_label, "memories")
    memory_id = str(uuid.uuid4())
    # Extract domain_name(s) from resolved_metrics
    resolved_metrics = state.get("resolved_metrics", [])
    domains = list({m.get("domain_name") for m in resolved_metrics if m.get("domain_name")})
    domain_str = ", ".join(domains) if domains else "Unknown"
    memory = {
        "user_query": state.get("user_query", ""),
        "sql_queries": state.get("sql_queries", []),
        "final_response": state.get("final_response", ""),
        "domain_name": domain_str
    }

    store.put(namespace, memory_id, memory)
    print("✅ Memory saved to store.")
    return state
