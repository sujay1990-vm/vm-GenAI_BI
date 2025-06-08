from langgraph.store.base import BaseStore
from langchain_core.tools import tool
from typing import TypedDict, List, Optional, List, Literal, Annotated
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# def make_retrieve_recent_memory_tool(store: BaseStore, user_id: str):
#     @tool
#     def retrieve_recent_memory(user_query: str) -> str:
#         """
#         Retrieves the most relevant recent memories (up to 3) for the given user query.
#         """
#         print("📚 Retrieving recent memory for user_id:", user_id)
#         namespace = (user_id, "memories")

#         recent_memories = store.search(namespace, query=user_query, limit=3)
#         print(f"🧠 Retrieved {len(recent_memories)} past memories")

#         memory_str = "\n\n".join([
#             f"User: {m.value.get('user_query', '')}\nRewritten: {m.value.get('reformulated_query', '')}\nAnswer: {m.value.get('final_response', '')}"
#             for m in recent_memories
#         ])
#         return memory_str or "(no relevant memory found)"

#     return retrieve_recent_memory


def make_retrieve_memory_node(store, user_id: str):
    def retrieve_memory_node(state: MessagesState):
        print("📚 [Node] Retrieving memory...")

        # Extract user query from the last HumanMessage
        user_query = next(
            (m.content for m in reversed(state["messages"]) if m.type == "human"),
            ""
        )
        print("📚 Retrieving recent memory for user_id:", user_id)
        namespace = (user_id, "memories")
        recent_memories = store.search(namespace, query=user_query, limit=3)
        print(f"🧠 Retrieved {len(recent_memories)} past memories")
        memory_str = "\n\n".join([
                f"- User: {m.value.get('user_query', '')}\n"
                f"- Final Response: {m.value.get('final_response', '')}"
                for m in recent_memories
            ]) or "(no relevant memory found)"

        message = (
            "🧠 The following is past memory retrieved from previous interactions. "
            "Use it for context only. Always validate current schema, metrics, and logic "
            "as memory may be stale or incomplete.\n\n"
            f"{memory_str}"
        )

        return {
            "messages": [SystemMessage(content=message)]
        }


    return retrieve_memory_node





