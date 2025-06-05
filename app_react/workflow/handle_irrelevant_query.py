from langchain_core.tools import tool
import json

@tool
def handle_irrelevant_query(user_query: str) -> str:
    """
    Handles unrelated, vague, or non-data-related questions by informing the user that the assistant is specialized for data queries only.
    """
    keywords = ["joke", "story", "weather", "age", "name", "your", "AI", "robot", "hello", "hi", "who are you", "how are you"]
    lower_query = user_query.lower()

    if any(k in lower_query for k in keywords) or len(lower_query.split()) <= 3:
        return (
            "⚠️ This assistant is designed strictly for answering questions related to your data using tools.\n\n"
            "Please ask a question that involves querying structured data or retrieving document-based insights."
        )
    return (
        "✅ Your question seems relevant. If this tool was triggered incorrectly, please refine your question slightly."
    )
