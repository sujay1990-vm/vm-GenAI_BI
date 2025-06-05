from langchain_core.tools import tool
import json
from llm import get_llm
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

llm = get_llm()

# Structured output schema
class FollowUpSuggestions(BaseModel):
    questions: List[str] = Field(description="3-5 follow-up questions relevant to the claims data")

# System prompt string
followup_prompt_text = """
You are a helpful assistant for claims adjusters working with structured claims data.

Given:
- The user's question: {user_input}
- The conversation history so far: {chat_history}
- The database schema: {schema_str}
- A summary of the data: {table_summary}
- A summary of any failed SQL output or execution issue (if any): {summary}

Suggest 3 to 5 concise follow-up questions the user might logically ask next. Make the questions relevant to claims analysis (e.g., costs, litigation, trends, frequency) and actionable. Format as bullet points.
"""

# Prompt template
followup_prompt = ChatPromptTemplate.from_messages([
    ("system", followup_prompt_text.strip()),
    ("human", "Please suggest helpful follow-up questions.")
])

# Chain
chain_followup_questions = followup_prompt | llm.with_structured_output(
    schema=FollowUpSuggestions,
    method="function_calling"
)

@tool
def suggest_follow_up_questions_tool(user_input: str, chat_history: str = "", schema_str: str = "", table_summary: str = "", summary: str = "") -> str:
    """
    Generates 3-5 concise and helpful follow-up questions based on user query, chat history, schema, and retrieved data summary.
    Intended for claims adjusters to explore claims-related data more effectively.
    """
    print("🔎 Generating follow-up questions...")

    # Run the LLM chain
    result = chain_followup_questions.invoke({
        "user_input": user_input,
        "chat_history": chat_history,
        "schema_str": schema_str,
        "table_summary": table_summary,
        "summary": summary
    })

    print("💡 Suggested follow-ups:\n", result.questions)
    return result.questions

