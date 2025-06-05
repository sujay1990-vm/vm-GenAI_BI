from langchain_core.tools import tool
import json
from llm import get_llm
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

llm = get_llm()

class FollowUpSuggestions(BaseModel):
    questions: List[str] = Field(description="3-5 follow-up questions relevant to the claims data")

# Just the instruction (no variables here)
followup_prompt_text = """
You are a helpful assistant for claims adjusters working with structured claims data.

Given the user's current question and previous conversation, suggest 3 to 5 concise follow-up questions.
Make the questions relevant to claims analysis (e.g., costs, litigation, trends, frequency) and actionable.
Format as bullet points.
"""

# Prompt template (system + human)
followup_prompt = ChatPromptTemplate.from_messages([
    ("system", followup_prompt_text.strip()),
    ("human", "User question: {user_input}\nConversation history: {chat_history}")
])

# Chain
chain_followup_questions = followup_prompt | llm.with_structured_output(
    schema=FollowUpSuggestions,
    method="function_calling"
)

@tool
def suggest_follow_up_questions_tool(user_input: str, chat_history: str = "") -> str:
    """
    Generates 3–5 concise and helpful follow-up questions based on user query and chat history.
    Intended for claims adjusters to explore claims-related data more effectively.
    """
    print("🔎 Generating follow-up questions...")

    # Run the LLM chain
    result = chain_followup_questions.invoke({
        "user_input": user_input,
        "chat_history": chat_history
    })

    formatted = "\n".join(f"- {q}" for q in result.questions)
    print("💡 Suggested follow-ups:\n", formatted)
    return formatted

