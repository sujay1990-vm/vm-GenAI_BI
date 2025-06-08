from langchain_core.tools import tool
import json
from llm import get_llm
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

llm = get_llm()


def extract_context(state):
    sql_outputs = []
    rag_outputs = []
    schema_info = ""
    final_answer = ""

    for m in state["messages"]:
        if m.type in {"ai", "assistant"}:
            final_answer = m.content
        elif m.type == "system":
            if "retrieved memory" in m.content.lower():
                continue  # skip memory
            elif "schema" in m.content.lower():
                schema_info = m.content
            elif "rag" in m.content.lower() or "📄" in m.content:
                rag_outputs.append(m.content)
            elif "sql" in m.content.lower() or "SELECT" in m.content:
                sql_outputs.append(m.content)

    return final_answer, sql_outputs, rag_outputs, schema_info


class FollowUpSuggestions(BaseModel):
    questions: List[str] = Field(description="3-5 follow-up questions relevant to the claims data")

class AnswerConfidence(BaseModel):
    confidence_score: float = Field(..., description="Confidence between 0 and 1")
    reasoning: str = Field(..., description="Explanation of confidence level")



# Just the instruction (no variables here)
followup_prompt = ChatPromptTemplate.from_messages([
    ("system", """
            You are an intelligent assistant helping users explore insurance claims data and policy guidelines.

            The user has just asked a question and received an answer based on structured data, documents, or schema.

            Based on:
            - The user’s latest question
            - The assistant’s answer
            - Any RAG or SQL content retrieved
            - Relevant schema definitions

            Suggest 3 to 5 follow-up questions that would help the user go deeper or explore related patterns.

            Make the questions actionable, investigative, and useful for decision-making.

            Format as bullet points.
            """),
                ("human", """User question: {user_input}
            Assistant answer: {final_answer}

            RAG context:
            {rag_context}

            SQL results:
            {sql_context}

            Schema info:
            {schema}
            """)
            ])

from langchain.prompts import ChatPromptTemplate

confidence_prompt = ChatPromptTemplate.from_messages([
                ("system", """
                    You are an AI assistant evaluating how reliable your final answer is.

                Your goal is to assess whether your answer is grounded in the retrieved SQL or RAG (document) content.

                Do not speculate about dataset quality, timeliness, or missing methodologies. 
                If the answer is directly based on the retrieved content, assign a high confidence (e.g., ≥ 0.9). Only lower the score if the answer goes beyond the retrieved information.

                Return:
                - A confidence score between 0 and 1 (be lenient; prefer high scores if supported by any data).
                - A short explanation that justifies the score, focusing only on support from SQL or RAG context.

                Do not mention whether the dataset is outdated, insufficient, or lacks methodology unless truly unsupported.
                """),
                ("human", """User question: {user_input}
            Assistant answer: {final_answer}

            RAG context:
            {rag_context}

            SQL results:
            {sql_context}

            Schema info:
            {schema}
            """)
            ])

# Prompt template (system + human)
# followup_prompt = ChatPromptTemplate.from_messages([
#     ("system", followup_prompt_text.strip()),
#     ("human", "User question: {user_input}\nConversation history: {chat_history}")
# ])

# Chain
chain_followup_questions = followup_prompt | llm.with_structured_output(
    schema=FollowUpSuggestions,
    method="function_calling"
)

chain_confidence_estimate = followup_prompt | llm.with_structured_output(schema=AnswerConfidence, method="function_calling")

# @tool
# def suggest_follow_up_questions_tool(user_input: str, chat_history: dict) -> str:
#     """
#     Generates 3–5 concise and helpful follow-up questions based on user query and chat history.
#     Intended for claims adjusters to explore claims-related data more effectively.
#     """
#     print("🔎 Generating follow-up questions...")

#     # Run the LLM chain
#     result = chain_followup_questions.invoke({
#         "user_input": user_input,
#         "chat_history": chat_history
#     })

#     formatted = "\n".join(f"- {q}" for q in result.questions)
#     return formatted

# 4. LangGraph node
def make_follow_up_node():
    def follow_up_node(state: MessagesState):
        print("💾 [Node] Follow up questions...")
        messages = state["messages"]

        user_input = next(
            (m.content for m in reversed(messages) if m.type == "human"),
            ""
        )

        final_answer, sql_outputs, rag_outputs, schema_info = extract_context(state)
        # Invoke LLM
        result = chain_followup_questions.invoke({
                "user_input": user_input,
                "final_answer": final_answer,
                "rag_context": "\n\n".join(rag_outputs),
                "sql_context": "\n\n".join(sql_outputs),
                "schema": schema_info
            })
        
        confidence_result = chain_confidence_estimate.invoke({
                "user_input": user_input,
                "final_answer": final_answer,
                "sql_context": "\n\n".join(sql_outputs),
                "rag_context": "\n\n".join(rag_outputs),
                "schema": schema_info
            })


        suggestions = "\n".join(f"- {q}" for q in result.questions)

        if not any([sql_outputs, rag_outputs, schema_info]):
            print("⚠️ No context to generate meaningful follow-up questions.")

        # Display confidence info
        print(f"\n✅ Final Answer Confidence: {confidence_result.confidence_score:.2f}")
        print(f"🧠 Reasoning: {confidence_result.reasoning}")

        # Optionally attach to state if needed for other nodes
        state["answer_confidence"] = {
            "score": confidence_result.confidence_score,
            "reasoning": confidence_result.reasoning
        }

        return {"messages": [
                SystemMessage(content=f"💡 Suggested follow-up questions:\n\n{suggestions}"),
                SystemMessage(content=f"✅ Final Answer Confidence: {confidence_result.confidence_score:.2f}"),
                SystemMessage(content=f"🧠 Reasoning: {confidence_result.reasoning}")
            ]
        }


    return follow_up_node